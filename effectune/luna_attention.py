import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from transformers.utils import logging
logger = logging.get_logger(__name__)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    # mask: 1 for unmasked
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


def init_lisa_params(module):
    std = 0.01
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


def zeros_init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.zero_()
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.zero_()
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


class attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, bias=True, use_output_proj=True, apply_state_layer_norm=None):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.dropout = dropout
        self.dp_k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dp_v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dp_q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.use_output_proj = use_output_proj
        if use_output_proj:
            self.dp_out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.apply_state_layer_norm = apply_state_layer_norm
        if apply_state_layer_norm is not None:
            self.state_layer_norm = nn.LayerNorm(self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, query, key, key_padding_mask=None):
        # decoder_states is query, P is key and value
        # output shape: (bsz, tgt_len, embed_dim)
        if self.apply_state_layer_norm == 'query':
            query = self.state_layer_norm(query)
        elif self.apply_state_layer_norm == 'key':
            key = self.state_layer_norm(key)

        bsz, src_len, embed_dim = key.size()
        tgt_len = query.size(1)

        query_states = self.dp_q_proj(query) * self.scaling
        key_states = self._shape(self.dp_k_proj(key), -1, bsz)
        value_states = self._shape(self.dp_v_proj(key), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if key_padding_mask is not None:
            assert key_padding_mask.size() == (bsz, src_len)
            key_padding_mask = _expand_mask(key_padding_mask, dtype=query.dtype, tgt_len=tgt_len)
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + key_padding_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        if self.use_output_proj:
            attn_output = self.dp_out_proj(attn_output)
        return attn_output


class luna_attention(nn.Module):
    def __init__(self, args, config, embed_dim, num_heads, num_layers=1, bias=True, do_pe_once=False):
        super().__init__()

        self.config = config
        self.p_len = args.preseqlen

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.dropout = args.prefix_dropout

        self.middle_p = nn.Parameter(torch.Tensor(self.p_len, embed_dim))
        nn.init.normal_(self.middle_p, mean=0., std=embed_dim ** -0.5)

        pe_layers = 1 if do_pe_once else num_layers
        # parameters of dp attention can be shared?
        # fixme
        if args.luna_option == "full_layer" or args.luna_option == "full_after":
            self.pe_attns = nn.ModuleList([attention(embed_dim, num_heads, self.dropout, bias, use_output_proj=False, apply_state_layer_norm='key') for _ in range(pe_layers)])
            self.dp_attns = nn.ModuleList([attention(embed_dim, num_heads, self.dropout, bias, apply_state_layer_norm='query') for _ in range(num_layers)])
        else:
            self.pe_attns = nn.ModuleList([attention(embed_dim, num_heads, self.dropout, bias, use_output_proj=False) for _ in range(pe_layers)])
            self.dp_attns = nn.ModuleList([attention(embed_dim, num_heads, self.dropout, bias) for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(self.embed_dim)  # layer norm for P, share?
        self.apply(zeros_init_params)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward_pe(self, encoder_states, encoder_padding_mask=None, p=None, layer_id=0):
        # P is query
        # encoder_states: B x S x d -> key
        # mask: B x S
        bsz, src_len, embed_dim = encoder_states.size()
        if p is None:
            # not the first layer
            expand_p = self.middle_p[None, :, :].expand(bsz, self.p_len, embed_dim)
        else:
            expand_p = p

        residual = expand_p
        attn_output = self.pe_attns[layer_id](expand_p, encoder_states, encoder_padding_mask)
        attn_output = self.layer_norm(attn_output + residual)
        return attn_output

    def forward_dp(self, decoder_states, p_prime, layer_id=0):
        # p_prime is shared across decoder layers
        # decoder_states is query, P is key and value
        # output shape: (bsz, tgt_len, embed_dim)
        # logger.info("layer = {}, p_prime = {}, decoder = {}".format(layer_id, p_prime.size(), decoder_states.size()))

        return self.dp_attns[layer_id](decoder_states, p_prime)

    def forward_bidrectional_dec(self, encoder_states, decoder_states, encoder_padding_mask=None, p=None, dec_layer_id=0):
        p_prime = self.forward_pe(encoder_states, encoder_padding_mask, p, dec_layer_id)
        d_prime = self.forward_dp(decoder_states, p_prime, dec_layer_id)
        return p_prime, d_prime

    def forward_bidrectional_enc(self, encoder_states, encoder_padding_mask=None, p=None, enc_layer_id=0):
        p_prime = self.forward_pe(encoder_states, encoder_padding_mask, p, enc_layer_id)
        e_prime = self.forward_dp(encoder_states, p_prime, enc_layer_id)
        return p_prime, e_prime


class luna_attention_enc_dec(nn.Module):
    def __init__(self, args, config, embed_dim, num_heads, share_params=True, bias=True):
        super().__init__()

        self.config = config
        self.p_len = args.preseqlen

        self.share_params = share_params
        num_enc_layers = num_dec_layers = args.num_bias_layers
        if share_params:
            self.encoder_luna_attns = luna_attention(args, config, embed_dim, num_heads, 1, bias=bias)
            self.decoder_luna_attns = luna_attention(args, config, embed_dim, num_heads, 1, bias=bias, do_pe_once=True)
        else:
            self.encoder_luna_attns = luna_attention(args, config, embed_dim, num_heads, num_enc_layers, bias=bias)
            self.decoder_luna_attns = luna_attention(args, config, embed_dim, num_heads, num_dec_layers, bias=bias, do_pe_once=True)

    def forward_decoder(self, encoder_states, decoder_states, encoder_padding_mask=None, p=None, layer_id=0):
        # used for updating P and not sharing parameters
        if self.share_params:
            layer_id = 0
        p_prime, d_prime = self.decoder_luna_attns.forward_bidrectional_dec(encoder_states, decoder_states, encoder_padding_mask, p, layer_id)
        return p_prime, d_prime

    def forward_encoder(self, encoder_states, encoder_padding_mask, p=None, layer_id=0):
        if self.share_params:
            layer_id = 0
        p_prime, e_prime = self.encoder_luna_attns.forward_bidrectional_enc(encoder_states, encoder_padding_mask, p, layer_id)
        return p_prime, e_prime

    def forward_pe(self, encoder_states, encoder_padding_mask):
        # call it after the encoder, before decoder, and it's done for once
        p_prime = self.decoder_luna_attns.forward_pe(encoder_states, encoder_padding_mask)
        return p_prime

    def forward_dp(self, decoder_states, p_prime, layer_id=0):
        if self.share_params:
            layer_id = 0
        d_prime = self.decoder_luna_attns.forward_dp(decoder_states, p_prime, layer_id)
        return d_prime


class SimpleAttnBias(nn.Module):
    def __init__(self, args, config, embed_dim, num_heads, bias=True):
        super().__init__()
        self.config = config

        # self.match_n_layer = config.decoder_layers if args.num_bias_layers < 0 else args.num_bias_layers
        self.match_n_layer = args.num_bias_layers
        self.match_n_head = config.decoder_attention_heads
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = args.mid_dim
        self.preseqlen = args.preseqlen
        self.prefix_dropout = args.prefix_dropout

        self.dropout = nn.Dropout(self.prefix_dropout)
        self.P = nn.Embedding(self.preseqlen, embed_dim)
        # attention from P to input embedding
        self.pe_attn = attention(embed_dim, num_heads, self.prefix_dropout, bias, use_output_proj=True)
        self.pds_attn = attention(embed_dim, num_heads, self.prefix_dropout, bias, use_output_proj=True)
        self.pdc_attn = attention(embed_dim, num_heads, self.prefix_dropout, bias, use_output_proj=True)

        self.input_tokens = torch.arange(self.preseqlen).long()
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.control_trans2 = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.beam_size = config.num_beams
        self.apply(init_lisa_params)

        self.biases = None

    def forward_pe_attn(self, inputs_embeds, inputs_masks):
        bsz = inputs_embeds.size(0)
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(inputs_embeds.device)
        p = self.P(input_tokens)
        p_prime_e = self.pe_attn(p, inputs_embeds, inputs_masks)
        p_prime_ds = self.pds_attn(p, inputs_embeds, inputs_masks)
        p_prime_dc = self.pdc_attn(p, inputs_embeds, inputs_masks)
        return p_prime_e, p_prime_ds, p_prime_dc

    def forward(self, p_prime_e, p_prime_ds, p_prime_dc, layer_idx=0):
        # p_prime: bsz x p_len x dim
        bsz = p_prime_e.size(0)
        old_bsz = bsz
        seqlen = self.preseqlen

        if not self.training:
            expanded_return_idx = (
                torch.arange(bsz).view(-1, 1).repeat(1, self.beam_size).view(-1).to(p_prime_e.device)
            )

        past_key_values = self.control_trans(p_prime_ds) # bsz, seqlen, layer*emb
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        if not self.training:
            past_key_values = past_key_values.index_select(0, expanded_return_idx)

        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        past_key_values2 = self.control_trans2(p_prime_dc)  # bsz, seqlen, layer*emb
        past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                 self.match_n_embd)
        if not self.training:
            past_key_values2 = past_key_values2.index_select(0, expanded_return_idx)

        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)

        past_key_values_enc = self.control_trans_enc(p_prime_e)  # bsz, seqlen, layer*emb
        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                       self.match_n_embd)
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        if not self.training:
            bsz = bsz * self.beam_size

        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {'self': {"prev_key": key_val[0].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                  "prev_value": key_val[1].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                  "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device) #bsz, preseqlen
                                  },
                         }

            key_val2 = past_key_values2[i]
            temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                            "prev_value": key_val2[1].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                            "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device)
                                            }
            key_val_enc = past_key_values_enc[i]
            # at generation time, this is expanded automatically to the beam size
            temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous().view(old_bsz*self.match_n_head, -1, self.match_n_embd),
                                    "prev_value": key_val_enc[1].contiguous().view(old_bsz*self.match_n_head, -1, self.match_n_embd),
                                    "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(key_val_enc.device)
                                    }
            result.append(temp_dict)

        return result

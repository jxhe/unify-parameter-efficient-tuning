import torch
import torch.nn as nn
from transformers.utils import logging
logger = logging.get_logger(__name__)


class luna_attention(nn.Module):
    def __init__(self, args, config, embed_dim, num_heads, bias=True):
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

        self.pe_k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.pe_v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.pe_q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dp_k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dp_v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dp_q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dp_out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _init_weights(self):
        logger.info("=============init weights! ==============")
        std = self.config.init_std
        for n, module in self.named_parameters():
            if n == "middle_p":
                continue
            logger.info("init " + n)

            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward_pe(self, encoder_states, encoder_padding_mask=None):
        # P is query
        # encoder_states: B x S x d -> key
        # mask: B x S
        bsz, src_len, embed_dim = encoder_states.size()
        tgt_len = self.p_len

        expand_p = self.middle_p[None, :, :].expand(bsz, self.p_len, embed_dim)
        query_states = self.pe_q_proj(expand_p) * self.scaling
        key_states = self._shape(self.pe_k_proj(encoder_states), -1, bsz)
        value_states = self._shape(self.pe_v_proj(encoder_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
        )

        if encoder_padding_mask is not None:
            expanded_mask = encoder_padding_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(encoder_padding_mask.dtype)
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + expanded_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        return attn_output

    def forward_dp(self, decoder_states, p_prime):
        # decoder_states is query, P is key and value
        # output shape: (bsz, tgt_len, embed_dim)
        bsz, src_len, embed_dim = p_prime.size()
        tgt_len = decoder_states.size(1)

        query_states = self.dp_q_proj(decoder_states) * self.scaling
        key_states = self._shape(self.dp_k_proj(p_prime), -1, bsz)
        value_states = self._shape(self.pe_v_proj(p_prime), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
        )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.dp_out_proj(attn_output)
        return attn_output
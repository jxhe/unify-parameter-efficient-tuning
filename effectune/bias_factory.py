import torch
import torch.nn as nn


def init_lisa_params(module):
    std = 0.02
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


def init_bias_mlp(module):
    std = 1e-3
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
    if module.bias is not None:
        module.bias.data.zero_()


class Prefix(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        self.match_n_layer = config.decoder_layers
        self.match_n_head = config.decoder_attention_heads
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = args.mid_dim
        self.preseqlen = args.preseqlen
        self.prefix_dropout = args.prefix_dropout

        self.dropout = nn.Dropout(self.prefix_dropout)

        self.input_tokens = torch.arange(self.preseqlen).long()
        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.wte2 = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans2 = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        self.apply(init_lisa_params)

    def forward(self, bsz, nsamples=1, device="gpu"):
        old_bsz = bsz
        bsz = bsz * nsamples
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        temp_control2 = self.wte2(input_tokens)
        past_key_values2 = self.control_trans2(temp_control2)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values2.shape
        past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                 self.match_n_embd)
        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)

        input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(device)
        temp_control_enc = self.wte_enc(input_tokens_enc)
        past_key_values_enc = self.control_trans_enc(temp_control_enc)  # bsz, seqlen, layer*emb
        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                       self.match_n_embd)
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

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


class MLP_Bias(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        self.match_n_layer = config.decoder_layers
        self.match_n_head = config.decoder_attention_heads
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = args.mid_dim
        self.preseqlen = args.preseqlen
        self.prefix_dropout = args.prefix_dropout

        self.dropout = nn.Dropout(self.prefix_dropout)

        self.src_len = args.max_source_length + 2
        self.tgt_len = args.max_target_length + 2
        self.tgt_input_tokens = torch.arange(self.tgt_len).long()
        self.src_input_tokens = torch.arange(self.src_len).long()

        self.wte = nn.Embedding(self.tgt_len, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * self.n_embd))

        self.wte_enc = nn.Embedding(self.src_len, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * self.n_embd))

        self.wte2 = nn.Embedding(self.tgt_len, self.n_embd)
        self.control_trans2 = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * self.n_embd))

        self.apply(init_bias_mlp)
        # initialization
        nn.init.constant_(self.wte.weight, 0.0)
        nn.init.constant_(self.wte_enc.weight, 0.0)
        nn.init.constant_(self.wte2.weight, 0.0)

    def forward(self, bsz, nsamples=1, device="gpu"):
        temp_control = self.wte(self.tgt_input_tokens.to(self.device))
        past_key_values = self.control_trans(temp_control)  # tgt_len, layer*emb
        past_key_values = past_key_values.view(self.tgt_len, self.match_n_layer, self.n_embd)
        past_key_values = self.dropout(past_key_values)

        temp_control2 = self.wte2(self.tgt_input_tokens.to(self.device))
        past_key_values2 = self.control_trans2(temp_control2)  # tgt_len, layer*emb
        past_key_values2 = past_key_values2.view(self.tgt_len, self.match_n_layer, self.n_embd)
        past_key_values2 = self.dropout(past_key_values2)

        temp_control_enc = self.wte_enc(self.src_input_tokens.to(self.device))
        past_key_values_enc = self.control_trans_enc(temp_control_enc)  # src_len, layer*emb
        past_key_values_enc = past_key_values_enc.view(self.src_len, self.match_n_layer, self.n_embd)
        past_key_values_enc = self.dropout(past_key_values_enc)

        result = []
        for ii in range(self.match_n_layer):
            temp_dict = {"encoder": past_key_values_enc[:, ii, :],
                         "self": past_key_values[:, ii, :],
                         "encoder_decoder": past_key_values2[:, ii, :]}
            result.append(temp_dict)
        return result


class Bias(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Option 1: a simple version, no transformations, each attention layer has its own bias parameters
        self.encoder_attn_bias = nn.ModuleList([nn.Embedding(args.max_source_length + 2, self.n_embd)
                                                for _ in range(self.match_n_layer)])
        self.decoder_self_attn_bias = nn.ModuleList([nn.Embedding(args.max_target_length + 2, self.n_embd)
                                                     for _ in range(self.match_n_layer)])

        self.decoder_cross_attn_bias = nn.ModuleList([nn.Embedding(args.max_target_length + 2, self.n_embd)
                                                      for _ in range(self.match_n_layer)])
        for embed in self.encoder_attn_bias:
            assert isinstance(embed, nn.Embedding)
            nn.init.constant_(embed.weight, 0.0)
        for embed in self.decoder_self_attn_bias:
            assert isinstance(embed, nn.Embedding)
            nn.init.constant_(embed.weight, 0.0)
        for embed in self.decoder_cross_attn_bias:
            assert isinstance(embed, nn.Embedding)
            nn.init.constant_(embed.weight, 0.0)

    def forward(self, bsz, nsamples=1, device="gpu"):
        result = []
        max_src_len = self.args.max_source_length + 2
        max_tgt_len = self.args.max_target_length + 2

        src_positions = torch.arange(0, max_src_len, dtype=torch.long, device=device)
        tgt_positions = torch.arange(0, max_tgt_len, dtype=torch.long, device=device)
        for ii in range(self.match_n_layer):
            temp_dict = {"encoder": self.encoder_attn_bias[ii].forward(src_positions),
                         "self": self.decoder_self_attn_bias[ii].forward(tgt_positions),
                         "encoder_decoder": self.decoder_cross_attn_bias[ii].forward(tgt_positions)}
            result.append(temp_dict)
        return result
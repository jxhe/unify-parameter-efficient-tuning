import torch
import torch.nn as nn


def init_lisa_params(module):
    std = 1e-20
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


def init_bias_mlp(module):
    std = 1e-2
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()


def init_bert_weights(module):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # std defaults to 0.02, this might need to be changed
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def init_zero_weights(module):
    if isinstance(module, nn.Embedding):
        nn.init.constant_(module.weight, 0.0)


class Prefix(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        # self.match_n_layer = config.decoder_layers if args.num_bias_layers < 0 else args.num_bias_layers
        self.match_n_layer = args.num_bias_layers
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

        # if args.lisa_option == "cross_attn":
        #     self.apply(init_lisa_params)

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


class PrefixDirectInit(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        # self.match_n_layer = config.decoder_layers if args.num_bias_layers < 0 else args.num_bias_layers
        self.match_n_layer = args.num_bias_layers
        self.match_n_head = config.decoder_attention_heads
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = args.mid_dim
        self.preseqlen = args.preseqlen
        self.prefix_dropout = args.prefix_dropout

        self.dropout = nn.Dropout(self.prefix_dropout)
        self.input_tokens = torch.arange(self.preseqlen).long()
        self.encoder_attn_key = nn.ModuleList([nn.Embedding(self.preseqlen, self.n_embd)
                                                for _ in range(self.match_n_layer)])
        self.encoder_attn_value = nn.ModuleList([nn.Embedding(self.preseqlen, self.n_embd)
                                               for _ in range(self.match_n_layer)])
        self.decoder_self_attn_key = nn.ModuleList([nn.Embedding(self.preseqlen, self.n_embd)
                                                     for _ in range(self.match_n_layer)])
        self.decoder_self_attn_value = nn.ModuleList([nn.Embedding(self.preseqlen, self.n_embd)
                                                    for _ in range(self.match_n_layer)])

        self.decoder_cross_attn_key = nn.ModuleList([nn.Embedding(self.preseqlen, self.n_embd)
                                                      for _ in range(self.match_n_layer)])
        self.decoder_cross_attn_value = nn.ModuleList([nn.Embedding(self.preseqlen, self.n_embd)
                                                     for _ in range(self.match_n_layer)])

        # fixme: choose a favorable init method
        self.apply(init_bert_weights)
        # self.apply(init_zero_weights)

        # lisa mlp init
        # wte = nn.Embedding(self.preseqlen, self.n_embd)
        # control_trans = nn.Sequential(
        #     nn.Linear(self.n_embd, self.mid_dim),
        #     nn.Tanh(),
        #     nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        # wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
        # control_trans_enc = nn.Sequential(
        #     nn.Linear(self.n_embd, self.mid_dim),
        #     nn.Tanh(),
        #     nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        # wte2 = nn.Embedding(self.preseqlen, self.n_embd)
        # control_trans2 = nn.Sequential(
        #     nn.Linear(self.n_embd, self.mid_dim),
        #     nn.Tanh(),
        #     nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd))

        # def init_mlp_lisa_local(key_module, val_module, wte_t, control_t):
        #     key_val = control_t(wte_t(self.input_tokens))
        #     seqlen, _ = key_val.shape
        #     key_val = key_val.view(seqlen, self.match_n_layer * 2, self.n_embd)
        #     key_val = key_val.permute([1, 0, 2]).split(2)

        #     for i in range(self.match_n_layer):
        #         key_module[i].weight.data = key_val[i][0].data
        #         val_module[i].weight.data = key_val[i][1].data

        # # import pdb; pdb.set_trace()
        # init_mlp_lisa_local(self.decoder_self_attn_key, self.decoder_self_attn_value,
        #     wte, control_trans)

        # init_mlp_lisa_local(self.encoder_attn_key, self.encoder_attn_value,
        #     wte_enc, control_trans_enc)

        # init_mlp_lisa_local(self.decoder_cross_attn_key, self.decoder_cross_attn_value,
        #     wte2, control_trans2)



    def _shape(self, x, bsz):
        y = x.view(bsz, self.preseqlen, self.match_n_head, self.match_n_embd)
        y = y.permute([0, 2, 1, 3])
        y = y.contiguous().view(bsz * self.match_n_head, -1, self.match_n_embd)
        return y

    def forward(self, bsz, nsamples=1, device="cuda"):
        old_bsz = bsz
        bsz = bsz * nsamples
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(device)
        input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(device)

        result = []
        for i, (enc_attn_k, enc_attn_v, dec_self_attn_k, dec_self_attn_v, dec_xattn_k, dec_xattn_v) in \
                enumerate(zip(self.encoder_attn_key, self.encoder_attn_value, self.decoder_self_attn_key,
                              self.decoder_self_attn_value, self.decoder_cross_attn_key, self.decoder_cross_attn_value)):
            temp_dict = {'self': {"prev_key": self._shape(dec_self_attn_k(input_tokens), bsz),
                                  "prev_value": self._shape(dec_self_attn_v(input_tokens), bsz),
                                  "prev_key_padding_mask": torch.zeros(bsz, self.preseqlen).to(device) #bsz, preseqlen
                                  },
                         'encoder_decoder': {"prev_key": self._shape(dec_xattn_k(input_tokens), bsz),
                                  "prev_value": self._shape(dec_xattn_v(input_tokens), bsz),
                                  "prev_key_padding_mask": torch.zeros(bsz, self.preseqlen).to(device)  #bsz, preseqlen
                                  },
                         'encoder': {"prev_key": self._shape(enc_attn_k(input_tokens_enc), old_bsz),
                                  "prev_value": self._shape(enc_attn_v(input_tokens_enc), old_bsz),
                                  "prev_key_padding_mask": torch.zeros(old_bsz, self.preseqlen).to(device) #bsz, preseqlen
                                  },
                        }
            result.append(temp_dict)
        return result


class MLP_Bias(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        self.match_n_layer = config.decoder_layers if args.num_bias_layers < 0 else args.num_bias_layers
        self.match_n_head = config.decoder_attention_heads
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        self.mid_dim = args.mid_dim
        self.preseqlen = args.preseqlen
        self.prefix_dropout = args.prefix_dropout

        self.dropout = nn.Dropout(self.prefix_dropout)

        self.src_len = config.max_source_length + 2
        self.tgt_len = config.max_target_length + 2
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
        # # initialization
        # nn.init.constant_(self.wte.weight, 0.0)
        # nn.init.constant_(self.wte_enc.weight, 0.0)
        # nn.init.constant_(self.wte2.weight, 0.0)

    def forward(self, bsz, nsamples=1, device="cuda"):
        temp_control = self.wte(self.tgt_input_tokens.to(device))
        past_key_values = self.control_trans(temp_control)  # tgt_len, layer*emb
        past_key_values = past_key_values.view(self.tgt_len, self.match_n_layer, self.n_embd)
        past_key_values = self.dropout(past_key_values)

        temp_control2 = self.wte2(self.tgt_input_tokens.to(device))
        past_key_values2 = self.control_trans2(temp_control2)  # tgt_len, layer*emb
        past_key_values2 = past_key_values2.view(self.tgt_len, self.match_n_layer, self.n_embd)
        past_key_values2 = self.dropout(past_key_values2)

        temp_control_enc = self.wte_enc(self.src_input_tokens.to(device))
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
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.match_n_layer = config.decoder_layers if args.num_bias_layers < 0 else args.num_bias_layers
        self.n_embd = config.d_model

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

    def forward(self, bsz, nsamples=1, device="cuda"):
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
        return


class Adapter_Layer(nn.Module):

    def __init__(self, 
                 config=None,
                 d_model=None,
                 bottleneck=None, 
                 dropout=0.0,
                 init_with_bert=True):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.preseqlen if bottleneck is None else bottleneck
        # self.non_linearity = args.non_linearity  # use ReLU by default

        self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout

        if init_with_bert:
            self.apply(init_bert_weights)


    def forward(self, x, add_residual=True):
        residual = x
        down = self.non_linear_func(self.down_proj(self.adapter_layer_norm_before(x)))
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class Adapter(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.num_layers = args.num_bias_layers
        self.encoder_adapters = nn.ModuleList([Adapter_Layer(config) for _ in range(args.num_bias_layers)])
        self.decoder_adapters = nn.ModuleList([Adapter_Layer(config) for _ in range(args.num_bias_layers)])

    def forward(self, bsz, nsamples=1, device="cuda"):
        results = []
        for ii in range(self.num_layers):
            results.append({"encoder_ffn_adapters": self.encoder_adapters[ii],
                            "decoder_ffn_adapters": self.decoder_adapters[ii]})
        return results


class Prefix_Adapter(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.prefix = Prefix(args, config)
        self.adapters = Adapter(args, config)

    def forward(self, bsz, nsamples=1, device="cuda"):
        prefix = self.prefix(bsz, nsamples, device)
        adapters = self.adapters(bsz)

        for ii, dic in enumerate(adapters):
            for key, value in dic.items():
                prefix[ii][key] = value
        return prefix

# class InputBias(nn.Module):
    # def __init__(self, args):
    #     super().__init__()
    #     self.args = args
    #
    #     self.prefix_embs = nn.Embedding(args.max_source_length + 2, self.n_embd)
    #     # Option 1: a simple version, no transformations, each attention layer has its own bias parameters
    #     self.encoder_attn_bias = nn.ModuleList([nn.Embedding(args.max_source_length + 2, self.n_embd)
    #                                             for _ in range(self.match_n_layer)])
    #     self.decoder_self_attn_bias = nn.ModuleList([nn.Embedding(args.max_target_length + 2, self.n_embd)
    #                                                  for _ in range(self.match_n_layer)])
    #
    #     self.decoder_cross_attn_bias = nn.ModuleList([nn.Embedding(args.max_target_length + 2, self.n_embd)
    #                                                   for _ in range(self.match_n_layer)])
    #     for embed in self.encoder_attn_bias:
    #         assert isinstance(embed, nn.Embedding)
    #         nn.init.constant_(embed.weight, 0.0)
    #     for embed in self.decoder_self_attn_bias:
    #         assert isinstance(embed, nn.Embedding)
    #         nn.init.constant_(embed.weight, 0.0)
    #     for embed in self.decoder_cross_attn_bias:
    #         assert isinstance(embed, nn.Embedding)
    #         nn.init.constant_(embed.weight, 0.0)
    #
    # def forward(self, bsz, nsamples=1, device="gpu"):
    #     result = []
    #     max_src_len = self.args.max_source_length + 2
    #     max_tgt_len = self.args.max_target_length + 2
    #
    #     src_positions = torch.arange(0, max_src_len, dtype=torch.long, device=device)
    #     tgt_positions = torch.arange(0, max_tgt_len, dtype=torch.long, device=device)
    #     for ii in range(self.match_n_layer):
    #         temp_dict = {"encoder": self.encoder_attn_bias[ii].forward(src_positions),
    #                      "self": self.decoder_self_attn_bias[ii].forward(tgt_positions),
    #                      "encoder_decoder": self.decoder_cross_attn_bias[ii].forward(tgt_positions)}
    #         result.append(temp_dict)
    #     return result


import torch
from transformers import PretrainedBartModel
import torch.nn as nn
from transformers.utils import logging
logger = logging.get_logger(__name__)


class PrefixTuning(PretrainedBartModel):
    def __init__(self, config, args, pretrained_model):
        super().__init__(config)
        self.args = args
        self.seq2seq_model = pretrained_model

        self.match_n_layer = config.decoder_layers
        self.match_n_head = config.decoder_attention_heads
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        if args.use_prefix == "lisa":
            self.setup_lisa(args)
        elif args.use_prefix == "learn_bias":
            self.setup_bias(args)

        self._init_weights()
        logger.info("Declare PrefixTuning model!")
        not_freeze_set = []
        if args.unfreeze_params != 'none':
            if args.unfreeze_params == 'LN':
                # not_freeze_set = ['layernorm']  # input layernorm
                not_freeze_set = ['attn_layer_norm']  # only optimize layer norm after attn
        logger.info(not_freeze_set)
        for n, p in self.seq2seq_model.named_parameters():
            if len(not_freeze_set) > 0:
                for nfs in not_freeze_set:
                    if nfs in n:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            else:
                p.requires_grad = False
        logger.info("already freezed parameters!")

    def _init_weights(self):
        logger.info("=============init weights! ==============")
        std = self.config.init_std
        for n, module in self.named_parameters():
            if n.startswith("seq2seq_model"):
                continue
            # print(n, module)
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def setup_lisa(self, args):
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

        self.get_prompt = self.get_prompt_lisa

    def get_prompt_lisa(self, bsz, nsamples=1):
        old_bsz = bsz
        bsz = bsz * nsamples
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
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

        input_tokens_enc = self.input_tokens.unsqueeze(0).expand(old_bsz, -1).to(self.device)
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
                                  "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device).bool() #bsz, preseqlen
                                  },
                         }

            key_val2 = past_key_values2[i]
            temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                            "prev_value": key_val2[1].contiguous().view(bsz*self.match_n_head, -1, self.match_n_embd),
                                            "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device).bool()
                                            }
            key_val_enc = past_key_values_enc[i]
            # at generation time, this is expanded automatically to the beam size
            temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous().view(old_bsz*self.match_n_head, -1, self.match_n_embd),
                                    "prev_value": key_val_enc[1].contiguous().view(old_bsz*self.match_n_head, -1, self.match_n_embd),
                                    "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).to(key_val_enc.device).bool()
                                    }
            result.append(temp_dict)

        return result

    def setup_bias(self, args):
        # Option 1: a simple version, no transformations, each attention layer has its own bias parameters
        self.encoder_attn_bias = nn.ModuleList([nn.Embedding(args.max_source_length + 2, self.n_embd)
                                                  for _ in range(self.match_n_layer)])
        self.decoder_self_attn_bias = nn.ModuleList([nn.Embedding(args.max_target_length + 2, self.n_embd)
                                                    for _ in range(self.match_n_layer)])

        self.decoder_cross_attn_bias = nn.ModuleList([nn.Embedding(args.max_target_length + 2, self.n_embd)
                                                    for _ in range(self.match_n_layer)])
        self.get_prompt = self.get_prompt_bias

    def get_prompt_bias(self, bsz, nsamples=1):
        result = []
        max_src_len = self.args.max_source_length + 2
        max_tgt_len = self.args.max_target_length + 2

        src_positions = torch.arange(0, max_src_len, dtype=torch.long, device=self.device)
        tgt_positions = torch.arange(0, max_tgt_len, dtype=torch.long, device=self.device)
        for ii in range(self.match_n_layer):
            temp_dict = {"encoder": self.encoder_attn_bias[ii].forward(src_positions),
                         "self": self.decoder_self_attn_bias[ii].forward(tgt_positions),
                         "encoder_decoder": self.decoder_cross_attn_bias[ii].forward(tgt_positions)}
            result.append(temp_dict)
        return result

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                cross_attn_head_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs,):

        bsz = input_ids.shape[0]
        prefix_state = self.get_prompt(bsz=bsz)

        output = self.seq2seq_model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    decoder_input_ids=decoder_input_ids,
                                    decoder_attention_mask=decoder_attention_mask,
                                    head_mask=head_mask,
                                    decoder_head_mask=decoder_head_mask,
                                    cross_attn_head_mask=cross_attn_head_mask,
                                    encoder_outputs=encoder_outputs,
                                    past_key_values=past_key_values,
                                    inputs_embeds=inputs_embeds,
                                    decoder_inputs_embeds=decoder_inputs_embeds,
                                    labels=labels,
                                    use_cache=use_cache,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict,
                                    prefix_state=prefix_state,
                                    **kwargs)
        return output
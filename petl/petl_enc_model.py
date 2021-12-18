import torch
from transformers import MBartPreTrainedModel, RobertaConfig
import torch.nn as nn
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from petl.petl_factory import Prefix, MLP_Bias, Bias, PrefixDirectInit, PrefixCrossAttn
from transformers.utils import logging
logger = logging.get_logger(__name__)


class PETLEncModel(MBartPreTrainedModel):
    def __init__(self, config, args, pretrained_model, **kwargs):
        super().__init__(config)
        self.args = args
        self.pretrained_model = pretrained_model

        if isinstance(config, RobertaConfig):
            self.match_n_layer = config.num_hidden_layers
            self.match_n_head = config.num_attention_heads
            self.n_embd = config.hidden_size
        else:
            self.match_n_layer = config.decoder_layers
            self.match_n_head = config.decoder_attention_heads
            self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head

        if "prefix" in args.attn_mode:
            self.setup_prefix(args, config)
        elif args.attn_mode == 'bitfit' or args.attn_mode == 'adapter':
            self.get_prompt = self.get_fake_prompt
        elif args.attn_mode == 'none':
            # includes only with ffn mode
            self.get_prompt = self.get_fake_prompt
        elif args.attn_mode == "prompt_tuning":
            self.get_prompt = self.get_fake_prompt
        elif args.attn_mode == "lora":
            self.get_prompt = self.get_fake_prompt
        else:
            raise ValueError

        logger.info("Declare PrefixTuning model!")

        not_freeze_set = []
        if args.unfreeze_params != 'none' and args.attn_mode != 'bitfit':
            if args.unfreeze_params == 'LN':
                # not_freeze_set = ['layernorm']  # input layernorm
                not_freeze_set = ['attn_layer_norm']  # only optimize layer norm after attn
            else:
                not_freeze_set = args.unfreeze_params.split(',')
            all_match = False
        elif args.attn_mode == 'bitfit':
            not_freeze_set = ['bias']
            all_match = True

        logger.info(not_freeze_set)

        freeze_set = []
        if args.ffn_mode == 'mh_adapter_random' or args.attn_option == 'mh_adapter':
            # freeze the random mapping matrix
            freeze_set = ['freeze_q_proj']

        for n, p in self.pretrained_model.named_parameters():
            if len(not_freeze_set) > 0 and self.check_params(n, not_freeze_set, all_match=all_match):
                print("tune "+ n)
                p.requires_grad = True
            else:
                p.requires_grad = False

            if len(freeze_set) > 0 and self.check_params(n, freeze_set, all_match=False):
                p.requires_grad = False

        logger.info("already freezed parameters!")

    def check_params(self, module_name, safe_list, all_match=True):
        check = [partial_name in module_name for partial_name in safe_list]
        return all(check) if all_match else any(check)

    def get_standard_prompt(self, bsz, nsamples=1):
        return self.prompt_model(bsz, nsamples, self.device)

    def setup_prefix(self, args, config):
        if args.attn_mode == "prefix_nomlp":
            self.prompt_model = PrefixDirectInit(args, config)
        else:
            self.prompt_model = Prefix(args, config)
        self.get_prompt = self.get_standard_prompt

    def setup_bias(self, args, config):
        self.prompt_model = Bias(args, config)
        self.get_prompt = self.get_standard_prompt

    def setup_bias_mlp(self, args, config):
        self.prompt_model = MLP_Bias(args, config)
        self.get_prompt = self.get_standard_prompt

    def get_fake_prompt(self, bsz, nsamples=-1):
        return None

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):

        bsz = input_ids.shape[0]
        prefix_state = self.get_prompt(bsz=bsz)

        output = self.pretrained_model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    head_mask=head_mask,
                                    inputs_embeds=inputs_embeds,
                                    labels=labels,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict,
                                    prefix_state=prefix_state,
                                    )
        return output

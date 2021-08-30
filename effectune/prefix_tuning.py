import torch
from transformers import PretrainedBartModel
import torch.nn as nn
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from effectune.luna_attention import luna_attention, luna_attention_enc_dec, SimpleAttnBias
from effectune.bias_factory import Prefix, MLP_Bias, Bias, Prefix_Adapter, PrefixDirectInit
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

        if "lisa" in args.attn_mode:
            self.setup_lisa(args, config)
        elif args.attn_mode == "learn_bias":
            # self.setup_bias(args, config)
            self.setup_bias_mlp(args, config)
        elif args.attn_mode == 'luna':
            self.setup_luna(args)
        elif args.attn_mode == 'dlisa':
            self.setup_dependent_lisa(args, config)
        elif args.attn_mode == 'bitfit' or args.attn_mode == 'adapter':
            self.get_prompt = self.get_fake_prompt

        logger.info("Declare PrefixTuning model!")

        not_freeze_set = []
        if args.unfreeze_params != 'none' and args.attn_mode != 'luna':
            if args.unfreeze_params == 'LN':
                # not_freeze_set = ['layernorm']  # input layernorm
                not_freeze_set = ['attn_layer_norm']  # only optimize layer norm after attn
            else:
                not_freeze_set = args.unfreeze_params.split(',')

            all_match = False
        elif args.attn_mode == 'luna':
            # fixme: other options, now tune the self_attn_layer_norm in decoder
            # not_freeze_set = ["decoder.layers.1.self_attn_layer_norm"]
            # not_freeze_set = ['attn_layer_norm', 'decoder']
            not_freeze_set = ['layer_norm']
            all_match = True
        elif args.attn_mode == 'bitfit':
            not_freeze_set = ['bias']
            all_match = True

        logger.info(not_freeze_set)
        for n, p in self.seq2seq_model.named_parameters():
            if len(not_freeze_set) > 0 and self.if_update_params(n, not_freeze_set, all_match=all_match):
                print("tune "+ n)
                p.requires_grad = True
            else:
                p.requires_grad = False
        logger.info("already freezed parameters!")

    def if_update_params(self, module_name, safe_list, all_match=True):
        check = [partial_name in module_name for partial_name in safe_list]
        return all(check) if all_match else any(check)

    def get_standard_prompt(self, bsz, nsamples=1):
        return self.prompt_model(bsz, nsamples, self.device)

    def setup_lisa(self, args, config):
        if args.attn_mode == "lisa_nomlp":
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

    def setup_luna(self, args):
        if args.luna_option == "full_before" or args.luna_option == "full_after":
            self.luna_attn = luna_attention_enc_dec(args, self.config, self.n_embd, self.match_n_head, share_params=args.share_luna_params)
        else:
            self.luna_attn = luna_attention(args, self.config, self.n_embd, self.match_n_head, args.num_bias_layers)
        self.get_prompt = self.get_prompt_luna_bias

    def get_prompt_luna_bias(self, bsz, nsamples=-1):
        return self.luna_attn

    def setup_dependent_lisa(self, args, config):
        self.bias_model = SimpleAttnBias(args, config, self.n_embd, self.match_n_head)
        self.get_prompt = self.get_prompt_dependent_lisa

    def get_prompt_dependent_lisa(self, bsz, nsamples=-1):
        return self.bias_model

    def get_fake_prompt(self, bsz, nsamples=-1):
        return None

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

    def generate(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            max_length: Optional[int] = None,
            min_length: Optional[int] = None,
            do_sample: Optional[bool] = None,
            early_stopping: Optional[bool] = None,
            num_beams: Optional[int] = None,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            bad_words_ids: Optional[Iterable[int]] = None,
            bos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            length_penalty: Optional[float] = None,
            no_repeat_ngram_size: Optional[int] = None,
            encoder_no_repeat_ngram_size: Optional[int] = None,
            num_return_sequences: Optional[int] = None,
            max_time: Optional[float] = None,
            max_new_tokens: Optional[int] = None,
            decoder_start_token_id: Optional[int] = None,
            use_cache: Optional[bool] = None,
            num_beam_groups: Optional[int] = None,
            diversity_penalty: Optional[float] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            forced_bos_token_id: Optional[int] = None,
            forced_eos_token_id: Optional[int] = None,
            remove_invalid_values: Optional[bool] = None,
            synced_gpus: Optional[bool] = None,
            override: Optional[Dict] = None, # added by Junxian
            **model_kwargs,
    ):
        prefix_state = self.get_prompt(input_ids.size(0), num_beams)
        generated_tokens = self.seq2seq_model.generate(input_ids=input_ids,
                                                       max_length=max_length,
                                                       min_length=min_length,
                                                       do_sample=do_sample,
                                                       early_stopping=early_stopping,
                                                       num_beams=num_beams,
                                                       temperature=temperature,
                                                       top_k=top_k,
                                                       top_p=top_p,
                                                       repetition_penalty=repetition_penalty,
                                                       bad_words_ids=bad_words_ids,
                                                       bos_token_id=bos_token_id,
                                                       pad_token_id=pad_token_id,
                                                       eos_token_id=eos_token_id,
                                                       length_penalty=length_penalty,
                                                       no_repeat_ngram_size=no_repeat_ngram_size,
                                                       encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
                                                       num_return_sequences=num_return_sequences,
                                                       max_time=max_time,
                                                       max_new_tokens=max_new_tokens,
                                                       decoder_start_token_id=decoder_start_token_id,
                                                       use_cache=use_cache,
                                                       num_beam_groups=num_beam_groups,
                                                       diversity_penalty=diversity_penalty,
                                                       prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                                       output_attentions=output_attentions,
                                                       output_hidden_states=output_hidden_states,
                                                       output_scores=output_scores,
                                                       return_dict_in_generate=return_dict_in_generate,
                                                       forced_bos_token_id=forced_bos_token_id,
                                                       forced_eos_token_id=forced_eos_token_id,
                                                       remove_invalid_values=remove_invalid_values,
                                                       synced_gpus=synced_gpus,
                                                       prefix_state=prefix_state,
                                                       **model_kwargs)
        return generated_tokens

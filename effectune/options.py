from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GenerationArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    min_length: Optional[int] = field(
        default=10,
        metadata={
            "help": "minimal generation length"
        },
    )

    max_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "max generation length"
        },
    )

    num_beams: Optional[int] = field(
        default=5,
        metadata={
            "help": "minimal generation length"
        },
    )

    no_repeat_ngram_size: Optional[int] = field(
        default=0,
        metadata={
            "help": "minimal generation length"
        },
    )

    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "length penalty"
        },
    )

@dataclass
class TuneArguments:
    attn_mode: Optional[str] = field(
        default="lisa",
        metadata={
            "choices": ["lisa", "lisa_nomlp",
            "learn_bias", "luna", "none",
            "dlisa", "adapter", "default_cross_attn_only"], \

            "help": "config for attention, none to disable; \
                lisa: lisa's mlp to output prefix P; \
                lisa_nomlp: prefix P as learned params; \
                learn_bias: add a learned bias param to attn output; \
                adapter: adapter mode",
        },
    )

    ffn_mode: Optional[str] = field(
        default="none",
        metadata={
            "choices": ["adapter", "none", "mh_adapter", "mh_adapter_random"],

            "help": "config for ffn, none to disable; \
            adapter: adapter mode; \
            mh_adapter: multi-head adapter, this approach adds a dxd param matrix; \
            mh_adapter_random: mh_adapter but with the dxd matrix fixed as a random mapping",
        },
    )

    attn_option: Optional[str] = field(
        default="concat",
        metadata={
            "choices": ["concat", "cross_attn", "cross_attn_gate",
                        "cross_attn_noln", "cross_attn_plug",
                        "cross_attn_plug_before_outproj",
                        "cross_attn_relu",
                        "kv_proj", "attn_adapter",
                        "attn_adapter_after_oproj", "none",
                        ], \

            "help": "specific attn configs; \
                concat: concat prefix to self, lisa's default version; \
                cross_attn: cross attention version of lisa's, a layernorm is added by default; \
                cross_attn_gate: cross attention version of lisa's, but with a learned gate function; \
                cross_attn_noln: similar to `cross_attn` without the added layernorm; \
                cross_attn_plug: cross_attn, but with Ho as input and Ho as output; \
                cross_attn_relu: change the softmax in cross_attn to relu; \
                kv_proj: P_k and P_v are projections from P; \
                attn_adapter: a single head adapter, \
                attn_adapter_after_oproj: Hi as input, add to Ho (after output proj)",

        },
    )

    ffn_option: Optional[str] = field(
        default="ffn_hi_input",
        metadata={
            "choices": ["ffn_hi_input", "ffn_ho_input", "none"], \

            "help": "specific ffn configs; \
                ffn_hi_input: ffn uses Hi as input; \
                ffn_ho_input: ffn uses Ho as input"
        },
    )


    attn_gate: Optional[str] = field(
        default="none",
        metadata={
            "help": "the gating schedule in attention change, none to disable; \
                use 'auto' to mimic the gating in prefix tuning; \
                use a float as the coefficient of original h to perform linear interpolation"
        },
    )

    ffn_gate: Optional[str] = field(
        default="none",
        metadata={
            "help": "the gating schedule in ffn change, none to disable; \
            use a float as the coefficient of original h to perform linear interpolation"
        },
    )

    ffn_num_heads: Optional[int] = field(
        default=1,
        metadata={
            "help": "the number of heads in mh_adapter/mh_adapter_random mode"
        },
    )

    adapter_post_layernorm: Optional[int] = field(
        default=0,
        metadata={
            "help": ""
        },
    )

    mid_dim: Optional[int] = field(
        default=800,
        metadata={
            "help": ""
        },
    )

    preseqlen: Optional[int] = field(
        default=200,
        metadata={
            "help": ""
        },
    )

    ffn_bn_len: Optional[int] = field(
        default=-1,
        metadata={
            "help": ""
        },
    )

    prefix_dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": ""
        },
    )

    unfreeze_params: Optional[str] = field(
        default="none",
        metadata={
            "help": ""
        },
    )

    luna_option: Optional[str] = field(
        default="full_layer",
        metadata={
            "help": "", "choices": ["self_attn", "full_layer", "full_before", "full_after"]
        },
    )

    num_bias_layers: Optional[int] = field(
        default=1,
        metadata={
            "help": ""
        },
    )

    share_luna_params: Optional[int] = field(
        default=1,
        metadata={
            "help": ""
        },
    )

    mh_reuse_proj: Optional[bool] = field(
        default=False,
        metadata={
            "help": ""
        },
    )

    layer_norm_before: Optional[int] = field(
        default=1,
        metadata={
            "help": ""
        },
    )

    layer_norm_after: Optional[int] = field(
        default=0,
        metadata={
            "help": ""
        },
    )

    init_with_bert: Optional[int] = field(
        default=1,
        metadata={
            "help": ""
        },
    )

    mydebug: Optional[int] = field(
        default=0,
        metadata={
            "help": ""
        },
    )

    analysis_opt: Optional[str] = field(
        default="",
        metadata={
            "help": ""
        },
    )

    load_path: Optional[str] = field(
        default="",
        metadata={
            "help": ""
        },
    )



# @dataclass
# class TuneArguments:
#     """
#     Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
#     """






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
        default="none",
        metadata={
            "choices": ["prefix", "prefix_nomlp",
            "none", "bitfit", "lora", "adapter", 
            "prompt_tuning"], \

            "help": "config for attention, none to disable; \
                prefix: mlp reparameterization to output prefix P; \
                prefix_nomlp: prefix P as learned params; \
                adapter: adapter mode; \
                bitfit: the bitfit baseline; \
                lora: the lora baseline; \
                prompt_tuning: the prompt tuning baseline", 
        },
    )


    attn_option: Optional[str] = field(
        default="concat",
        metadata={
            "choices": ["none", 
                        "concat", 
                        "cross_attn",
                        "cross_attn_noln",
                        "cross_attn_relu",
                        "parallel",
                        "sequential",
                        ], \

            "help": "specific attn configs; \
                concat: concat prefix to self, this is prefix tuning baseline; \
                cross_attn_noln: prefix tuning with vanilla add composition (instead of gated add) \
                cross_attn: cross_attn_noln plus a layernorm layer \
                cross_attn_relu: basically multi-head adapter; \
                parallel: parallel insertion form; need to be used under 'adapter' mode; \
                sequential: sequential insertion form; need to be used under 'adapter' mode;",

        },
    )

    attn_composition: Optional[str] = field(
        default="add",
        metadata={
            "choices": ["add", "gate_add"],
            "help": "the composition function \
                add: vanilla adding; \
                gate_add: gated adding like prefix tuning"
        },
    )

    ffn_mode: Optional[str] = field(
        default="none",
        metadata={
            "choices": ["adapter", "none", "lora"],

            "help": "config for ffn, none to disable; \
            adapter: adapter mode; \
            lora: the lora baseline",
        },
    )

    ffn_option: Optional[str] = field(
        default="none",
        metadata={
            "choices": ["parallel", "sequential", "pfeiffer", "none"], \

            "help": "specific ffn configs; \
                parallel: parallel insertion form; \
                sequential: sequential insertion form; \
                pfeiffer: the Pfeiffer adapter config"
        },
    )


    ffn_adapter_layernorm_option: Optional[str] = field(
        default="in",
        metadata={
            "choices": ["in", "out", "none"],
            "help": "ffn adapter layernorm options; \
                none: no layernorm; \
                in: layernorm applied to input; \
                out: layernorm applied to output"
        },
    )

    ffn_adapter_init_option: Optional[str] = field(
        default="bert",
        metadata={
            "choices": ["bert", "lora"],
            "help": "ffn adapter option"
        },
    )

    ffn_adapter_scalar: Optional[str] = field(
        default="1",
        metadata={
            "help": "the scaling hyperparam for scaled adding composition; \
                set to 'learnable_scalar' to learn this as a parameter"
        },
    )


    mid_dim: Optional[int] = field(
        default=800,
        metadata={
            "help": ""
        },
    )

    attn_bn: Optional[int] = field(
        default=200,
        metadata={
            "help": "the attention bottleneck dimension"
        },
    )

    ffn_bn: Optional[int] = field(
        default=-1,
        metadata={
            "help": "the ffn bottleneck dimension"
        },
    )

    prefix_dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": ""
        },
    )

    unfreeze_params: Optional[str] = field(
        default="ef_",
        metadata={
            "help": "param names that contain the string will \
                be unfreezed, all other params will be freezed"
        },
    )


    load_path: Optional[str] = field(
        default="",
        metadata={
            "help": ""
        },
    )

    lora_alpha: Optional[float] = field(
        default=32.0,
        metadata={
            "help": "scaling: alpha / r"
        },
    )

    lora_dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "scaling: alpha / r"
        },
    )

    lora_init: Optional[str] = field(
        default="lora",
        metadata={
            "choices": ["bert", "lora"],
            "help": ""
        },
    )

@dataclass
class MBARTArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    dropout: Optional[float] = field(
        default=0.3,
        metadata={
            "help": ""
        },
    )

    attention_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": ""
        },
    )





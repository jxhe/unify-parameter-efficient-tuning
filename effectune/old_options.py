import argparse


def add_efficient_tuning_args(parser):
    group = parser.add_argument_group('prefix')
    group.add_argument('--use_prefix', type=str, default=None, choices=["lisa", "learn_bias", "luna"])
    group.add_argument('--mid_dim', type=int, default=800)
    group.add_argument('--preseqlen', type=int, default=200)
    group.add_argument('--prefix_dropout', type=float, default=0.0)
    group.add_argument('--luna_option', type=str, default="full_layer", choices=["self_attn", "full_layer", "full_after", "full_before"])
    group.add_argument('--num_bias_layers', type=int, default=1)
    group.add_argument('--share_luna_params', type=int, default=1)
    return parser


def add_gen_args(parser):
    parser.add_argument('--eval_max_length', type=int, default=62)
    parser.add_argument('--eval_min_length', type=int, default=11)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    parser.add_argument('--length_penalty', type=float, default=1.0)

def add_tune_args(parser):
    group = parser.add_argument_group('finetune')
    group.add_argument('--unfreeze_params', type=str,
                       choices=['LN', 'none'],
                       default='none')
    return parser
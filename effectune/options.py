import argparse


def add_lisa_args(parser):
    group = parser.add_argument_group('prefix')
    group.add_argument('--use_prefix', type=str, default="none", 
        choices=["lisa", "learn_bias", "none"])
    group.add_argument('--mid_dim', type=int, default=800)
    group.add_argument('--preseqlen', type=int, default=200)
    group.add_argument('--prefix_dropout', type=float, default=0.0)
    return parser


def add_gen_args(parser):
    parser.add_argument('--eval_max_length', type=int, default=62)
    parser.add_argument('--eval_min_length', type=int, default=11)


def add_tune_args(parser):
    group = parser.add_argument_group('finetune')
    group.add_argument('--unfreeze_params', type=str,
                        choices=['LN', 'none'],
                        default='none')
    return parser




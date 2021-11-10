import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
from collections import defaultdict

labelsize = 16
legendsize = 14
mpl.rcParams['xtick.labelsize'] = labelsize
mpl.rcParams['ytick.labelsize'] = labelsize
mpl.rcParams['axes.labelsize'] = labelsize
mpl.rcParams['axes.titlesize'] = labelsize
mpl.rcParams['font.size'] = labelsize
plt.style.use('seaborn-deep')
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.usetex'] = True
colormap = plt.cm.gist_ncar


def plot_ax(ax, params, ys, legends, ylabel, full, title=None, add_legend=True):
    labelsize = 20
    legendsize = 20
    mpl.rcParams['xtick.labelsize'] = labelsize
    mpl.rcParams['ytick.labelsize'] = labelsize
    mpl.rcParams['axes.labelsize'] = labelsize
    mpl.rcParams['axes.titlesize'] = labelsize
    mpl.rcParams['font.size'] = labelsize
    color_base = ["blue", "red", "green", "tab:orange", "purple", "tab:cyan"]
    markers = ["o", "v", "s", "*", "8"]
    sorted_xs = list(set([x for xs in params for x in xs]))
    sorted_xs = sorted(sorted_xs)
    xticks = [format(xx) for xx in sorted_xs]
    for ii, (x, y) in enumerate(zip(params[::-1], ys[::-1])):
        ax.plot(x, y, c=color_base[ii], marker=markers[ii], ms=10, linewidth=3)

    ax.set_xlim(ax.get_xlim()[0], 15)
    p1 = ax.get_xlim()
    p1 = [p1[0]-0.1, p1[1]+1.0]
    p2 = [full, full]
    ax.plot(p1, p2, "--", ms=6, c="black", linewidth=2)
    # ax.set_xscale('log', basex=10)
    legends = legends[::-1] + ["Full Fine-tuning", "Ours"]
    if add_legend:
        ax.legend(legends, loc="best", fontsize=legendsize)
    # ax.set_xticks(sorted_xs, xticks)
    if title is not None:
        ax.set(xlabel=r"Fine-tuned Parameters (\%)", ylabel=ylabel)
    else:
        ax.set(title=title, xlabel=r"Fine-tuned Parameters (\%)", ylabel=ylabel)
    ax.grid()
    ax.set_facecolor("white")


def plot_intro():
    color_base = ["blue", "purple", "green", "tab:orange", "red", "tab:cyan"]
    # color_base = ["blue", "blue", "blue", "blue", "red", "tab:cyan"]
    color_base = ["dodgerblue", "mediumvioletred", "olivedrab", "goldenrod", "firebrick", "tab:cyan"]
    color_base = ["dodgerblue", "hotpink", "olivedrab", "goldenrod", "crimson", "tab:cyan"]
    color_base = ["gray", "dodgerblue", "olivedrab", "hotpink", "crimson", "tab:cyan"]
    markers = ["o", "v", "s", "*", "D"]
    markers = ["o", "o", "o", "o", "D"]
    fig, ax = plt.subplots(1, 1)
    full = 21.94
    legends = ["Full Fine-tuning", "BitFit", "PrefixTuning", "Adapter", "LoRA", "Ours"]
    params = [0.08, 3.6, 12.3, 14.4, 6.7]
    xsum = [17.32, 20.46, 20.98, 20.5, 21.9]
    for ii, (param, r2) in enumerate(zip(params, xsum)):
        ax.scatter(param, r2, c=color_base[ii], marker=markers[ii], edgecolor='black', linewidth=1, s=300)

    ax.set_xlim(ax.get_xlim()[0], 15)
    p1 = ax.get_xlim()
    p1 = [p1[0]-0.1, p1[1]+1.0]
    p2 = [full, full]
    ax.plot(p1, p2, "--", ms=6, c="black", linewidth=2)

    # ax.legend(legends, loc='best', fontsize=12)
    ax.grid()
    ax.set_facecolor("white")
    ax.set(xlabel=r"Fine-tuned Parameters (\%)", ylabel="ROUGE-2")
    fig.set_size_inches(5, 5)
    fig.savefig("intro.pdf", bbox_inches='tight')


def compute_params(r):
    base = 200 * 2 * 3 * 1024 * 12
    base_params = 3.6
    print(r * 1.0 / base * base_params)
    return r * 1.0 / base * base_params


def format(n):
    return r"{:.1f}%".format(n)


def plot_overview():
    d, L = 1024, 12

    # fig, axes = plt.subplots(2, 1)
    # percentage of parameters
    params_bitfit = [0.08]
    # params_prompt = [compute_params(d * 1), compute_params(d * 30), compute_params(d * 200), compute_params(d * 300)]
    params_prompt = [compute_params(d * 300)]
    params_pt = [compute_params(1 * 2 * 3 * d * L), compute_params(30 * 2 * 3 * d * L),
                 compute_params(200 * 2 * 3 * d * L), compute_params(512 * 2 * 3 * d * L)]
    params_hously_adapter_ffn_ho = [compute_params(30 * 2 * 2 * d * L),
                                compute_params(200 * 2 * 2 * d * L),
                                compute_params(512 * 2 * 2 * d * L), compute_params(1024 * 2 * 2 * d * L)]
    params_lora_attn = [compute_params(1*4*3*d*L), compute_params(30*4*3*d*L), compute_params(200*4*3*d*L),
                        compute_params(400*4*3*d*L)]
    params_lora_ffn = [compute_params(1*10*2*d*L), compute_params(102*10*2*d*L), compute_params(120*10*2*d*L)]

    params_hously_adapter_attn_ho = [compute_params(1 * 2 * 3 * d * L), compute_params(30 * 2 * 3 * d * L),
                                    compute_params(200 * 2 * 3 * d * L),
                                    compute_params(512 * 2 * 3 * d * L), compute_params(1024 * 2 * 3 * d * L)]
    # print("prompt: 300")
    # print(params_prompt)
    # print("pt: 1, 30, 200, 512")
    # print(params_pt)
    # print("ho/hi ffn: 1, 30, 200, 512, 1024")
    # print(params_hously_adapter_ffn_ho)
    # print("ho/hi attn: 1, 30, 200, 512, 1024")
    # print(params_hously_adapter_attn_ho)
    # print("lora attn: 1, 30, 200, 400")
    # print(params_lora_attn)
    # print("lora ffn: 1, 102, 120")
    # print(params_lora_ffn)

    # xsum
    xsum_bitfit = [17.32]
    # xsum_prompt = [5.33, 14, 15.49, 15.98]  # 1, 30?, 200, 300
    # xsum_prompt = [15.98]  # 300
    xsum_pt = [18.14, 20.01, 20.46, 20.40]  # 1, 30, 200, 512
    xsum_hously_adapter_ffn_ho = [17, 18.81, 20.4, 20.58, 20.98]  # 1, 30, 200?, 512?, 1024?
    xsum_hously_adapter_ffn_ho = [18.81, 20.4, 20.58, 20.98]  # 1, 30, 200?, 512?, 1024?
    xsum_lora_attn = [17.4, 19.59, 20.29, 20.5]  # 1, 30, 200, 400

    # mt
    mt_bitfit = [26.4]
    # mt_prompt = [6.0, 16.7, 21]  # 1, 30, 200
    # mt_prompt = [21]  # 200
    mt_pt = [30.2, 35.2, 35.6, 35.1]  # 1, 30, 200, 512
    mt_hously_adapter_ffn_ho = [24.3, 33.0, 35.6, 36.3, 36.7]  # 1, 30, 200, 512, 1024
    mt_hously_adapter_ffn_ho = [33.0, 35.6, 36.3, 36.7]  # 1, 30, 200, 512, 1024
    mt_lora_attn = [25.5, 34.2, 36.2, 36.6]  # 1, 30, 200, 400

    # legends = ["BitFit (bias)", "PromptTuning (input)", "PrefixTuning (attn)", "Adapter (ffn)", "LoRA (attn)"]

    # plot_ax(axes[0], [params_bitfit, params_prompt, params_pt, params_hously_adapter_ffn_ho, params_lora_attn],
    #         [xsum_bitfit, xsum_prompt, xsum_pt, xsum_hously_adapter_ffn_ho, xsum_lora_attn], legends, "ROUGE-2", full=21.94, ours=21.90,
    #         title="(a) abstractive text summarization", add_legend=False)
    # plot_ax(axes[1], [params_bitfit, params_prompt, params_pt, params_hously_adapter_ffn_ho, params_lora_attn],
    #         [mt_bitfit, mt_prompt, mt_pt, mt_hously_adapter_ffn_ho, mt_lora_attn], legends, "BLEU", full=37.3, ours=37.5,
    #         title="(b) machine translation")

    fig, ax = plt.subplots(1, 1)

    legends = ["BitFit", "PrefixTuning", "Adapter", "LoRA"]
    plot_ax(ax, [params_bitfit, params_pt, params_hously_adapter_ffn_ho, params_lora_attn],
            [xsum_bitfit, xsum_pt, xsum_hously_adapter_ffn_ho, xsum_lora_attn], legends, "XSum ROUGE-2", full=21.94,
            title=None, add_legend=False)
    fig.set_size_inches(5, 5)
    fig.savefig("xsum_overview.pdf", bbox_inches='tight')

    fig, ax = plt.subplots(1, 1)
    plot_ax(ax, [params_bitfit, params_pt, params_hously_adapter_ffn_ho, params_lora_attn],
            [mt_bitfit, mt_pt, mt_hously_adapter_ffn_ho, mt_lora_attn], legends, "MT BLEU", full=37.3,
            title=None)
    fig.set_size_inches(5,5)
    fig.savefig("mt_overview.pdf", bbox_inches='tight')

def plot_table4():
    color_base = ["blue", "red", "green", "tab:orange", "tab:cyan", "purple", ]
    markers = ["o", "v", "s", "*", "D"]

    fig, ax = plt.subplots(1, 1)
    ylabel = "XSum ROUGE-2"
    params_pt = [3.6, 9.2]
    params_lora = [7.2]
    params_adapter = [3.6, 9.2]
    r2_pt = [20.46, 20.40]
    r2_lora = [20.29]
    r2_adapter = [20.31, 20.83]

    ffn_params_lora = [6.1]
    ffn_r2_lora = [21.31]
    ffn_params_adapter = [2.4, 6.1, 12.3]
    ffn_r2_adapter = [20.66, 20.98, 21.24]

    ax.plot(params_pt, r2_pt, c=color_base[0], marker=markers[0], ms=10, linewidth=2)
    ax.plot(params_adapter, r2_adapter, c=color_base[0], marker=markers[1], ms=10, linewidth=2)
    ax.plot(params_lora, r2_lora, c=color_base[0], marker=markers[2], ms=10, linewidth=2)

    ax.plot(ffn_params_adapter, ffn_r2_adapter, "--", c=color_base[1], marker=markers[1], ms=10, linewidth=2)
    ax.plot(ffn_params_lora, ffn_r2_lora, "--", c=color_base[1], marker=markers[2], ms=10, linewidth=2)
    # legends = ["attn-PT", "attn-PA", "attn-LoRA", "ffn-PA",
    #            "ffn-LoRA"]
    # ax.legend(legends, loc="lower right", fontsize=12)
    ax.set(xlabel=r"Fine-tuned Parameters (\%)", ylabel=ylabel)
    ax.grid()
    ax.set_facecolor("white")
    fig.set_size_inches(5, 3)
    fig.savefig("xsum_modification_position.pdf", bbox_inches='tight')


    fig, ax = plt.subplots(1, 1)
    ylabel = "MT BLEU"
    params_pt = [3.6, 9.2]
    params_lora = [7.2]
    params_adapter = [3.6, 9.2]
    bleu_pt = [35.6, 35.1]
    bleu_lora = [36.2]
    bleu_adapter = [35.6, 36.2]

    ffn_params_lora = [6.1]
    ffn_params_adapter = [2.4, 6.1, 12.3]
    ffn_bleu_lora = [36.5]
    ffn_bleu_adapter = [36.4, 37.1, 37.3]

    ax.plot(params_pt, bleu_pt, c=color_base[0], marker=markers[0], ms=10, linewidth=2)
    ax.plot(params_adapter, bleu_adapter, c=color_base[0], marker=markers[1], ms=10, linewidth=2)
    ax.plot(params_lora, bleu_lora, c=color_base[0], marker=markers[2], ms=10, linewidth=2)

    ax.plot(ffn_params_adapter, ffn_bleu_adapter, "--", c=color_base[1], marker=markers[1], ms=10, linewidth=2)
    ax.plot(ffn_params_lora, ffn_bleu_lora, "--", c=color_base[1], marker=markers[2], ms=10, linewidth=2)
    # legends = ["attn-Prefix Tuning", "attn-Parallel Adapter", "attn-LoRA", "ffn-Parallel Adaptaer", "ffn-LoRA"]
    # ax.legend(legends, loc="lower right", fontsize=12, bbox_to_anchor=(1.27, 0.005))
    legends = ["Prefix (attn)", "PA (attn)", "LoRA (attn)", "PA (ffn)", "LoRA (ffn)"]
    ax.legend(legends, loc="lower right", fontsize=12, bbox_to_anchor=(1.11, 0.00))
    ax.set(xlabel=r"Fine-tuned Parameters (\%)", ylabel=ylabel)
    ax.grid()
    ax.set_facecolor("white")
    fig.set_size_inches(5, 3)
    fig.savefig("mt_modification_position.pdf", bbox_inches='tight')


# plot_overview()
plot_intro()
# plot_table4()
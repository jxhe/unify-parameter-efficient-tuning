import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
from collections import defaultdict

input_dir = "/home/chuntinz/tir5/tride/checkpoints/xsum/20210904/xsum_tride.am_lisa.ao_cross_attn.fm_adapter.fo_ffn_hi_input.go_cross_attn.abn30.fbn512.mh_reuse_proj_True.unfreeze_ef_.ms100000.ls0.1.warm0.wd0.01/analysis"

labelsize = 14
legendsize = 14
mpl.rcParams['xtick.labelsize'] = labelsize
mpl.rcParams['ytick.labelsize'] = labelsize
mpl.rcParams['font.size'] = labelsize
plt.style.use('seaborn-deep')
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['text.usetex'] = True
colormap = plt.cm.gist_ncar


def plot_line(ax, ys, title=None):
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(ys)))))
    legends = ["layer_{}".format(ii) for ii in range(12)]
    for idx, y in enumerate(ys):
        ax.plot(np.arange(len(y)), y, alpha=0.6)
    ax.legend(legends, loc='best', fontsize=10)
    # ax.set(title=title, xlabel="steps", ylabel=title)
    ax.set(xlabel="steps", ylabel="{} prefix weight".format(title))


def plot_files(fprefix):
    ys = []

    print("read {}".format(fprefix))
    for ii in range(12):
        y = []
        with open(os.path.join(input_dir, "{}_{}.txt".format(fprefix, ii))) as fin:
            for line in fin:
                fields = line.strip().split("\t")
                w_prefix = fields[3].strip().split("=")[-1]
                y.append(float(w_prefix))
        ny = []
        K = 10
        for ii, yy in enumerate(y):
            if ii % K == 0 and ii > 0:
                ny.append(np.mean(y[ii-K:ii]))

        ys.append(ny)
    print("done and plot")
    fig, ax = plt.subplots(1)
    plot_line(ax, ys, fprefix)
    fig.set_size_inches(20, 10)
    fig.savefig(os.path.join(input_dir, "{}.pdf".format(fprefix)), bbox_inches='tight')

plot_files("eself")
plot_files("dself")
plot_files("dcross")


from pathlib import Path

import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib import colormaps

from .joint_model import num_dim, mu_true, var_true, log_sigma_true

cmap = colormaps.get_cmap('plasma')
plt.rcParams.update(plt.rcParamsDefault)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
plt.rcParams["font.family"] = "Modern"


output_dir = Path(__file__).parent.resolve() / "output"

result_files = ['dlrd', 'sasa', 'sasa+']

textwidth = 6.5
fontsize = 8

ncols = 3
figwidth = textwidth
wspace = 0.55
ax_width = 1.4

offset = 0.037
left = (figwidth - (ncols + (ncols - 1) * wspace) * ax_width) / figwidth / 2 + offset
right = 1 - left + offset*2

nrows = 2
ax_height = 1.25
hspace = 0.4
top_abs = 0.3
bottom_abs = 0.75
figheight = nrows * ax_height + (nrows - 1) * ax_height * hspace + bottom_abs + top_abs
top = 1 - top_abs / figheight
bottom = bottom_abs / figheight


fig, axes = plt.subplots(nrows, ncols, figsize=(figwidth, figheight))
plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)

indices = np.unique(np.logspace(2, 5, 100, dtype=int))
colors = [cmap(0), cmap(0.4), cmap(0.8)]


for col, lr in enumerate(["l01", "l001", "l0001"]):
    for i, result_file in enumerate(result_files):
        print(result_file)

        with open(f"{output_dir}sgd_n2_{lr}_{result_file}.pickle", 'rb') as handle:
            results = pickle.load(handle)

        variational_params = np.array(results['iteration_data']['variational_parameters'])[indices]

        mu = variational_params[:, :num_dim]
        log_sigma = variational_params[:, num_dim:]
        var = np.exp(2 * log_sigma)

        div = np.sum(
            log_sigma - log_sigma_true + (var_true + (mu - mu_true) ** 2) / (2 * var) - 0.5, axis=1
        )
        div += np.sum(
            log_sigma_true - log_sigma + (var + (mu - mu_true) ** 2) / (2 * var_true) - 0.5, axis=1
        )

        width = 1.2
        learning_rate = np.array(results['iteration_data']['learning_rate'])
        axes[0, col].plot(indices, div, color=colors[i], linewidth=width)
        axes[1, col].plot(learning_rate, color=colors[i], linewidth=width)


axes[0, 0].set_title('$\eta_0=1\mathrm{e}{-2}$', size=fontsize + 1, y=1.04)
axes[0, 1].set_title('$\eta_0=1\mathrm{e}{-3}$', size=fontsize + 1, y=1.04)
axes[0, 2].set_title('$\eta_0=1\mathrm{e}{-4}$', size=fontsize + 1, y=1.04)

for col in range(ncols):
    for row in range(nrows):
        ax = axes[row, col]
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Iteration $i$', size=fontsize)
        ax.set_xlim(np.min(indices), np.max(indices))
        ax.grid()
        ax.minorticks_off()
        ax.tick_params(axis='both', which='both', labelsize=fontsize)
        ax.set_xticks(np.logspace(2, 5, num=4))
    axes[0, col].set_ylabel(r'$D_{\mathrm{J}}(q^*_{\boldsymbol{\lambda}} \parallel q_{\boldsymbol{\lambda}_i})$', size=fontsize)
    axes[1, col].set_ylabel('$\eta_i$', size=fontsize)

    axes[0, col].set_yticks(np.logspace(-2, 2, num=5))
    axes[1, col].set_yticks(np.logspace(-14, -2, num=7))

for col in range(ncols):
    axes[0, col].set_ylim(min(axes[0, 0].get_ylim()[0], axes[0, 1].get_ylim()[0], axes[0, 2].get_ylim()[0]), max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1], axes[0, 2].get_ylim()[1]))
    axes[1, col].set_ylim(min(axes[1, 0].get_ylim()[0], axes[1, 1].get_ylim()[0], axes[1, 2].get_ylim()[0]), max(0.03, axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1], axes[1, 2].get_ylim()[1]))


fig.legend(['DLRD', 'SASA', 'SASA+'], loc='lower center', ncols=3, fontsize=fontsize)
plt.savefig(f'div_quadnormal_sasa.pdf')


# plt.show()

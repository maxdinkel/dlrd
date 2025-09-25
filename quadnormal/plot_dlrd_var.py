from pathlib import Path

import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib import colormaps

from joint_model import num_dim, mu_true, var_true, log_sigma_true

cmap = colormaps.get_cmap('plasma')
plt.rcParams.update(plt.rcParamsDefault)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
plt.rcParams["font.family"] = "Modern"


output_dir = Path(__file__).parent.resolve() / "output"

result_files = ['l01', 'l01_dlrd', 'l001', 'l001_dlrd', 'l0001', 'l0001_dlrd']

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
bottom_abs = 0.88
figheight = nrows * ax_height + (nrows - 1) * ax_height * hspace + bottom_abs + top_abs
top = 1 - top_abs / figheight
bottom = bottom_abs / figheight


fig, axes = plt.subplots(nrows, ncols, figsize=(figwidth, figheight))
plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)

indices = np.unique(np.logspace(2, 5, 100, dtype=int))
colors = [cmap(0), cmap(0.4), cmap(0.8)]

handles = []
for row, optimizer in enumerate(['adam', 'sgd']):
    for i, lr in enumerate(['l01', 'l001', 'l0001']):
        print(lr)
        for k, suffix in enumerate(['', '_dlrd']):
            for seed in range(10):

                with open(output_dir / f"{optimizer}_n2_s{seed}_{lr}{suffix}.pickle", 'rb') as handle:
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

                handles.append(axes[row, i].plot(indices, div, color=colors[k], linewidth=0.5)[0])


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
    axes[1, col].set_ylabel(r'$D_{\mathrm{J}}(q^*_{\boldsymbol{\lambda}} \parallel q_{\boldsymbol{\lambda}_i})$', size=fontsize)

    axes[0, col].set_yticks(np.logspace(-3, 3, num=7))
    axes[1, col].set_yticks(np.logspace(-3, 3, num=7))

for col in range(ncols):
    axes[0, col].set_ylim(min(axes[0, 0].get_ylim()[0], axes[0, 1].get_ylim()[0]), max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1]))
    axes[1, col].set_ylim(min(axes[1, 0].get_ylim()[0], axes[1, 1].get_ylim()[0]), max(axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1]))


axes[0, 0].set_title('$\eta_0=1\mathrm{e}{-2}$', size=fontsize+1, y=1.04)
axes[0, 1].set_title('$\eta_0=1\mathrm{e}{-3}$', size=fontsize+1, y=1.04)
axes[0, 2].set_title('$\eta_0=1\mathrm{e}{-4}$', size=fontsize+1, y=1.04)


fig.legend([handles[-11], handles[-1]], ['static', 'DLRD'], loc='lower center', ncols=2, fontsize=fontsize)
plt.savefig(output_dir / 'div_quadnormal_var.pdf')

# plt.show()

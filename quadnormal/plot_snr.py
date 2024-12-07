from pathlib import Path

import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib import colormaps

cmap = colormaps.get_cmap('plasma')
plt.rcParams.update(plt.rcParamsDefault)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
plt.rcParams["font.family"] = "Modern"


output_dir = Path(__file__).parent.resolve() / "output"

np.random.seed(2)
num_dim = 100

textwidth = 6.5
fontsize = 8

ncols = 2
figwidth = textwidth
wspace = 0.6
ax_width = 1.5

left = (figwidth - (ncols + (ncols - 1) * wspace) * ax_width) / figwidth / 2
right = 1 - left

nrows = 1
ax_height = 1.3
hspace = 0.5
top_abs = 0.1
bottom_abs = 0.38
figheight = nrows * ax_height + (nrows - 1) * ax_height * hspace + bottom_abs + top_abs
top = 1 - top_abs / figheight
bottom = bottom_abs / figheight


fig, axes = plt.subplots(nrows, ncols, figsize=(figwidth, figheight))
plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)

indices = np.unique(np.logspace(0, 5, 1000, dtype=int))


with open(output_dir / 'adam_n2_l01.pickle', 'rb') as handle:
    results = pickle.load(handle)

variational_params = np.array(results['iteration_data']['variational_parameters'])


k = np.arange(len(variational_params)).reshape(-1, 1)
a = np.cumsum(variational_params, axis=0)
b = np.cumsum(variational_params ** 2, axis=0)
c = np.cumsum(variational_params * k, axis=0)

snr = 1 / (k * (k+1) * (k+2) / 12 * (b - 1 / (k+1) * a ** 2) / (c - k/2 * a) ** 2 - 1)
snr[:2, :] = np.nan
snr_mean = np.mean(snr, axis=1)


for i in range(num_dim):
    axes[0].plot(variational_params[:, i], color=cmap(i / num_dim), linewidth=0.5)

axes[1].plot(snr_mean, color=cmap(0.0), linewidth=1.2)
axes[1].hlines(1, xmin=1, xmax=1e5, color=cmap(0.4), linestyles='--', linewidth=1.2)
axes[1].annotate(r'$\rho_{\mathrm{min}}$', xy=(1.3e4, 2), size=fontsize, color=cmap(0.4))

for ax in axes:
    ax.set_xscale('log')
    ax.set_xlim(1, 1e5)
    ax.set_xlabel('Iteration $i$', size=fontsize)
    ax.set_xticks([1, 1e1, 1e2, 1e3, 1e4, 1e5])
    ax.tick_params(axis='both', which='both', labelsize=fontsize)


axes[0].set_yticks([-3, -2, -1, 0, 1, 2, 3])
axes[1].set_yticks([1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8])
axes[0].set_ylim([-3, 3])
axes[1].set_ylim([1e-4, 1e6])

axes[0].set_ylabel(r'$\boldsymbol{\lambda}_i$', size=fontsize)
axes[1].set_ylabel(r'$\overline{\rho}_i$', size=fontsize)
axes[1].set_yscale('log')

axes[1].grid()
plt.savefig(output_dir / 'snr.pdf')

# plt.show()

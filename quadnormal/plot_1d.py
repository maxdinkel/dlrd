import matplotlib.pyplot as plt
import pickle
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib import cm

from queens.global_settings import GlobalSettings
from queens.variational_distributions import MeanFieldNormalVariational
from queens.main import run_iterator
from queens.distributions import NormalDistribution
from queens.parameters import Parameters

from ..my_optimizers import MyAdam
from ..my_rpvi import MyRPVIIterator

cmap = colormaps.get_cmap('plasma')
plt.rcParams.update(plt.rcParamsDefault)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
plt.rcParams["font.family"] = "Modern"

output_dir = Path(__file__).parent.resolve() / "output"

np.random.seed(2)
num_dim = 1

# Define true parameters
loc = np.random.randn(num_dim)
scale = np.random.uniform(1, 5, num_dim)

mean = loc
v = np.random.randn(num_dim, num_dim) * 0.1
cov = v @ v + np.eye(num_dim) * 1
precision = np.linalg.inv(cov)

mu_true = mean.reshape(-1)
var_true = (- np.diag(precision) + np.sqrt(np.diag(precision) ** 2 + 24 / scale ** 4)) / (12 / scale ** 4)
log_sigma_true = np.log(var_true) / 2

m = np.linspace(-0.5, 0.1, 100)
log_s = np.linspace(-0.1, 0.02, 100)
M, log_S = np.meshgrid(m, log_s)
S = np.exp(log_S)
Z = (
    1 / (2 * scale ** 4) * (3 * S ** 4 + 6 * (M - mean) ** 2 * S ** 2 + (M - mean) ** 4)
    + 1 / 2 * precision * (S ** 2 + (M - mean) ** 2)
    - log_S
)
variational_params_true = np.concatenate((mu_true, log_sigma_true))


class LikelihoodModel:
    def evaluate_and_gradient(self, samples):
        logpdf_quad = - 0.5 * np.sum(((samples - loc) / scale) ** 4, axis=1)
        dist = samples.reshape(-1, num_dim) - mean.reshape(1, -1)
        logpdf_normal = -0.5 * (np.dot(dist, precision) * dist).sum(axis=1)
        logpdf = logpdf_quad + logpdf_normal

        grad_logpdf_quad = - 2 / scale * ((samples - loc) / scale) ** 3
        grad_logpdf_normal = np.dot(-dist, precision)
        grad_logpdf = grad_logpdf_quad + grad_logpdf_normal

        return logpdf, grad_logpdf


x = NormalDistribution(mean=0, covariance=1)
parameters = Parameters(**{f'x_{i+1}': x for i in range(num_dim)})
parameters.joint_logpdf = lambda samples: np.zeros(samples.shape[0])
parameters.grad_joint_logpdf = lambda samples: np.zeros(samples.shape)

likelihood_model = LikelihoodModel()

for learning_rate in [0.01, 0.001, 0.0001]:
    optimizer = MyAdam(learning_rate=learning_rate, max_iteration=int(1e3)+1)
    experiment_name = f'adam_n1_1d_l{str(learning_rate)[2:]}'

    global_settings = GlobalSettings(experiment_name=experiment_name, output_dir=output_dir)
    with global_settings:
        np.random.seed(42)
        iterator = MyRPVIIterator(
            global_settings=global_settings,
            model=likelihood_model,
            parameters=parameters,
            stochastic_optimizer=optimizer,
            variational_distribution=MeanFieldNormalVariational(num_dim),
            n_samples_per_iter=1,
            verbose_every_n_iter=10_000
        )

        run_iterator(iterator, global_settings)


result_files = ['l0001', 'l001', 'l01']


textwidth = 6.5
fontsize = 8

ncols = 3
figwidth = textwidth
wspace = 0.6
ax_width = 1.23

left = 0.09
right = left + (ncols + (ncols - 1) * wspace) * ax_width / figwidth

nrows = 1
ax_height = ax_width
hspace = 0.5
top_abs = 0.2
bottom_abs = 0.38
figheight = nrows * ax_height + (nrows - 1) * ax_height * hspace + bottom_abs + top_abs
figheight = np.round(figheight, decimals=5)
top = 1 - top_abs / figheight
bottom = bottom_abs / figheight


fig, axes = plt.subplots(nrows, ncols, figsize=(figwidth, figheight))
plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)

indices = np.arange(1001, step=1)

for i, result_file in enumerate(result_files):
    print(result_file)

    with open(output_dir / f'adam_n1_1d_{result_file}.pickle', 'rb') as handle:
        results = pickle.load(handle)

    variational_params = np.array(results['iteration_data']['variational_parameters'])[indices]

    cmap_range = cmap(indices / indices[-1])

    axes[i].contour(M, log_S, Z, colors='grey', alpha=0.5, levels=20)
    axes[i].plot(variational_params[:, 0], variational_params[:, 1], linewidth=1, color='k', alpha=0.99, zorder=0)
    axes[i].scatter(variational_params[:, 0], variational_params[:, 1], color=cmap_range, marker='.', zorder=5, s=10)
    axes[i].scatter(variational_params_true[0], variational_params_true[1], color='k', s=30, zorder=10, label=r'$\boldsymbol{\lambda}^*$')
    axes[i].legend(loc=(0.45, 0.03), framealpha=1)

    axes[i].set_xlim([-0.5, 0.1])
    axes[i].set_ylim([-0.05, 0.02])
    axes[i].set_ylim([-0.04, 0.01])

    axes[i].set_ylim([-0.1, 0.02])

    axes[i].set_xlabel('$\lambda_1$', size=fontsize)
    axes[i].set_ylabel('$\lambda_2$', size=fontsize)
    axes[i].tick_params(axis='both', which='both', labelsize=fontsize)
    axes[i].set_xticks([-0.5, -0.2, 0.1])
    axes[i].set_yticks([-0.1, -0.04, 0.02])


axes[0].set_title('$\eta=1\mathrm{e}{-4}$', size=fontsize+1)
axes[1].set_title('$\eta=1\mathrm{e}{-3}$', size=fontsize+1)
axes[2].set_title('$\eta=1\mathrm{e}{-2}$', size=fontsize+1)


cax = plt.axes((0.91, bottom, 0.015, top-bottom))

norm = cm.colors.Normalize(vmax=indices.max(), vmin=indices.min())
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, cmap=cmap)
cbar.set_label('Iteration', size=fontsize)
cbar.ax.tick_params(labelsize=fontsize)

plt.savefig(f'1d_convergence.pdf')

# plt.show()
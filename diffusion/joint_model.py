import numpy as np

from queens.distributions.normal import NormalDistribution
from queens.parameters.parameters import Parameters
from queens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood

from diffusion_model import num_components, DiffusionModel


num_dim = num_components
forward_model = DiffusionModel()

np.random.seed(43)
ground_truth = np.random.randn(num_components)
experimental_data = forward_model.evaluate(ground_truth.reshape(1, -1))
obs = experimental_data['result']
noise_std = 0.05
obs = obs + np.random.randn(*obs.shape) * noise_std

likelihood_model = GaussianLikelihood(
    forward_model=forward_model,
    noise_type='fixed_variance',
    noise_value=noise_std ** 2,
    y_obs=obs
)

x = NormalDistribution(mean=0, covariance=1)
parameters = Parameters(**{f'x_{i+1}': x for i in range(num_components)})

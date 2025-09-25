import numpy as np
from scipy.stats import norm
from scipy.special import log_ndtr

from queens.variational_distributions.variational_distribution import VariationalDistribution

class MeanFieldSkewNormalVariational(VariationalDistribution):
    def __init__(self, dimension):
        super().__init__(dimension)
        self.n_parameters = 3 * dimension

    def initialize_variational_parameters(self, random=False):
        variational_parameters = np.zeros(self.n_parameters)
        return variational_parameters

    def reconstruct_distribution_parameters(self, variational_parameters):
        mean = variational_parameters[: self.dimension]
        std = np.exp(variational_parameters[self.dimension : 2*self.dimension])
        alpha = variational_parameters[-self.dimension:]
        return mean, std, alpha

    def _grad_reconstruct_distribution_parameters(self, variational_parameters):
        grad_mean = np.ones((1, self.dimension))
        grad_std = (np.exp(variational_parameters[self.dimension:2*self.dimension])).reshape(1, -1)
        grad_alpha = np.ones((1, self.dimension))
        grad_reconstruct_params = np.hstack((grad_mean, grad_std, grad_alpha))
        return grad_reconstruct_params

    def draw(self, variational_parameters, n_draws=1):
        return self.conduct_reparameterization(variational_parameters, n_draws)[0]

    def logpdf(self, variational_parameters, x):
        mean, std, alpha = self.reconstruct_distribution_parameters(variational_parameters)
        std = std[np.newaxis, :]
        mean = mean[np.newaxis, :]
        alpha = alpha[np.newaxis, :]
        z = (x - mean) / std
        logpdf = np.sum(np.log(2) - np.log(std) + norm.logpdf(z) + log_ndtr(alpha * z), axis=1)

        return logpdf

    def pdf(self, variational_parameters, x):
        pdf = np.exp(self.logpdf(variational_parameters, x))
        return pdf

    def grad_sample_logpdf(self, variational_parameters, x):
        mean, std, alpha = self.reconstruct_distribution_parameters(variational_parameters)
        std = std[np.newaxis, :]
        mean = mean[np.newaxis, :]
        alpha = alpha[np.newaxis, :]
        z = (x - mean) / std

        gradient = - z / std + np.exp(norm.logpdf(alpha * z) - log_ndtr(alpha * z)) * alpha / std
        return gradient

    def conduct_reparameterization(self, variational_parameters, n_samples):
        standard_normal_sample_batch = np.random.normal(0, 1, size=(n_samples, self.dimension, 2))
        mean, std, alpha = self.reconstruct_distribution_parameters(variational_parameters)

        u = standard_normal_sample_batch[:, :, 0]
        v = standard_normal_sample_batch[:, :, 1]

        delta = alpha / np.sqrt(1 + alpha ** 2)

        samples = mean + std * (delta * np.abs(u) + np.sqrt(1 - delta ** 2) * v)

        return samples, standard_normal_sample_batch

    def grad_params_reparameterization(
        self, variational_parameters, standard_normal_sample_batch, upstream_gradient
    ):
        grad_reconstruct_params = self._grad_reconstruct_distribution_parameters(
            variational_parameters
        )
        u = standard_normal_sample_batch[:, :, 0]
        v = standard_normal_sample_batch[:, :, 1]
        mean, std, alpha = self.reconstruct_distribution_parameters(variational_parameters)
        delta = alpha / np.sqrt(1 + alpha ** 2)
        gradient = (
            np.hstack((
                upstream_gradient,
                upstream_gradient * (delta * np.abs(u) + np.sqrt(1 - delta ** 2) * v),
                upstream_gradient * (std / (1 + alpha ** 2) ** (3/2) * (np.abs(u) - alpha * v))
            ))
            * grad_reconstruct_params
        )
        return gradient
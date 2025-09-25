import numpy as np
from scipy.stats import norm

from queens.variational_distributions.variational_distribution import VariationalDistribution


class MeanFieldSinhArcsinhVariational(VariationalDistribution):
    def __init__(self, dimension):
        super().__init__(dimension)
        self.n_parameters = 4 * dimension

    def initialize_variational_parameters(self, random=False):
        return np.zeros(self.n_parameters)

    def reconstruct_distribution_parameters(self, variational_parameters):
        mean = variational_parameters[:self.dimension]
        scale = np.exp(variational_parameters[self.dimension:2*self.dimension])
        skewness = variational_parameters[2*self.dimension:3*self.dimension]
        tailweight = np.exp(variational_parameters[3*self.dimension:])
        return mean, scale, skewness, tailweight

    def _grad_reconstruct_distribution_parameters(self, variational_parameters):
        grad_mean = np.ones((1, self.dimension))
        grad_scale = np.exp(variational_parameters[self.dimension:2*self.dimension]).reshape(1, -1)
        grad_skewness = np.ones((1, self.dimension))
        grad_tailweight = np.exp(variational_parameters[3*self.dimension:]).reshape(1, -1)
        return np.hstack((grad_mean, grad_scale, grad_skewness, grad_tailweight))

    def draw(self, variational_parameters, n_draws=1):
        return self.conduct_reparameterization(variational_parameters, n_draws)[0]

    def logpdf(self, variational_parameters, x):
        mean, scale, alpha, tau = self.reconstruct_distribution_parameters(variational_parameters)

        u = (x - mean) / scale
        t = np.arcsinh(u)
        w = (t - alpha) / tau
        z = np.sinh(w)

        logpdf = np.sum(
            norm.logpdf(z)
            + 0.5 * np.log1p(z**2)
            - np.log(tau)[None, :]
            - np.log(scale)[None, :]
            - np.log(np.cosh(t)), axis=1
        )
        return logpdf

    def pdf(self, variational_parameters, x):
        return np.exp(self.logpdf(variational_parameters, x))

    def grad_sample_logpdf(self, variational_parameters, x):
        mean, scale, alpha, tau = self.reconstruct_distribution_parameters(variational_parameters)

        u = (x - mean) / scale
        t = np.arcsinh(u)
        w = (t - alpha) / tau
        z = np.sinh(w)

        cosh_t = np.cosh(t)
        sqrt1pz2 = np.sqrt(1.0 + z**2)

        dtdx = 1.0 / (scale * cosh_t)
        dzdx = sqrt1pz2 / (tau * scale * cosh_t)

        grad = (-z + z / (1.0 + z**2)) * dzdx - np.tanh(t) * dtdx
        return grad

    def conduct_reparameterization(self, variational_parameters, n_samples):
        mean, scale, alpha, tau = self.reconstruct_distribution_parameters(variational_parameters)
        z = np.random.normal(size=(n_samples, self.dimension))
        t = tau * np.arcsinh(z) + alpha
        transformed = np.sinh(t)
        samples = mean + scale * transformed
        return samples, z

    def grad_params_reparameterization(self, variational_parameters, standard_normal_sample_batch, upstream_gradient):
        grad_reconstruct_params = self._grad_reconstruct_distribution_parameters(variational_parameters)
        mean, scale, alpha, tau = self.reconstruct_distribution_parameters(variational_parameters)

        z = standard_normal_sample_batch
        asinh_z = np.arcsinh(z)
        t = tau * asinh_z + alpha
        sinh_t = np.sinh(t)
        cosh_t = np.cosh(t)

        grad_mean = upstream_gradient
        grad_scale = upstream_gradient * sinh_t
        grad_skewness = upstream_gradient * scale * cosh_t
        grad_tailweight = upstream_gradient * scale * cosh_t * asinh_z

        gradient = np.hstack((grad_mean, grad_scale, grad_skewness, grad_tailweight)) * grad_reconstruct_params
        return gradient
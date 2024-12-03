import numpy as np
from queens.distributions import NormalDistribution
from queens.parameters import Parameters


np.random.seed(2)
num_dim = 100

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

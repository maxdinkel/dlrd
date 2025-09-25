from sklearn.datasets import load_wine
import numpy as np
from scipy.special import logsumexp, softmax

from queens.distributions.normal import NormalDistribution
from queens.parameters import Parameters


inputs, targets = load_wine(return_X_y=True)
inputs = (inputs - np.mean(inputs, axis=0)) / np.std(inputs, axis=0)
inputs = np.concatenate((np.ones((inputs.shape[0], 1)), inputs), axis=1)

num_data, num_features = inputs.shape
num_classes = int(targets.max()) + 1
num_dim = num_features * num_classes

one_hot_targets = np.zeros((num_data, num_classes))
one_hot_targets[np.arange(num_data), targets] = 1.0


class MultinomialLogisticModel:
    @staticmethod
    def _linear(samples):
        samples = samples.reshape(samples.shape[0], num_features, num_classes)
        return np.einsum('ij,kjl->kil', inputs, samples)

    @staticmethod
    def _evaluate_from_linear(linear):
        linear_target = np.take_along_axis(linear, targets[np.newaxis, :, np.newaxis],
                                           axis=2).squeeze(2)
        return np.sum(linear_target - logsumexp(linear, axis=2), axis=1)

    def evaluate(self, samples):
        linear = self._linear(samples)
        log_likelihood = self._evaluate_from_linear(linear)
        return log_likelihood

    def evaluate_and_gradient(self, samples):
        linear = self._linear(samples)
        log_likelihood = self._evaluate_from_linear(linear)

        diff = one_hot_targets[np.newaxis, :, :] - softmax(linear, axis=2)
        log_likelihood_grad = np.einsum('nd,snk->sdk', inputs, diff)
        grad = log_likelihood_grad.reshape(samples.shape[0], num_dim)

        return log_likelihood, grad


likelihood_model = MultinomialLogisticModel()

x = NormalDistribution(mean=0, covariance=10)
parameters = Parameters(**{f'x_{i+1}': x for i in range(num_dim)})

from sklearn.datasets import load_breast_cancer
import numpy as np

from queens.distributions.normal import NormalDistribution
from queens.parameters import Parameters

inputs, targets = load_breast_cancer(return_X_y=True)
inputs = (inputs - np.mean(inputs, axis=0)) / np.std(inputs, axis=0)
inputs = np.concatenate((np.ones((inputs.shape[0], 1)), inputs), axis=1)
num_dim = inputs.shape[1]

class LogisticModel:
    def evaluate(self, samples):
        linear = np.sum(inputs[np.newaxis, :, :] * samples.reshape(-1, 1, inputs.shape[1]), axis=2)
        max_value = np.clip(-linear, a_max=None, a_min=0)
        log_logistic_fun = -(np.log((np.exp(-max_value) + np.exp(-linear - max_value))) + max_value)
        log_likelihood = np.sum(
            log_logistic_fun * targets + (log_logistic_fun - linear) * (1 - targets), axis=1)
        return log_likelihood

    def evaluate_and_gradient(self, samples):
        linear = np.sum(inputs[np.newaxis, :, :] * samples[:, np.newaxis, :], axis=2)
        max_value = np.clip(-linear, a_max=None, a_min=0)
        log_logistic_fun = -(np.log((np.exp(-max_value) + np.exp(-linear - max_value))) + max_value)
        log_likelihood = np.sum(
            log_logistic_fun * targets + (log_logistic_fun - linear) * (1 - targets), axis=1)

        grad_log_likelihood = (
            np.sum(inputs[np.newaxis, :, :] / (1 + np.exp(linear))[:, :, np.newaxis], axis=1)
            + np.sum((targets - 1)[:, np.newaxis] * inputs, axis=0, keepdims=True)
        )
        return log_likelihood, grad_log_likelihood


likelihood_model = LogisticModel()

x = NormalDistribution(mean=0, covariance=100)
parameters = Parameters(**{f'x_{i+1}': x for i in range(num_dim)})


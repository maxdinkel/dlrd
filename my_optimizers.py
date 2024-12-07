from functools import partial
import math

import numpy as np
from scipy import stats

from queens.stochastic_optimizers import Adam, SGD, Adamax, RMSprop

my_options = {
    "optimization_type": 'max',
    "rel_l1_change_threshold": -1,
    "rel_l2_change_threshold": -1,
    "clip_by_value_threshold": np.inf,
    "clip_by_l2_norm_threshold": np.inf
}

MySGD = partial(SGD, **my_options)
MyAdam = partial(Adam, **my_options)
MyAdamax = partial(Adamax, **my_options)
MyRMSprop = partial(RMSprop, **my_options)


class SGDSASAPlus(SGD):
    """Stochastic gradient descent optimizer."""

    def __init__(
        self, learning_rate, max_iteration, n_min=1000, k_test=100, delta=0.05, theta=0.125, tau=0.1
    ):
        """Initialize optimizer.

        Args:
            learning_rate (float): Learning rate for the optimizer
            max_iteration (int): Maximum number of iterations
        """
        super().__init__(learning_rate=learning_rate, max_iteration=max_iteration, **my_options)

        self.n_min = n_min
        self.k_test = k_test
        self.delta = delta
        self.theta = theta
        self.tau = tau
        self.simple_statistics = []

        self.k = 0  # identical to self.iteration
        self.k0 = 0

    def do_single_iteration(self, gradient):
        r"""Single iteration for a given gradient.

        Args:
            gradient (np.array): Current gradient
        """
        x = self.current_variational_parameters
        d = - gradient * self.precoefficient    # Translate to minimization problem

        self.current_variational_parameters = (
            self.current_variational_parameters - self.learning_rate * d
        )

        self.simple_statistics.append(np.dot(x, d) - self.learning_rate / 2 * np.dot(d, d))
        n = math.ceil(self.theta * (self.k - self.k0))
        if n > self.n_min and self.k % self.k_test == 0:
            p = int(math.floor(math.sqrt(n)))
            n = int(p ** 2)
            simple_statistics = np.array(self.simple_statistics[-n:])

            mean = np.mean(simple_statistics)

            # batch means estimator
            batches = simple_statistics.reshape(p, p)
            batch_means = np.mean(batches, 1)
            diffs = batch_means - mean
            std = np.sqrt(p / (p - 1) * np.sum(diffs ** 2))
            dof = p - 1

            half_width = stats.t.ppf(1 - self.delta / 2.0, dof) * std / math.sqrt(n)

            if mean - half_width < 0 < mean + half_width:
                self.learning_rate *= self.tau
                self.k0 = self.k
                print("Stationarity reached. learning_rate={:.2e}".format(self.learning_rate))

        self.k += 1
        self.iteration += 1


class SGDSASA(SGD):
    """Stochastic gradient descent optimizer."""

    def __init__(
        self, learning_rate, max_iteration, m=1000, delta=0.02, gamma=0.2, zeta=0.1
    ):
        """Initialize optimizer.

        Args:
            learning_rate (float): Learning rate for the optimizer
            max_iteration (int): Maximum number of iterations
        """
        super().__init__(learning_rate=learning_rate, max_iteration=max_iteration, **my_options)

        self.m = m
        self.delta = delta
        self.gamma = gamma
        self.zeta = zeta

        self.zq = []
        self.vq = []

    def do_single_iteration(self, gradient):
        r"""Single iteration for a given gradient.

        Args:
            gradient (np.array): Current gradient
        """
        x = self.current_variational_parameters
        d = - gradient * self.precoefficient    # Translate to minimization problem

        self.current_variational_parameters = (
            self.current_variational_parameters - self.learning_rate * d
        )

        self.zq.append(np.dot(x, d) - self.learning_rate / 2 * np.dot(d, d))
        self.vq.append(self.learning_rate / 2 * np.dot(d, d))

        if self.iteration % self.m == 0 and self.iteration > 0:
            n = int(len(self.zq) / 2)
            p = int(np.sqrt(n))
            n = int(p ** 2)

            zq = np.array(self.zq[-n:])
            vq = np.array(self.vq[-n:])

            zq_mean = np.mean(zq)
            vq_mean = np.mean(vq)

            # batch means estimator
            batches = zq.reshape(p, p)
            batch_means = np.mean(batches, 1)
            diffs = batch_means - zq_mean
            std = np.sqrt(p / (p - 1) * np.sum(diffs ** 2))
            dof = p - 1

            half_width = stats.t.ppf(1 - self.delta / 2.0, dof) * std / math.sqrt(n)
            threshold = self.delta * vq_mean

            if -threshold < zq_mean - half_width and zq_mean + half_width < threshold:
                self.learning_rate *= self.zeta
                self.zq = []
                self.vq = []
                print("Stationarity reached. learning_rate={:.2e}".format(self.learning_rate))

        self.iteration += 1


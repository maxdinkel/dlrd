from pathlib import Path

import numpy as np

from queens.global_settings import GlobalSettings
from queens.stochastic_optimizers.learning_rate_decay import DynamicLearningRateDecay
from queens.variational_distributions import FullRankNormalVariational
from queens.main import run_iterator

from ..my_optimizers import MyAdam, MyRMSprop
from ..my_rpvi import RPVIIterator, my_rpvi_options
from .joint_model import num_dim, likelihood_model, parameters


class CustomInitRPVIIterator(RPVIIterator):
    def _initialize_variational_params(self):
        self.variational_params = self.variational_distribution.construct_variational_parameters(
            np.zeros(num_dim), np.eye(num_dim)
        )


output_dir = Path(__file__).parent.resolve() / "output"

for optimizer_class in [MyAdam, MyRMSprop]:
    for learning_rate in [0.01, 0.001, 0.0001]:
        for decay in [True, False]:
            optimizer = optimizer_class(learning_rate=learning_rate, max_iteration=int(1e6)+1)
            experiment_name = f'{optimizer_class.func.__name__.lower()}_n8_l{str(learning_rate)[2:]}'
            if decay:
                optimizer.learning_rate_decay = DynamicLearningRateDecay()
                experiment_name = experiment_name + '_dlrd'

            global_settings = GlobalSettings(experiment_name=experiment_name, output_dir=output_dir)
            with global_settings:
                np.random.seed(42)
                iterator = CustomInitRPVIIterator(
                    global_settings=global_settings,
                    model=likelihood_model,
                    parameters=parameters,
                    stochastic_optimizer=optimizer,
                    variational_distribution=FullRankNormalVariational(num_dim),
                    n_samples_per_iter=8,
                    verbose_every_n_iter=50_000,
                    **my_rpvi_options
                )
                run_iterator(iterator, global_settings)


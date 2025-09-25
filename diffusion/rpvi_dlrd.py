from pathlib import Path

import numpy as np

from queens.global_settings import GlobalSettings
from queens.stochastic_optimizers.learning_rate_decay import DynamicLearningRateDecay
from queens.variational_distributions import FullRankNormalVariational
from queens.main import run_iterator

from my_optimizers import MyAdam, MyAdamax
from my_rpvi import MyRPVIIterator
from joint_model import num_dim, likelihood_model, parameters


output_dir = Path(__file__).parent.resolve() / "output"

for optimizer_class in [MyAdam, MyAdamax]:
    for learning_rate in [0.01, 0.001, 0.0001]:
        for decay in [True, False]:
            optimizer = optimizer_class(learning_rate=learning_rate, max_iteration=int(1e5) + 1)
            experiment_name = f'{optimizer_class.func.__name__.lower()}_n8_l{str(learning_rate)[2:]}'
            if decay:
                optimizer.learning_rate_decay = DynamicLearningRateDecay()
                experiment_name = experiment_name + '_dlrd'

            global_settings = GlobalSettings(experiment_name=experiment_name, output_dir=output_dir)
            with global_settings:
                np.random.seed(42)
                iterator = MyRPVIIterator(
                    global_settings=global_settings,
                    model=likelihood_model,
                    parameters=parameters,
                    stochastic_optimizer=optimizer,
                    variational_distribution=FullRankNormalVariational(num_dim),
                    n_samples_per_iter=8,
                    verbose_every_n_iter=10_000,
                )
                run_iterator(iterator, global_settings)

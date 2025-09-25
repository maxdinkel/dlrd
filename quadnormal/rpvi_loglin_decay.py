from pathlib import Path

import numpy as np

from queens.stochastic_optimizers.learning_rate_decay import LogLinearLearningRateDecay
from queens.global_settings import GlobalSettings
from queens.variational_distributions import MeanFieldNormalVariational
from queens.main import run_iterator

from joint_model import likelihood_model, parameters, num_dim
from my_optimizers import MySGD
from my_rpvi import MyRPVIIterator


output_dir = Path(__file__).parent.resolve() / "output"

for slope in [1, 0.5]:
    for learning_rate in [0.01, 0.001]:
        optimizer = MySGD(learning_rate=learning_rate, max_iteration=int(1e5)+1)
        optimizer.learning_rate_decay = LogLinearLearningRateDecay(slope=slope)
        experiment_name = f'sgd_n2_l{str(learning_rate)[2:]}_loglin_s{str(slope).replace(".", "_")}'

        global_settings = GlobalSettings(experiment_name=experiment_name, output_dir=output_dir)
        with global_settings:
            np.random.seed(42)
            iterator = MyRPVIIterator(
                global_settings=global_settings,
                model=likelihood_model,
                parameters=parameters,
                stochastic_optimizer=optimizer,
                variational_distribution=MeanFieldNormalVariational(num_dim),
                n_samples_per_iter=2,
                verbose_every_n_iter=10_000
            )
            run_iterator(iterator, global_settings)

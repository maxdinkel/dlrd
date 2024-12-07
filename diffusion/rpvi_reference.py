from pathlib import Path
from copy import deepcopy

import numpy as np

from queens.stochastic_optimizers.learning_rate_decay import StepwiseLearningRateDecay
from queens.global_settings import GlobalSettings
from queens.variational_distributions import FullRankNormalVariational
from queens.main import run_iterator

from .joint_model import likelihood_model, parameters, num_dim
from ..my_optimizers import MyAdam
from ..my_rpvi import RPVIIterator, my_rpvi_options


output_dir = Path(__file__).parent.resolve() / "output"

rpvi_options = deepcopy(my_rpvi_options)
rpvi_options["result_description"]['iterative_field_names'] = []
experiment_name = f'adam_n32_l01_stepwise'
global_settings = GlobalSettings(experiment_name=experiment_name, output_dir=output_dir)
with global_settings:
    np.random.seed(42)
    lr_decay = StepwiseLearningRateDecay(decay_factor=0.1, decay_interval=int(1e5))
    optimizer = MyAdam(learning_rate=0.01, max_iteration=int(1e6) + 1, learning_rate_decay=lr_decay)
    iterator = RPVIIterator(
        global_settings=global_settings,
        model=likelihood_model,
        parameters=parameters,
        stochastic_optimizer=optimizer,
        variational_distribution=FullRankNormalVariational(num_dim),
        n_samples_per_iter=32,
        verbose_every_n_iter=10_000,
        **rpvi_options
    )
    run_iterator(iterator, global_settings)

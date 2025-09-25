from pathlib import Path

import numpy as np

from queens.global_settings import GlobalSettings
from queens.stochastic_optimizers.learning_rate_decay import DynamicLearningRateDecay
from queens.main import run_iterator

from my_optimizers import MyAdam, MyRMSprop
from my_rpvi import RPVIIterator, my_rpvi_options
from joint_model import num_dim, likelihood_model, parameters
from sinh import MeanFieldSinhArcsinhVariational


class CustomInitRPVIIterator(RPVIIterator):
    def _initialize_variational_params(self):
        self.variational_params = np.zeros(self.variational_distribution.n_parameters)
        self.variational_params[num_dim:2 * num_dim] = 0.5 * np.log(10)  # Initialize with prior variance


output_dir = Path(__file__).parent.resolve() / "output"
num_samples = 16
for optimizer_class in [MyAdam, MyRMSprop]:
    for learning_rate in [0.01, 0.001, 0.0001]:
        for decay in [True, False]:
            optimizer = optimizer_class(learning_rate=learning_rate, max_iteration=int(1e6)+1)
            experiment_name = f'{optimizer_class.func.__name__.lower()}_n{num_samples}_l{str(learning_rate)[2:]}'
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
                    variational_distribution=MeanFieldSinhArcsinhVariational(num_dim),
                    n_samples_per_iter=num_samples,
                    verbose_every_n_iter=50_000,
                    **my_rpvi_options
                )
                run_iterator(iterator, global_settings)


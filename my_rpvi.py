from functools import partial

from queens.iterators.reparameteriztion_based_variational_inference import RPVIIterator


my_rpvi_options = {
    "max_feval": int(1e100),
    "score_function_bool": False,
    "natural_gradient": False,
    "FIM_dampening": False,
    "random_seed": 42,
    "result_description": {
        'write_results': True,
        'iterative_field_names': ["n_sims", "variational_parameters", 'learning_rate', "elbo"]
    },
    "variational_transformation": None,
    "variational_parameter_initialization": 'prior',
}


MyRPVIIterator = partial(RPVIIterator, **my_rpvi_options)

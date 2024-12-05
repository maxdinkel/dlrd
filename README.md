## Installation
### Install queens
- cd <your-path-to-QUEENS>
- git checkout dlrd
- conda env create
- conda activate queens
- pip install -e .[develop]

Optionally GPU support:
- pip install --upgrade "jax[cuda12]"


### Run experiments
- cd <your-path-to-dynamic_learning_rate_decayS>
- cd ..
- conda activate queens
- python -m dynamic_learning_rate_decay.run

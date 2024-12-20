### Install queens
- `git clone git@github.com:maxdinkel/queens.git <your-path-to-QUEENS>`
- `cd <your-path-to-QUEENS>`
- `git checkout dlrd`
- `conda env create`
- `conda activate queens`
- `pip install -e .`

Optionally GPU support:
- `pip install --upgrade "jax[cuda12]"`


### Run experiments
- `git clone git@github.com:maxdinkel/dlrd.git <your-path-to-dlrd>`
- `cd <your-path-to-dlrd>/..`
- `conda activate queens`
- `python -m dynamic_learning_rate_decay.run`

### Install queens
- `git clone https://github.com/maxdinkel/queens.git <your-path-to-QUEENS>`
- `cd <your-path-to-QUEENS>`
- `git checkout dlrd`
- `conda env create`
- `conda activate queens`
- `pip install -e .`

Optionally GPU support:
- `pip install --upgrade "jax[cuda12]"`


### Run experiments
- `git clone https://github.com/maxdinkel/dlrd.git dlrd`
- `cd dlrd`
- `conda activate queens`
- `PYTHONPATH=. python quadnormal/run.py`
- `PYTHONPATH=. python logistic/run.py`
- `PYTHONPATH=. python wine/run.py`
- `PYTHONPATH=. python diffusion/run.py`

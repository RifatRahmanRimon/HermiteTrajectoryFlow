# Multi-Marginal Flow Matching

Implementation of the Multi-Marginal Flow Matching model. 

## Installation

We recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) to create a new environement for installing the MMFM package.
Create and activate the environment using the following commands:
```sh
conda env create -f environment.yml

conda activate mmfm
```
After installing the dependencies, install the package:
```sh
pip install -e .
```
Keep the `-e` if you want to continue editing the code.
To use the GPU support, we recommend to install pytorch with cuda first. 

## Usage

The following code shows how to train an MMFM model:

```python
from mmfm.conditional_flow_matching import MultiMarginalFlowMatcher

MMFM = MultiMarginalFlowMatcher()  # see code for default arguments

for batch in data_loader:
    # each mini-batch contains a tuple of samples, labels/conditions and 
    # measurement timepoint
    x, targets, timepoints = batch
    optimizer.zero_grad()
    # Next, we sample timepoint, sample and compute the gradient/velocity...
    t, xt, ut, _, _ = MMFM.sample_location_and_conditional_flow(x, timepoints)
    # ...and train a network to predict the gradient
    vt = mmfm_model(torch.cat([xt, targets, t], dim=1))
    loss = torch.mean((vt - ut) ** 2)
    loss.backward()
    optimizer.step()
```

## Experiments

To reproduce the tables and figures of the synthetic and real-world experiments, please check out the code
in the `./experiments` folder.

### Synthetic Experiments

```shell
.
├── data
├── eval_exp1.ipynb
├── eval_exp2.ipynb
├── gridsearch_mmfm.sh      # Retrain all MMFM models (gridsearch wrapper)
├── gridsearch_totcfm.sh    # Retrain all time-specific CFM models (w/ or w/o shared weights) (gridsearch wrapper)
├── train_fsi.py            # Train FSI model
├── train_mmfm.py           # Train MMFM model
├── train_tcotcfm.py        # Train time- and condition-specific CFM model
└── train_totcfm.py         # Train time-specific CFM model
```

First, run the corresponding shell scripts to execute the gridsearch over the hyperparmeters.
Internally they will call the `train_xyz.py` scripts to train the actual model. Second, run one of the analysis notebooks,
which will load all the models, select the best ones by evaluating them on the validation data and
reproduce the figures and tables.

# Acceleration of Frank-Wolfe Algorithms with Open Loop Step-Sizes

## Installation guide

Download the repository and store it in your preferred location, say ~/tmp.

Open your terminal and navigate to ~/tmp.

Run the command:
```shell script
$ conda env create --file environment.yml
```

This will create the conda environment inf_dim_fw.

Activate the conda environment with:
```shell script
$ conda activate inf_dim_fw
```
Navigate to ~/tmp/inf_dim_fw

Execute the experiments:
```python3 script
>>> python3 -m experiments.afw_difw_experiments
```
```python3 script
>>> python3 experiments.kernel_herding
```
```python3 script
>>> python3 experiments.locally_accelerated_convergence_rate_experiments
```
```python3 script
>>> python3 experiments.non_polytope_boundary_experiments
```
```python3 script
>>> python3 experiments.non_polytope_exterior_experiments
```
```python3 script
>>> python3 experiments.non_polytope_interior_experiments
```
```python3 script
>>> python3 experiments.polytope_experiments
```

The experiments are then stored in ~/tmp/inf_dim_fw/experiments/figures.
# Acceleration of Frank-Wolfe algorithms with open-loop step-sizes

Code for the paper:
Wirth, E.S., Kerdreux, T., & Pokutta, S. (2022).
[Acceleration of Frank-Wolfe algorithms with open-loop step-sizes](https://arxiv.org/abs/2205.12838).

## Installation guide

Download the repository and store it in your preferred location, say ~/tmp.

Open your terminal and navigate to ~/tmp.

Run the command:
```shell script
$ conda env create --file environment.yml
```

This will create the conda environment open_loop_fw.

Activate the conda environment with:
```shell script
$ conda activate open_loop_fw
```
Navigate to ~/tmp

Execute the experiments:
```python3 script
>>> python3 -m experiments.afw_difw_experiments
```
```python3 script
>>> python3 -m experiments.kernel_herding
```
```python3 script
>>> python3 -m experiments.locally_accelerated_convergence_rate_experiments
```
```python3 script
>>> python3 -m experiments.non_polytope_boundary_experiments
```
```python3 script
>>> python3 -m experiments.non_polytope_exterior_experiments
```
```python3 script
>>> python3 -m experiments.non_polytope_interior_experiments
```
```python3 script
>>> python3 -m experiments.polytope_experiments
```

The experiments are then stored in ~/tmp/experiments/figures.
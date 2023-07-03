# Acceleration of Frank-Wolfe Algorithms with Open-Loop Step-Sizes

Code for the paper:
[Wirth, E., Kerdreux, T. and Pokutta, S., 2023, April. Acceleration of Frank-Wolfe Algorithms with Open-Loop Step-Sizes. In International Conference on Artificial Intelligence and Statistics (pp. 77-100). PMLR.](https://proceedings.mlr.press/v206/wirth23a.html)


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

To perform the experiments in the paper:

```python3 script
>>> python3 -m experiments.exterior
```
```python3 script
>>> python3 -m experiments.polytope
```
```python3 script
>>> python3 -m experiments.kernel_herding
```
```python3 script
>>> python3 -m experiments.gisette
```
```python3 script
>>> python3 -m experiments.movielens
```



The experiments are then stored in ~/tmp/experiments/figures.

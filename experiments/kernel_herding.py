# Kernel-herding experiments for both uniform and non-uniform densities.

import random
import autograd.numpy as np
from all_functions.auxiliary_functions import distribution_to_string
from all_functions.feasible_region import HilbertSpaceWhaba
from all_functions.objective_function import confirm_rho_distribution, mu_from_rho, SquaredLoss
from all_functions.plotting import primal_gap_plotter, determine_y_lims, only_min
from all_functions.experiments_auxiliary_functions import run_experiment
from global_ import *
import matplotlib as mpl

random.seed(RANDOM)
np.random.seed(RANDOM)

mpl.rcParams['agg.path.chunksize'] = CHUNKSIZE
mpl.rcParams['axes.linewidth'] = LINEWIDTH

fw_step_size_rules = [
    {"step type": "line-search"},
    {"step type": "open-loop", "a": 1, "b": 1, "c": 1, "d": 1},
    {"step type": "open-loop", "a": 2, "b": 1, "c": 2, "d": 1},
]



rho = [np.random.rand(random.randint(2, 5), 1), np.random.rand(random.randint(2, 5), 1)]
distribution_as_string = distribution_to_string(rho)
rho = confirm_rho_distribution(rho)
mu = mu_from_rho(rho)
mus = [None, mu]

for mu in mus:
    legend = False
    if mu == mus[-1]:
        legend = True
    feasible_region = HilbertSpaceWhaba(iterations_lmo=LMO_KH)
    objective_function = SquaredLoss(mu=mu)
    if mu is None:
        file_name = "kernel_herding_uniform"
    else:
        file_name = "kernel_herding_non_uniform"

    primal_gaps, labels = run_experiment(ITERATIONS_KH, objective_function, feasible_region,
                                         fw_step_size_rules=fw_step_size_rules)
    primal_gaps = only_min(primal_gaps)
    primal_gap_plotter(y_data=primal_gaps,
                       labels=labels,
                       iterations=ITERATIONS_KH,
                       file_name=file_name,
                       x_lim=(1, ITERATIONS_KH),
                       y_lim=determine_y_lims(primal_gaps),
                       y_label=r'$\mathrm{min}_i  \ h_i$',
                       directory="experiments/figures/kernel_herding",
                       legend=legend
                       )

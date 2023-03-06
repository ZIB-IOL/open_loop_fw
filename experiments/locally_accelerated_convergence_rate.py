# performs an experiment that measures the local convergence rate of the FW algorithm

from all_functions.plotting import contour_plotter
from all_functions.problem_settings import polytope_experiment, probability_simplex_interior_fast_ls_ss
import autograd.numpy as np
from all_functions.experiments_auxiliary_functions import run_experiment, compute_convergence_rates
from global_ import *
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = CHUNKSIZE
mpl.rcParams['axes.linewidth'] = 3

fw_step_size_rules = [{"step type": "open-loop", "a": 4, "b": 1, "c": 4, "d": 1}]

rhos = [-1, 2]

iterations = 1001
dimensions = list(np.linspace(5, iterations, 150, dtype=np.int64))
n_iterates = 100
for rho in rhos:

    convergence_rates = []
    for dimension in dimensions:
        if rho == 2:
            file_name = "locally_accelerated_convergence_rate_exterior"
            feasible_region, objective_function = polytope_experiment(dimension, rho)
        elif rho == -1:
            file_name = "locally_accelerated_convergence_rate_interior"
            feasible_region, objective_function = probability_simplex_interior_fast_ls_ss(dimension)

        primal_gaps, _ = run_experiment(iterations + n_iterates, objective_function, feasible_region,
                                        run_more=int(60 * iterations),
                                        fw_step_size_rules=fw_step_size_rules)
        primal_gaps = primal_gaps[0]
        convergence_rates.append(compute_convergence_rates(primal_gaps, n_iterates)[:iterations])

    contour_plotter(x_values=np.array(dimensions),
                    y_values=np.array(range(iterations)),
                    z_values=np.array(convergence_rates),
                    levels_min=0.8,
                    levels_max=2.05,
                    n_levels=9,
                    ticks_min=0.8,
                    ticks_max=2.05,
                    n_ticks=9,
                    directory="experiments/figures/locally_accelerated_convergence_rate",
                    file_name=file_name)


# Comparison of AFW and DIFW when the feasible region is a subset of the Boolean hypercube and the objective is
# uniformly convex.

from all_functions.plotting import primal_gap_plotter, determine_y_lims, only_min
from all_functions.problem_settings import polytope_experiment, probability_simplex_interior_fast_ls_ss
from all_functions.experiments_auxiliary_functions import run_experiment
from global_ import *
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = CHUNKSIZE
mpl.rcParams['axes.linewidth'] = LINEWIDTH

afw_step_size_rules = [
    # {"step type": "line-search afw"},
    {"step type": "open-loop", "a": 2, "b": 1, "c": 2, "d": 1},
    {"step type": "open-loop", "a": 4, "b": 1, "c": 4, "d": 1},
    # {"step type": "open-loop", "a": 8, "b": 1, "c": 8, "d": 1},
]

difw_step_size_rules = [
    # {"step type": "line-search difw probability simplex"},
    {"step type": "open-loop", "a": 2, "b": 1, "c": 2, "d": 1},
    # {"step type": "open-loop", "a": 4, "b": 1, "c": 4, "d": 1},
    {"step type": "open-loop", "a": 8, "b": 1, "c": 8, "d": 1},
]

rhos = [-1, 2 / DIMENSION, 2]

for rho in rhos:
    legend = False
    if rho == rhos[-1]:
        legend = True

    if rho != -1:
        file_name = "probability_simplex_afw_difw_rho_" + str(rho)
        feasible_region, objective_function = polytope_experiment(DIMENSION, rho)
    elif rho == -1:
        file_name = "probability_simplex_afw_difw_interior"
        feasible_region, objective_function = probability_simplex_interior_fast_ls_ss(DIMENSION)



    primal_gaps, labels = run_experiment(ITERATIONS, objective_function, feasible_region, run_more=RUN_MORE,
                                         afw_step_size_rules=afw_step_size_rules,
                                         difw_step_size_rules=difw_step_size_rules)


    primal_gaps = only_min(primal_gaps)
    primal_gap_plotter(y_data=primal_gaps,
                       labels=labels,
                       iterations=ITERATIONS,
                       file_name=file_name,
                       x_lim=(1, ITERATIONS),
                       y_lim=determine_y_lims(primal_gaps),
                       y_label=r'$\mathrm{min}_i  \ h_i$',
                       directory="figures/afw_difw",
                       legend=legend
                       )


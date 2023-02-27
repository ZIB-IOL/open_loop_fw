# Uniformly convex feasible region, not strongly convex objective, and unconstrained optimum in the exterior experiment.


import random
import autograd.numpy as np
from all_functions.plotting import primal_gap_plotter, determine_y_lims, only_min
from all_functions.problem_settings import uniformly_convex
from all_functions.experiments_auxiliary_functions import run_experiment
from global_ import *
import matplotlib as mpl

random.seed(RANDOM)
np.random.seed(RANDOM)

mpl.rcParams['agg.path.chunksize'] = CHUNKSIZE
mpl.rcParams['axes.linewidth'] = LINEWIDTH

ps = [2, 3, 5]
location = "exterior"
convexity = "not uniformly convex"


for p in ps:
    legend = False
    if p == ps[-1]:
        legend = True
    file_name = "lp" + "_" + str(p) + "_ball_" + "location" + "_" + str(location)
    feasible_region, objective_function, cst = uniformly_convex(DIMENSION, p=p, location=location, convexity=convexity)

    fw_step_size_rules = [
        {"step type": "line-search"},
        {"step type": "short-step"},
        {"step type": "open-loop", "a": 1, "b": 1, "c": 1, "d": 1},
        {"step type": "open-loop", "a": 2, "b": 1, "c": 2, "d": 1},
        {"step type": "open-loop", "a": 4, "b": 1, "c": 4, "d": 1},
        {"step type": "open-loop", "a": 6, "b": 1, "c": 6, "d": 1},
        {"step type": "open-loop constant", "cst": cst}
    ]
    primal_gaps, labels = run_experiment(ITERATIONS, objective_function, feasible_region, run_more=RUN_MORE,
                                         fw_step_size_rules=fw_step_size_rules)
    primal_gaps = only_min(primal_gaps)
    primal_gap_plotter(y_data=primal_gaps,
                       labels=labels,
                       iterations=ITERATIONS,
                       file_name=file_name,
                       x_lim=(1, ITERATIONS),
                       y_lim=determine_y_lims(primal_gaps),
                       y_label=r'$\mathrm{min}_i  \ h_i$',
                       directory="figures/non_polytope",
                       legend=legend
                       )

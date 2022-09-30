from all_functions.plotting import primal_gap_plotter, determine_y_lims, only_min
from all_functions.problem_settings import polytope_experiment

from all_functions.experiments_auxiliary_functions import run_experiment
from global_ import *
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = CHUNKSIZE
mpl.rcParams['axes.linewidth'] = LINEWIDTH

fw_step_size_rules = [
    {"step type": "line search"},
    {"step type": "open loop", "a": 2, "b": 1, "c": 2, "d": 1},
]

rhos = [0.25]

for rho in rhos:
    legend = False
    if rho == rhos[-1]:
        legend = True
    file_name = "probability_simplex_rho_" + str(rho)

    feasible_region, objective_function = polytope_experiment(DIMENSION, rho)

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
                       directory="figures/presentation",
                       legend=legend
                       )
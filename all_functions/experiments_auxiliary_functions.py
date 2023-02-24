from all_functions.frank_wolfe import frank_wolfe, decomposition_invariant_frank_wolfe, away_step_frank_wolfe, \
    momentum_guided_frank_wolfe
import numpy as np
from scipy import stats


def compute_convergence_rates(data, n_iterates):
    """Computes the convergence rate of the data according to the order estimation procedure in [1]

    Args:
        data
        n_iterates:
            The average over how many iterates we want to compute the converge rate

    References:
        [1] Senning, Jonathan R. "Computing and Estimating the Rate of Convergence" (PDF). gordon.edu.
        Retrieved 2021-05-18.
    """
    data = np.log(data)
    xdata = np.asarray(list(np.log(range(1, len(data) + 1))))
    convergence_rates = [-stats.linregress(xdata[i:i + n_iterates], data[i:i + n_iterates])[0] for i in
                         range(0, len(data) - n_iterates)]

    return convergence_rates


def translate_step_types(current_label, step):
    if step["step type"] == "open-loop":
        current_label = current_label + " " + "open-loop with" + " " + r"$\ell={}$".format(str(int(step["a"])))
    if step["step type"] == "open-loop constant":
        current_label = current_label + " " + "constant"
    elif step["step type"] in ["line-search", "line-search difw probability simplex", "line-search afw"]:
        current_label = current_label + " " + "line-search"
    elif step["step type"] in ["short-step", "short-step difw probability simplex", "short-step afw"]:
        current_label = current_label + " " + "short-step"
    return current_label


def run_experiment(iterations,
                   objective_function,
                   feasible_region,
                   run_more: int = 0,
                   fw_step_size_rules: list = [],
                   difw_step_size_rules: list = [],
                   afw_step_size_rules: list = [],
                   mfw_step_size_rules: list = []
                   ):
    """
    Minimizes objective_function over feasible_region.

    Args:
        iterations: int
            The number of iterations.
        objective_function
        feasible_region
        run_more: int, Optional
            Number of additional ITERATIONS to run to determine f(x*). (Default is 0.)
        fw_step_size_rules: list
            The types of FW step-size rules we want to run. (Default is [].)
        difw_step_size_rules: list
            The types of DIFW step-size rules we want to run. (Default is [].)
        afw_step_size_rules: list
            The types of AFW step-size rules we want to run. (Default is [].)
        mfw_step_size_rules: list
            The types of MFW step-size rules we want to run. (Default is [].)

    Returns:
        Returns a list of lists of primal gaps and a list of labels.
    """
    labels = []
    data = []
    for step in fw_step_size_rules:
        current_label = translate_step_types("FW", step)

        iterate_list, loss_list, fw_gap_list, x, x_p_list = frank_wolfe(feasible_region=feasible_region,
                                                                        objective_function=objective_function,
                                                                        n_iters=(int(iterations + run_more)),
                                                                        step=step)

        if run_more == 0:
            data_list = loss_list
        else:
            data_list = [loss_list[i] - loss_list[-1] for i in range(len(loss_list))][:iterations]
        data.append(data_list)
        labels.append(current_label)

    for step in mfw_step_size_rules:
        current_label = translate_step_types("MFW", step)

        iterate_list, loss_list, fw_gap_list, x, x_p_list = momentum_guided_frank_wolfe(
            feasible_region=feasible_region, objective_function=objective_function,
            n_iters=(int(iterations + run_more)), step=step)

        if run_more == 0:
            data_list = loss_list
        else:
            data_list = [loss_list[i] - loss_list[-1] for i in range(len(loss_list))][:iterations]
        data.append(data_list)
        labels.append(current_label)

    for step in afw_step_size_rules:
        current_label = translate_step_types("AFW", step)
        iterate_list, loss_list, fw_gap_list, x, x_p_list = away_step_frank_wolfe(feasible_region=feasible_region,
                                                                                  objective_function=objective_function,
                                                                                  n_iters=(
                                                                                      int(iterations + run_more)),
                                                                                  step=step)

        if run_more == 0:
            data_list = loss_list
        else:
            data_list = [loss_list[i] - loss_list[-1] for i in range(len(loss_list))][:iterations]
        data.append(data_list)
        labels.append(current_label)

    for step in difw_step_size_rules:
        current_label = translate_step_types("DIFW", step)
        iterate_list, loss_list, fw_gap_list, x, x_p_list = decomposition_invariant_frank_wolfe(
            feasible_region=feasible_region,
            objective_function=objective_function,
            n_iters=(int(iterations + run_more)),
            step=step)

        if run_more == 0:
            data_list = loss_list
        else:
            data_list = [loss_list[i] - loss_list[-1] for i in range(len(loss_list))][:iterations]
        data.append(data_list)
        labels.append(current_label)

    return data, labels

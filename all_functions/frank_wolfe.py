from all_functions.auxiliary_functions import fd
from all_functions.feasible_region import away_oracle, vertex_among_active_vertices

import autograd.numpy as np


def frank_wolfe(feasible_region,
                objective_function,
                step: dict,
                n_iters: int = 100):
    """Performs Frank-Wolfe/the herding algorithm.

        Args:
            feasible_region:
                The type of feasible region.
            objective_function: Optional
                The type of objective function.
            step: dict
                A dictionnary containing the information about the step type. The dictionary can have the following arg-
                uments:
                    "step type": Choose from "open-loop", "exact", "line-search", "short-step".
                Additional Arguments:
                    For "open-loop", provide integer values for the keys "a", "b", "c" that affect the step type as
                    follows: a / (b * iteration + c)
                    For "line-search", provide an integer for the number of exhaustive search steps for the key
                    "number of iterations for line-search".
            n_iters: integer, Optional
                The number of iterations. (Default is 100.)

        Returns:
            iterate_list: list
                Returns a list containing the iterate at each iteration.
            loss_list: list
                Returns a list containing the loss at each iteration.
            fw_gap_list: list
                Returns a list containing the FW gap at each iteration.
            x:
                Returns x, the final iterate of the algorithm
            x_p_list:
                Returns a list containing the values of ||x_t - p_t|| at each iteration.

        References:
            [1] "Marguerite Frank, Philip Wolfe, et al. An algorithm for quadratic programming. Naval research logistics
            quarterly, 3(1-2):95–110, 1956."
    """

    x = feasible_region.initial_point()
    loss_list = []
    fw_gap_list = []
    iterate_list = []
    x_p_list = []

    for i in range(1, n_iters):
        if isinstance(x, np.ndarray):
            gradient = objective_function.evaluate_gradient(x)
            p_fw, fw_gap, x_p = feasible_region.linear_minimization_oracle(gradient, x)
            x_p_list.append(x_p)
            try:
                scalar = objective_function.compute_step_size(i, x, p_fw, gradient, step=step)
            except:
                break
        else:
            p_fw, fw_gap = feasible_region.linear_minimization_oracle(objective_function, x)
            scalar = objective_function.compute_step_size(i, x, p_fw, step=step)
        if isinstance(x, np.ndarray):
            x = (1 - scalar) * x.flatten() + scalar * p_fw.flatten()
        else:
            x.update(p_fw, scalar=scalar)
        loss = objective_function.evaluate_loss(x)
        iterate_list.append(x)
        loss_list.append(loss)
        fw_gap_list.append(fw_gap)
        if loss < 10e-60:
            break
    return iterate_list, loss_list, fw_gap_list, x, x_p_list


def away_step_frank_wolfe(feasible_region,
                          objective_function,
                          step: dict,
                          n_iters: int = 100):
    """Performs Away-Step Frank-Wolfe.

        Args:
            feasible_region:
                The type of feasible region.
            objective_function: Optional
                The type of objective function.
            step: dict
                A dictionnary containing the information about the step-size rule.
            n_iters: integer, Optional
                The number of iterations. (Default is 100.)

        Returns:
            iterate_list: list
                Returns a list containing the iterate at each iteration.
            loss_list: list
                Returns a list containing the loss at each iteration.
            fw_gap_list: list
                Returns a list containing the FW gap at each iteration.
            x:
                Returns x, the final iterate of the algorithm
            x_p_list:
                Returns a list containing the values of ||x_t - p_t|| at each iteration.

        References:
            [1] "Simon Lacoste-Julien and Martin Jaggi. On the global linear convergence of frank-wolfe optimization
            variants. arXiv preprint arXiv:1511.05932, 2015."
    """

    x = feasible_region.initial_point().flatten()
    L = objective_function.L
    diameter = feasible_region.diameter
    active_vertices = fd(x)
    lambdas = np.array([[1.0]])
    loss_list = []
    fw_gap_list = []
    iterate_list = []
    x_p_list = []
    l = 1

    for i in range(1, n_iters):
        gradient = objective_function.evaluate_gradient(x).flatten()
        p_fw, fw_gap, x_p = feasible_region.linear_minimization_oracle(gradient, x)
        direction_fw = p_fw.flatten() - x.flatten()
        p_a, index_a = away_oracle(active_vertices, gradient)
        direction_a = x.flatten() - p_a.flatten()
        if gradient.T.dot(direction_fw) <= gradient.T.dot(direction_a):
            direction = direction_fw
            max_step = 1
        else:
            direction = direction_a
            max_step = lambdas[index_a, 0] / (1 - lambdas[index_a, 0])

        try:
            eta = objective_function.compute_step_size(i, x, direction, gradient, step=step)
        except:
            break
        # eta = objective_function.compute_step_size(l, x, direction, gradient, step=step)
        gamma = min(eta, max_step)

        # fw step
        if gradient.T.dot(direction_fw) <= gradient.T.dot(direction_a):
            fw_vertex_active_index = vertex_among_active_vertices(active_vertices, p_fw)

            # delete all other vertices if we take a full length step
            if gamma == max_step:
                active_vertices = fd(p_fw)
                lambdas = np.array([[1.0]])

            else:
                lambdas = lambdas * (1 - gamma)
                # fw vertex not in active set
                if fw_vertex_active_index is None:
                    # add the new fw vertex to the active set
                    active_vertices = np.hstack((active_vertices, fd(p_fw)))
                    # add lambda for the new fw vertex
                    lambdas = np.vstack((lambdas, np.array([[gamma]])))
                # fw vertex in active set
                else:
                    # update lamba for fw vertex
                    lambdas[fw_vertex_active_index, 0] += gamma
        # away step
        else:
            # update all vertices
            lambdas = lambdas * (1 + gamma)

            # away vertex has to be deleted
            if gamma == max_step:
                if index_a == 0:
                    lambdas = lambdas[1:, :]
                    active_vertices = active_vertices[:, 1:]
                elif index_a == lambdas.shape[0] - 1:
                    lambdas = lambdas[:-1, :]
                    active_vertices = active_vertices[:, -1:]
                else:
                    lambdas = np.vstack((lambdas[:index_a, :], lambdas[index_a + 1:, :]))
                    active_vertices = np.hstack((active_vertices[:, :index_a], active_vertices[:, index_a + 1:]))
            # update away vertex
            else:
                lambdas[index_a, 0] -= gamma

        if step["step type"] == "open-loop":
            if (eta ** 2 - gamma ** 2) * L * diameter ** 2 >= (eta - gamma) * gradient.T.dot(
                    direction_a - direction_fw):
                l += 1
            else:
                pass

        x = x + gamma * direction.flatten()
        loss = objective_function.evaluate_loss(x)
        iterate_list.append(x)
        loss_list.append(loss)

        if step["step type"] != "open-loop" and i > 1 and loss_list[-1] > 10e-10:
            assert loss_list[-1] - 10e-10 <= loss_list[-2], "Increse in loss should never happen."
        fw_gap_list.append(fw_gap)

    return iterate_list, loss_list, fw_gap_list, x, x_p_list


def decomposition_invariant_frank_wolfe(feasible_region,
                                        objective_function,
                                        step: dict,
                                        n_iters: int = 100,
                                        epsilon: float = 1e-16):
    """Performs Decomposition-Invariant Frank-Wolfe.

        Args:
            feasible_region:
                The type of feasible region.
            objective_function:
                The type of objective function.
            step: dict
                A dictionnary containing the information about the step-size rule.
            n_iters: integer, Optional
                The number of iterations. (Default is 100.)
            epsilon: float, Optional
                Used as a tolerance in the construction of gradient tilde.

        Returns:
            iterate_list: list
                Returns a list containing the iterate at each iteration.
            loss_list: list
                Returns a list containing the loss at each iteration.
            fw_gap_list: list
                Returns a list containing the FW gap at each iteration.
            x:
                Returns x, the final iterate of the algorithm
            x_p_list:
                Returns a list containing the values of ||x_t - p_t|| at each iteration.

        References:
            [1] "Dan Garber and Ofer Meshi. Linear-memory and decomposition-invariant linearly convergent conditional
            gradient algorithm for structured polytopes. Advances in neural information processing systems,
            29:1001–1009, 2016."
    """

    x = feasible_region.initial_point()
    gradient = objective_function.evaluate_gradient(x)
    x, _, _ = feasible_region.linear_minimization_oracle(gradient, x)
    loss_list = []
    fw_gap_list = []
    iterate_list = []
    x_p_list = []

    for i in range(0, n_iters):
        gradient = objective_function.evaluate_gradient(x)
        gradient_difw = gradient.copy()
        gradient_difw[:, np.newaxis][(np.abs(x) <= epsilon)] = - 1e+16
        p, fw_gap, x_p = feasible_region.linear_minimization_oracle(gradient, x)
        p_difw, _, _ = feasible_region.linear_minimization_oracle(-gradient_difw, x)
        direction = (p - p_difw).flatten()
        x_p_list.append(x_p)
        try:
            scalar = objective_function.compute_step_size(i, x, direction, gradient, step=step)
        except:
            break
        if step["step type"] in ["line-search difw probability simplex", "short-step difw probability simplex"]:
            x = x.flatten() + scalar * direction
        elif step["step type"] == "open-loop":
            j = 0
            while True:
                if 2 ** (-j) <= scalar:
                    scalar = 2 ** (-j)
                    break
                else:
                    j += 1
            x = x.flatten() + scalar * (p.flatten() - p_difw.flatten())

        loss = objective_function.evaluate_loss(x)
        if loss < 10e-30:
            break

        iterate_list.append(x)
        loss_list.append(loss)
        fw_gap_list.append(fw_gap)

    return iterate_list, loss_list, fw_gap_list, x, x_p_list


def momentum_guided_frank_wolfe(feasible_region,
                                objective_function,
                                step: dict,
                                n_iters: int = 100):
    """Performs momentum-guided Frank-Wolfe.

        Args:
            feasible_region:
                The type of feasible region.
            objective_function: Optional
                The type of objective function.
            step: dict
                A dictionnary containing the information about the step type. The dictionary can have the following arg-
                uments:
                    "step type": Choose from "open-loop".
                Additional Arguments:
                    For "open-loop", provide integer values for the keys "a", "b", "c" that affect the step type as
                    follows: a / (b * iteration + c)
            n_iters: integer, Optional
                The number of iterations. (Default is 100.)

        Returns:
            iterate_list: list
                Returns a list containing the iterate at each iteration.
            loss_list: list
                Returns a list containing the loss at each iteration.
            fw_gap_list: list
                Returns a list containing the FW gap at each iteration.
            x:
                Returns x, the final iterate of the algorithm
            x_p_list:
                Returns a list containing the values of ||x_t - p_t|| at each iteration.

        References:
            [1] "Li, B., Coutino, M., Giannakis, G.B. and Leus, G., 2021. A momentum-guided frank-wolfe algorithm.
            IEEE Transactions on Signal Processing, 69, pp.3597-3611."
    """

    x = feasible_region.initial_point()
    v = x
    theta = np.zeros(x.shape)
    gradient = theta
    loss_list = []
    fw_gap_list = []
    iterate_list = []
    x_p_list = []

    for i in range(1, n_iters):
        scalar = objective_function.compute_step_size(i, x, v, gradient, step=step)

        y = (1-scalar)*x + scalar*v
        gradient = objective_function.evaluate_gradient(y)
        theta = (1-scalar)*theta.flatten() + scalar*gradient.flatten()
        v, fw_gap, x_p = feasible_region.linear_minimization_oracle(theta, x)
        x_p_list.append(x_p)
        x = (1-scalar)*x.flatten() + scalar*v.flatten()
        loss = objective_function.evaluate_loss(x)
        iterate_list.append(x)
        loss_list.append(loss)
        fw_gap_list.append(fw_gap)
        if loss < 10e-60:
            break
    return iterate_list, loss_list, fw_gap_list, x, x_p_list

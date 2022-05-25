import numpy as np

from all_functions.feasible_region import lpnorm, UnitSimplex, LpBall
from all_functions.objective_function import SquaredLossFinDim


def polytope_experiment(dimension: int, rho: float):
    """Creates a problem setting for which FW with ls and ss converge linearly or sublinearly and FW with ol
    converges at a rate of 1/t² after a certain number of ITERATIONS. Specificaly,
        f(x) = 1/2 ||x-b||_2^2,
        where b = rho (0, ... , 0, 1, ... 1)^T,

        and
        the feasible region is the probability simplex.

    The optimum lies in the boundary.

    If rho > 1/2 ==> linear for ls and os and if rho < 1/2 ==> sublinear.
        """
    A = np.identity(dimension)

    b = np.ones(dimension)
    b[:int(dimension / 2)] = 0

    b = b * rho

    objective_function = SquaredLossFinDim(A=A, b=b)
    constraint_set = UnitSimplex(dimension=dimension)
    return constraint_set, objective_function


def probability_simplex_interior_fast_ls_ss(dimension):
    """Creates a problem setting for which FW with ls and ss converge linearly and FW with ol
    converges at a rate of 1/t² after a certain number of ITERATIONS. Specificaly,
        f(x) = 1/2 ||x-b||_2^2,
        where b = (1/dimension, ... , 1/dimension, ... 1/dimension)^T,

        and
        the feasible region is the probability simplex.

    The unconstrained optimum lies in the interior.
        """

    A = np.identity(dimension)
    b = np.ones((dimension, 1))
    b = b / lpnorm(b, p=1)

    objective_function = SquaredLossFinDim(A=A, b=b)
    constraint_set = UnitSimplex(dimension=dimension)
    return constraint_set, objective_function


def uniformly_convex(dimension, p=2, location: str = "interior", convexity: str = "strong"):
    """Creates a problem setting for which FW with ls and ss converge linearly and FW with ol
    converges at a rate of 1/t² after a certain number of ITERATIONS. Specificaly,
        f(x) = 1/2 ||x-b||_2^2,
        where b is a non-sparse random vector.

        and
        the feasible region is the probability simplex.

    The unconstrained optimum lies in location, which is in ["interior", "boundary", "exterior"].
        """
    A = np.identity(dimension) + 0.1 * np.random.random((dimension, dimension))

    if convexity == "not uniformly convex":
        A = np.triu(A)
        A[-int(dimension / 3 + 2):, :] = 0

    A = A.T.dot(A)

    if convexity == "strong":
        A = A + np.identity(dimension)
    x = np.random.random((dimension, 1)) + 0.25 * np.ones((dimension, 1))

    cst = None
    if location == "interior":
        x = x / (2 * lpnorm(x, p=p))
    if location == "boundary":
        x = x / (lpnorm(x, p=p))
    if location == "exterior":
        A = A / np.sqrt(max(np.linalg.eigvalsh(A.T.dot(A))))
        x = 1.5 * x / (lpnorm(x, p=p))

    b = A.dot(x)

    objective_function = SquaredLossFinDim(A=A, b=b)
    constraint_set = LpBall(dimension=dimension, p=p)

    if location == "exterior":
        L = objective_function.L
        alpha = 1
        lmbda = np.linalg.norm(b) - 1
        cst = (alpha * lmbda) / (2 * L)
    return constraint_set, objective_function, cst

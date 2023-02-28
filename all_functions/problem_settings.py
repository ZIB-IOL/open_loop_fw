import autograd.numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from all_functions.feasible_region import lpnorm, UnitSimplex, LpBall, NuclearNormBall
from all_functions.objective_function import SquaredLossFinDim, LogisticLossFinDim, \
    HuberLossCollaborativeFilteringFinDim

import os


def polytope_experiment(dimension: int, rho: float):
    """Creates a problem setting for which FW with ls and ss converge linearly or sublinearly and FW with ol
    converges at a rate of 1/t² after a certain number of iterations. Specificaly,
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
    feasible_region = UnitSimplex(dimension=dimension)
    return feasible_region, objective_function


def probability_simplex_interior_fast_ls_ss(dimension):
    """Creates a problem setting for which FW with ls and ss converge linearly and FW with ol
    converges at a rate of 1/t² after a certain number of iterations. Specificaly,
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
    feasible_region = UnitSimplex(dimension=dimension)
    return feasible_region, objective_function


def uniformly_convex(dimension, p=2, location: str = "interior", convexity: str = "strong"):
    """Creates a problem setting for which FW with ls and ss converge linearly and FW with ol
    converges at a rate of 1/t² after a certain number of iterations. Specificaly,
        f(x) = 1/2 ||Ax-b||_2^2,
        where A is a random matrix and b is a non-sparse random vector and the feasible region is an lp ball.

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
    feasible_region = LpBall(dimension=dimension, p=p)

    if location == "exterior":
        L = objective_function.L
        alpha = 1
        lmbda = np.linalg.norm(b) - 1
        cst = (alpha * lmbda) / (2 * L)
    return feasible_region, objective_function, cst


def uniformly_convex_logistic_regression(samples, dimension, p=2):
    """Creates objective
        f(x) = (1/m)sum_i log(1 + exp(-b_i * a_i^T * x)),
    where a_i and b_i are random vectors and the region is an lp ball. The optimum is always in the exterior.
    """
    # Generate a random dataset
    A = np.random.rand(samples, dimension) / samples  # 100 samples with 3 features
    b = np.random.randint(0, 2, size=samples)  # binary labels
    b[(b == 0)] = -1

    # # Add a bias term
    # A = np.hstack((np.ones((A.shape[0], 1)), A))

    objective_function = LogisticLossFinDim(A=A, b=b)
    feasible_region = LpBall(dimension=A.shape[1], p=p)

    return feasible_region, objective_function


def gisette(p=2):
    """Create the feasible region and objective function for logistic regression for the gisette dataset.
    The data set has 2000 samples and 5000 features.

    References:
        [1] Isabelle Guyon, Steve R. Gunn, Asa Ben-Hur, Gideon Dror, 2004. Result analysis of the NIPS 2003 feature
        selection challenge. In: NIPS.
    """


    # Load the data files into numpy arrays
    A = np.loadtxt(os.path.dirname(__file__) + '/../datasets/gisette/gisette_train.data')
    A = A[:2000, :]
    b = np.loadtxt(os.path.dirname(__file__) + '/../datasets/gisette/gisette_train.labels')
    b = b[:2000]

    scaler = StandardScaler()
    A = scaler.fit_transform(A)
    m, n = A.shape
    print("Dimensions: ", (m, n))

    objective_function = LogisticLossFinDim(A=A, b=b)
    feasible_region = LpBall(dimension=A.shape[1], p=p)

    return feasible_region, objective_function


def movielens(radius: int = 5000):
    """Create the feasible region and objective function for collaborative filtering for the movielens dataset.
    The matrix storing the movie reviews is of dimension 943 x 1682.


    References:
        [1] F. M. Harper and J. A. Konstan. The MovieLens datasets: History and context. ACM Transactions on Interactive
        Intelligent Systems, 5(4):19:1–19:19, 2015.
    """


    # Load the data files into numpy arrays
    data = pd.read_csv(os.path.dirname(__file__) + '/../datasets/movielens100k.csv',
                       names=['user id', 'item id', 'rating', 'timestamp'])
    A = pd.pivot_table(data, values='rating', index='user id', columns='item id').values
    m, n = A.shape
    print("Dimensions: ", (m, n))

    objective_function = HuberLossCollaborativeFilteringFinDim(A=A)
    feasible_region = NuclearNormBall(m, n, radius=radius)



    return feasible_region, objective_function

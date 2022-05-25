
import numpy as np

from all_functions.auxiliary_functions import fd, get_non_zero_indices
from all_functions.hilbert_space_element import ElementMarginalPolytope
from all_functions.objective_function import SquaredLoss


class HilbertSpaceWhaba:
    """A class used to represent the Hilbert space from [1, 2].

    Let Y = [0, 1]. Define the RKHS
    H = {x: Y -> R | x(y) = sqrt(2) sum_j^infty a_j cos(2pi j y) + b_j sin(2pi j y), f' in L^2(Y), a_j, b_j in R}.
    The scalar product is <w, x> = integral_Y w'(y) x'(y) dx.
    The associated kernel is k(y, z) = sum_j=1^infty 2 / (2pi j)^2 cos(2pi j (y - z)).
    The associated feature map is Phi: Y -> H, Phi(y) = k(t, y) = sum_j=1^infty 2 / (2pi j)^2 cos(2pi j (t - y)).
    Elements of the marginal polytope C are of the form x = sum_i^n w_i Phi(y_i).

    Attributes:
        iterations_lmo: integer
                The number of exhaustive search steps we conduct to solve the linear minimization problem.

    Methods:
        linear_minimization_oracle(f: object, iterations)
            Solves the linear minimization problem min_p in C <x, y>.
        initial_point()
            Returns the initial vertex.

    References:
        [1] Francis Bach, Simon Lacoste-Julien, and Guillaume Obozinski. On the equivalence between herding and
            conditional gradient algorithms. arXiv preprint arXiv:1203.4523, 2012.
        [2] Grace Wahba. Spline models for observational data. SIAM, 1990.
    """

    def __init__(self, iterations_lmo: int = 100):
        self.iterations_lmo = iterations_lmo
        assert self.iterations_lmo > 0, "Number of ITERATIONS needs to be greater than 0."

    def linear_minimization_oracle(self,
                                   objective_function: SquaredLoss,
                                   x: ElementMarginalPolytope,
                                   ):
        """Solves the linear minimization problem min_p in C <x - mu, p>.

        Returns the element p_t in the Hilbert space H such that <x - mu, p> is minimized.

        Args:
            objective_function: instance of an objective function class
                An instance of an objective function class.
            x: instance of ElementMarginalPolytope
                An instance of ElementMarginalPolytope.

        Returns:
            p: instance of ElementMarginalPolytope
                An instance of ElementMarginalPolytope which is the approximate solution to the linear minimization
                problem.
            wolfe_gap: float
                The Frank-Wolfe gap.
        """

        optimal_value = 10e16
        optimal_point = None
        p = None
        for idx in range(0, self.iterations_lmo):
            current_point = idx / self.iterations_lmo
            current_p = ElementMarginalPolytope(np.array([1]), np.array([current_point]))
            current_value = objective_function.evaluate_gradient(x, current_p)
            if current_value < optimal_value:
                optimal_value = current_value
                optimal_point = current_point
                p = current_p

        assert optimal_point is not None, "Linear minimization oracle did not find a correct point."
        assert p is not None, "Linear minimization oracle did not find a correct point."

        wolfe_gap = objective_function.evaluate_gradient(x, x) - optimal_value
        return p, wolfe_gap

    def initial_point(self):
        """Returns the initial vertex."""
        x = ElementMarginalPolytope()
        return x


def lpnorm(vector, p):
    """Computes the Lp norm of a vector"""
    if p == 1:
        return np.linalg.norm(vector, ord=1)
    elif p == 2:
        return np.linalg.norm(vector, ord=2)
    elif p == -1:
        return np.linalg.norm(vector, ord=np.inf)
    else:
        vector = vector.flatten()
        absolute_vector = np.abs(vector)
        power_vector = absolute_vector ** p
        solution = float(np.sum(power_vector) ** (1 / p))
        return solution


class LpBall:
    """A class used to represent the Lp ball of radius 1.

    Args:
        dimension: integer, Optional
            The number of data points.
        p: float, Optional
            Set p = -1 for L infinity ball. (Default is 1.0)

    Methods:
        linear_minimization_oracle(gradient: np.ndarray, x: np.ndarray)
            Solves the linear minimization problem min_g in Lp <grad f.T(x), g>.
        membership_oracle(x,epsilon: float):
            Determines whether x is in the feasible region, on the boundary, or exterior the feasible region.
        initial_point()
            Returns the initial vertex.
    """

    def __init__(self, dimension: int = 400, p: float = 1.0):
        self.dimension = dimension
        self.p = p
        if self.p > 1:
            self.q = 1 / (1 - 1 / self.p)
        self.diameter = 2

    def linear_minimization_oracle(self,
                                   gradient: np.ndarray,
                                   x: np.ndarray):
        """Solves the linear minimization problem min_g in Lp <grad f.T(x), g>.

        Args:
            gradient: np.ndarray
            x: np.ndarray

        Returns:
            p: np.ndarray
                The solution to the linear minimization problem.
            wolfe_gap: float
                The Frank-Wolfe gap.
            pt_xt: float
                ||x_t - p_t||.
        """
        if self.p == 1:
            tmp_pos = np.abs(gradient).argmax()
            sign = np.sign(gradient[tmp_pos])
            p = np.zeros(self.dimension)
            p[tmp_pos] = - sign
            assert np.linalg.norm(p, ord=1) <= 1, "p is not in the L1 ball."
            wolfe_gap = float(gradient.T.dot(fd(x)) - gradient.T.dot(fd(p)))
        elif self.p == -1:
            gradient = gradient.flatten()
            p = -np.sign(gradient)
            assert (np.abs(p) <= 1).all(), "p is not in the Linfinity ball."
            wolfe_gap = float(fd(gradient).T.dot(fd(x)) - fd(gradient).T.dot(fd(p)))
        else:
            # The solution to min_||f||_p <= 1 <f,g> is given by f_i = g_i^{q-1}/||g||_q^{q-1}.
            x = x.flatten()
            gradient = gradient.flatten()
            p = -np.sign(gradient) * np.abs(gradient) ** (self.q - 1) / (
                    (lpnorm(gradient, self.q)) ** (self.q - 1))
            wolfe_gap = float(gradient[:, np.newaxis].T.dot(x[:, np.newaxis])
                              - gradient[:, np.newaxis].T.dot(p[:, np.newaxis]))
            assert abs(lpnorm(p, self.p) - 1) < 10e-10, "p is not in the Lp ball."

        pt_xt = np.linalg.norm(x.flatten() - p.flatten())
        return p, wolfe_gap, pt_xt

    def membership_oracle(self, x: np.ndarray, epsilon: float = 10e-10):
        """Determines whether x is in the interior, boundary, or exterior of the feasible region."""
        norm = lpnorm(x, 1)
        if abs(1 - norm) <= epsilon:
            return "boundary"
        elif (1 - norm) > epsilon:
            return "interior"
        elif (1 - norm) < epsilon:
            return "exterior"

    def initial_point(self):
        """Returns the initial vertex."""
        x = np.zeros((self.dimension, 1))
        x[0] = 1
        return x


class UnitSimplex:
    """A class used to represent the probability simplex.

    Args:
        dimension: integer, Optional
            The number of data points.

    Methods:
        linear_minimization_oracle(gradient: np.ndarray, x: np.ndarray)
            Solves the linear minimization problem min_g in probability simplex <grad f.T(x), g>.
        membership_oracle(x,epsilon: float):
            Determines whether x is in the feasible region, on the boundary, or exterior the feasible region.
        initial_point()
            Returns the initial vertex.
    """

    def __init__(self, dimension: int = 400):
        self.dimension = dimension
        self.diameter = 2

    def linear_minimization_oracle(self,
                                   gradient: np.ndarray,
                                   x: np.ndarray):
        """Solves the linear minimization problem min_g in Lp <grad f.T(x), g>.

        Args:
            gradient: np.ndarray
            x: np.ndarray

        Returns:
            p: np.ndarray
                The solution to the linear minimization problem.
            wolfe_gap: float
                The Frank-Wolfe gap.
            pt_xt: float
                ||x_t - p_t||.
        """
        tmp_pos = gradient.argmin()
        p = np.zeros(self.dimension)
        p[tmp_pos] = 1
        assert self.membership_oracle(p) in ["boundary", "interior"], "p is not in the L1 ball."
        wolfe_gap = float(gradient.T.dot(fd(x)) - gradient.T.dot(fd(p)))

        pt_xt = np.linalg.norm(x.flatten() - p.flatten())
        return p, wolfe_gap, pt_xt

    def membership_oracle(self, x: np.ndarray, epsilon: float = 10e-10):
        """Determines whether x is in the interior, boundary, or exterior of the feasible region."""
        norm = lpnorm(x, 1)
        if (x >= -epsilon).all():
            if abs(1 - norm) <= epsilon:
                return "boundary"
            elif (1 - norm) > epsilon:
                return "interior"
        elif (1 - norm) < epsilon:
            return "exterior"

    def initial_point(self):
        """Returns the initial vertex."""
        x = np.zeros((self.dimension, 1))
        x[0] = 1
        return x


def away_oracle(active_vertices: np.ndarray, direction: np.ndarray):
    """Solves the maximization problem max_{i} in probability simplex or l1 ball <direction, active_vertices[:, i]>.

        Args:
            active_vertices: np.ndarray
                A matrix whose column vectors are vertices of the probability simplex
            direction: np.ndarray
                Gradient at x.

        Returns:
            active_vertices_idx: int
                Reference to the column in the active train_or_test corresponding to the away vertex.
            away_vertex: np.ndarray
                The away vertex.
    """
    tmp = active_vertices.T.dot(direction)
    active_vertices_idx = np.argmax(tmp)
    away_vertex = active_vertices[:, active_vertices_idx]
    return away_vertex, active_vertices_idx


def vertex_among_active_vertices(active_vertices: np.ndarray, fw_vertex: np.ndarray):
    """Checks if the fw_vertex is in the set of active vertices for l1 ball or probability simplex

    Args:
        active_vertices: cp.ndarray
            A matrix whose column vectors are vertices of the l1 ball.
        fw_vertex: cp.ndarray
            The Frank-Wolfe vertex.

    Returns:
        active_vertex_index:
            Returns the position of fw_vertex in active_vertices as an int. If fw_vertex is not a column of
            active_vertices, this value is None.
    """
    active_vertices = fd(active_vertices)
    index = get_non_zero_indices(fw_vertex)
    assert len(index) == 1, "Vertices should have exactly one non-zero entry."
    index = index[0]
    value = fd(fw_vertex)[index, 0]
    crucial_row = active_vertices[index, :]
    list_of_indices = get_non_zero_indices(crucial_row)
    assert len(list_of_indices) <= 2, "Vertices should not occur twice in active_vertices."
    for active_vertex_index in list_of_indices:
        if crucial_row[active_vertex_index] * value > 0:
            return active_vertex_index
    return None






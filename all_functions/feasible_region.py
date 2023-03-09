import autograd.numpy as np
from scipy.sparse.linalg import svds

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
        """Solves the linear minimization problem min_p in C <x - mu, fw_vertex>.

        Returns the element p_t in the Hilbert space H such that <x - mu, fw_vertex> is minimized.

        Args:
            objective_function: instance of an objective function class
                An instance of an objective function class.
            x: instance of ElementMarginalPolytope
                An instance of ElementMarginalPolytope.

        Returns:
            fw_vertex: instance of ElementMarginalPolytope
                An instance of ElementMarginalPolytope which is the approximate solution to the linear minimization
                problem.
            fw_gap: float
                The FW gap.
        """

        optimal_value = 10e16
        optimal_point = None
        fw_vertex = None
        for idx in range(0, self.iterations_lmo):
            current_point = idx / self.iterations_lmo
            current_p = ElementMarginalPolytope(np.array([1]), np.array([current_point]))
            current_value = objective_function.evaluate_gradient(x, current_p)
            if current_value < optimal_value:
                optimal_value = current_value
                optimal_point = current_point
                fw_vertex = current_p

        assert optimal_point is not None, "Linear minimization oracle did not find a correct point."
        assert fw_vertex is not None, "Linear minimization oracle did not find a correct point."

        fw_gap = objective_function.evaluate_gradient(x, x) - optimal_value
        return fw_vertex, fw_gap

    def initial_point(self):
        """Returns the initial vertex."""
        x = ElementMarginalPolytope()
        return x


def lpnorm(vector, p):
    """Computes the lp norm of a vector"""
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
    """A class used to represent the lp ball of radius 1.

    Args:
        dimension: integer, Optional
            The number of data points.
        p: float, Optional
            Set p = -1 for L infinity ball. (Default is 1.0.)
        radius: float, Optional
            (Default is 1.0.)

    Methods:
        linear_minimization_oracle(v: np.ndarray, x: np.ndarray)
            Solves the linear minimization problem min_g in lp <v, g>.
        membership_oracle(x,epsilon: float):
            Determines whether x is in the feasible region, on the boundary, or exterior the feasible region.
        initial_point()
            Returns the initial vertex.
    """

    def __init__(self, dimension: int = 400, p: float = 1.0, radius: float = 1.0):
        self.dimension = dimension
        self.p = p
        if self.p > 1:
            self.q = 1 / (1 - 1 / self.p)
        self.radius = radius
        self.diameter = 2 * self.radius

    def linear_minimization_oracle(self,
                                   v: np.ndarray,
                                   x: np.ndarray):
        """Solves the linear minimization problem min_g in lp <v, g>.

        Args:
            v: np.ndarray
            x: np.ndarray

        Returns:
            fw_vertex: np.ndarray
                The solution to the linear minimization problem.
            fw_gap: float
                The FW gap.
            distance_iterate_fw_vertex: float
                The distance between the iterate x and the FW vertex fw_vertex.
        """
        if self.p == 1:
            v = v.flatten()
            tmp_pos = np.abs(v).argmax()
            sign = np.sign(v[tmp_pos])
            fw_vertex = np.zeros(self.dimension)
            fw_vertex[tmp_pos] = - sign
            assert np.linalg.norm(fw_vertex, ord=1) <= 1, "p is not in the feasible region."
        elif self.p == -1:
            v = v.flatten()
            fw_vertex = -np.sign(v)
            assert (np.abs(fw_vertex) <= 1).all(), "p is not in the feasible region."

        else:
            # The solution to min_||f||_p <= 1 <f,g> is given by f_i = g_i^{q-1}/||g||_q^{q-1}.
            x = x.flatten()
            v = v.flatten()
            fw_vertex = -np.sign(v) * np.abs(v) ** (self.q - 1) / (
                    (lpnorm(v, self.q)) ** (self.q - 1))
            assert abs(lpnorm(fw_vertex, self.p) - 1) < 10e-10, "p is not in the feasible region."
        fw_gap = float(fd(v).T.dot(fd(x)) - fd(v).T.dot(fd(fw_vertex)))
        distance_iterate_fw_vertex = np.linalg.norm(x.flatten() - fw_vertex.flatten())
        return fw_vertex, fw_gap, distance_iterate_fw_vertex

    def membership_oracle(self, x: np.ndarray, epsilon: float = 10e-10):
        """Determines whether x is in the interior, boundary, or exterior of the feasible region."""
        norm = lpnorm(x, 1)
        if abs(self.radius - norm) <= epsilon:
            return "boundary"
        elif (self.radius - norm) > epsilon:
            return "interior"
        elif (self.radius - norm) < epsilon:
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
            The number of data points. (Default is 400.)
        radius: float, Optional
            (Default is 1.0.)

    Methods:
        linear_minimization_oracle(v: np.ndarray, x: np.ndarray)
            Solves the linear minimization problem min_g in probability simplex <v, g>.
        membership_oracle(x,epsilon: float):
            Determines whether x is in the feasible region, on the boundary, or exterior the feasible region.
        initial_point()
            Returns the initial vertex.
    """

    def __init__(self, dimension: int = 400, radius: float = 1.0):
        self.dimension = dimension
        self.radius = radius
        self.diameter = 2 * self.radius

    def linear_minimization_oracle(self,
                                   v: np.ndarray,
                                   x: np.ndarray):
        """Solves the linear minimization problem min_g in probability simplex <v, g>.

        Args:
            v: np.ndarray
            x: np.ndarray

        Returns:
            fw_vertex: np.ndarray
                The solution to the linear minimization problem.
            fw_gap: float
                The FW gap.
            distance_iterate_fw_vertex: float
                The distance between the iterate x and the FW vertex fw_vertex.
        """
        tmp_pos = v.argmin()
        fw_vertex = np.zeros(self.dimension)
        fw_vertex[tmp_pos] = 1
        assert self.membership_oracle(fw_vertex) in ["boundary", "interior"], "fw_vertex is not in the feasible region."
        fw_gap = float(v.T.dot(fd(x)) - v.T.dot(fd(fw_vertex)))

        pt_xt = np.linalg.norm(x.flatten() - fw_vertex.flatten())
        return fw_vertex, fw_gap, pt_xt

    def membership_oracle(self, x: np.ndarray, epsilon: float = 10e-10):
        """Determines whether x is in the interior, boundary, or exterior of the feasible region."""
        norm = lpnorm(x, 1)
        if (x >= -epsilon).all():
            if abs(self.radius - norm) <= epsilon:
                return "boundary"
            elif (self.radius - norm) > epsilon:
                return "interior"
        elif (self.radius - norm) < epsilon:
            return "exterior"

    def initial_point(self):
        """Returns the initial vertex."""
        x = np.zeros((self.dimension, 1))
        x[0] = 1
        return x


def away_oracle(active_vertices: np.ndarray, direction: np.ndarray):
    """Solves the maximization problem max_{i} in C <direction, active_vertices[:, i]>.

        Args:
            active_vertices: np.ndarray
            direction: np.ndarray

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
        active_vertices: np.ndarray
            A matrix whose column vectors are vertices of the l1 ball.
        fw_vertex: np.ndarray
            The Frank-Wolfe vertex.

    Returns:
        active_vertex_index:
            Returns the position of fw_vertex in active_vertices as an int. If fw_vertex is not a column of
            active_vertices, this value is None.
    """
    active_vertices = fd(active_vertices)
    num_cols = active_vertices.shape[1]
    fw_vertex = fd(fw_vertex)
    # Loop through the columns of active_vertices
    for i in range(num_cols):
        # Check if the ith column is identical to x
        if np.array_equal(active_vertices[:, i], fw_vertex):
            # Return the index of the identical column
            return i
    # If no identical column was found, return None
    return None

class NuclearNormBall:
    """A class used to represent the nuclear norm ball in R^{m x n}.

    Args:
        m: integer
        n: integer
        radius: float, Optional
            (Default is 1.0.)

    Methods:
        linear_minimization_oracle(v: np.ndarray, x: np.ndarray)
            Solves the linear minimization problem min_g in nuclear norm ball <v, g>.
        membership_oracle(x,epsilon: float):
            Determines whether x is in the feasible region, on the boundary, or exterior the feasible region.
        initial_point()
            Returns the initial vertex.
    """

    def __init__(self, m: int, n: int, radius: float = 1.0):
        self.m = m
        self.n = n
        self.radius = radius
        self.diameter = 2 * self.radius

    def linear_minimization_oracle(self, v: np.ndarray, x: np.ndarray):
        """Solves the linear minimization problem min_g in nuclear norm ball <v, g>.

        Args:
            v: np.ndarray
            x: np.ndarray

        Returns:
            fw_vertex: np.ndarray
                The solution to the linear minimization problem.
            fw_gap: float
                The FW gap.
            distance_iterate_fw_vertex: float
                The distance between the iterate x and the FW vertex fw_vertex.
        """

        G = np.reshape(-v, (self.m, self.n))
        u1, s, u2 = svds(G, k=1, which='LM')
        fw_vertex = np.reshape(self.radius * np.outer(u1, u2), len(v))

        assert self.membership_oracle(fw_vertex) in ["boundary", "interior"], "fw_vertex is not in the feasible region."
        fw_gap = float(v.T.dot(fd(x)) - v.T.dot(fd(fw_vertex)))

        distance_iterate_fw_vertex = np.linalg.norm(x.flatten() - fw_vertex.flatten())
        return fw_vertex, fw_gap, distance_iterate_fw_vertex

    def membership_oracle(self, x: np.ndarray, epsilon: float = 10e-10):
        """Determines whether x is in the interior, boundary, or exterior of the feasible region."""

        x_mat = np.reshape(x, (self.m, self.n))
        norm = np.linalg.norm(x_mat, 'nuc')
        if abs(self.radius - norm) <= epsilon:
            return "boundary"
        elif (self.radius - norm) > epsilon:
            return "interior"
        elif (self.radius - norm) < epsilon:
            return "exterior"

    def initial_point(self):
        """Returns the initial vertex."""
        x = self.radius * np.reshape(np.outer(np.identity(self.m)[0], np.identity(self.n)[0]), self.m * self.n)
        return x

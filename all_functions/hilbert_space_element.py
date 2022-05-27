import numpy as np
import math
from all_functions.auxiliary_functions import fd


class ElementMarginalPolytope:
    """A class used to represent the elements of the marginal polytope C.

    Let Y = [0, 1]. Define the RKHS
    H = {x: Y -> R | x(y) = sqrt(2) sum_j^infty a_j cos(2pi j y) + b_j sin(2pi j y), f' in L^2(Y), a_j, b_j in R}.
    The scalar product is <w, x> = integral_Y x'(y) w'(y) dy.
    The associated kernel is k(y, z) = sum_j=1^infty 2 / (2pi j)^2 cos(2pi j (y - z)).
    The associated feature map is Phi: Y -> H, Phi(y) = k(t, y) = sum_j=1^infty 2 / (2pi j)^2 cos(2pi j (t - y)).
    Elements of the marginal polytope C are of the form x(y) = sum_i^n w_i Phi(y_i).

    Attributes:
        weights: np.ndarray
            An array containing the weights of the element x of C, i.e., [w_1, ..., w_n].
        points: np.ndarray
            An array containing the points of the element x of C, i.e., [y_1, ..., y_n].

    Methods:
        update(new_weights: np.ndarray,  new_points: np.ndarray, scalar: float = 1)
            Updates weights and points.
        compute_scalar_product()
            Computes the scalar product of the element y of C with another element of C, that is, <w, y>.
        compute_scalar_product_with_mu(mu)
            Computes the scalar product with mu = sum_j=1 to len(mu[0]) a_j cos(2pi j x) + b_j sin(2pi j x).
        __copy__()
            Creates a copy of the instance.
        print_weights_points()
            Prints the weights and points.
        check_status_of_weights_and_points()
            Checks whether the weights and points are in the correct format.
        check_shape_of_weights_and_points()
        check_sum_weights()
        check_points_are_in_feasibility_region()

    """

    def __init__(self, weights: np.ndarray = None, points: np.ndarray = None):

        if weights is None:
            self.weights = np.array([1]).astype(np.float64)
        else:

            self.weights = weights.astype(np.float64)
        if points is None:
            self.points = np.array([0]).astype(np.float64)
        else:
            self.points = points.astype(np.float64)
        self.check_status_of_weights_and_points()

    def update(self, w, scalar: float = 1):
        """Updates weights and points.

        Args:
            w: instance of ElementMarginalPolytope
                An instance of ElementMarginalPolytope.
            scalar: float
                A float in [0, 1]. We multiply the weights of the current element of H with (1 - scalar) and the
                weights of the other element of H with scalar and take the combination of the two. (Default is 1.)
        """
        assert (0 <= scalar) & (scalar <= 1), "Scalar has to be in [0, 1]."
        if scalar == 1:
            self.weights = w.weights
            self.points = w.points
        else:
            self.weights *= (1 - scalar)
            self.weights = np.append(self.weights, scalar * w.weights)
            self.points = np.append(self.points, w.points)
        self.check_status_of_weights_and_points()

    def compute_scalar_product(self, w):
        """Computes the scalar product of the element x of C with another element of C, that is, <x, w>.

        Args:
            w: instance of ElementMarginalPolytope
                An instance of ElementMarginalPolytope.

        Returns:
            The scalar product <x, w>.
        """

        w_weights = fd(w.weights)
        w_points = fd(w.points)
        x_weights = fd(self.weights)
        x_points = fd(self.points)

        weights_matrix = x_weights.dot(w_weights.T)
        points_matrix = kernel_evaluation(x_points, w_points.T)

        multiplied_matrix = np.multiply(weights_matrix, points_matrix)
        scalar_product = float(np.sum(multiplied_matrix))
        return scalar_product

    def compute_scalar_product_with_mu(self, mu):
        """Computes the scalar product with mu = sum_j=1 to len(mu[0]) a_j cos(2pi j x) + b_j sin(2pi j x).

        Specifically, returns:
            (sum_{i = 1}^len(weights)w_i sum_{j=1}^len(mu[0]) a_j cos( 2 pi j x_i) + b_j sin (2 pi j x_i)),
        where w_i, x_i, a_j, b_j are the weights, points, and a-entries and b-entries of mu, respectively.
        """

        mu_a = fd(mu[0])
        mu_b = fd(mu[1])
        assert mu_a.shape[0] == mu_b.shape[0], "The entries for cos and sin for mu have to have the same length."
        x_weights = fd(self.weights)
        x_points = fd(self.points)

        vec_a = fd(np.arange(1, mu_a.shape[0] + 1))
        mat_a = vec_a.dot(x_points.T)
        cos_mat = np.cos(2 * np.pi * mat_a)
        a_cos_mat = mu_a.T.dot(cos_mat)
        w_cos_mat = float(a_cos_mat.dot(x_weights))

        vec_b = fd(np.arange(1, mu_b.shape[0] + 1))
        mat_b = vec_b.dot(x_points.T)
        sin_mat = np.sin(2 * np.pi * mat_b)
        b_sin_mat = mu_b.T.dot(sin_mat)
        w_sin_mat = float(b_sin_mat.dot(x_weights))

        return w_cos_mat + w_sin_mat

    def __copy__(self):
        """Creates a copy of the instance."""
        return type(self)(self.weights, self.points)

    def print_weights_points(self):
        """Prints the weights and points."""
        print("Weights: ", self.weights)
        print("Points: ", self.points)

    def check_status_of_weights_and_points(self):
        """Checks whether the weights and points are in the correct format."""
        self.check_shape_of_weights_and_points()
        self.check_sum_weights()
        self.check_points_are_in_feasibility_region()

    def check_shape_of_weights_and_points(self):
        assert self.weights.shape == self.points.shape, "Every data point requires an associated weight."

    def check_sum_weights(self):
        assert abs(sum(self.weights) - 1) <= 10e-10, "The sum of the weights has to be 1."

    def check_points_are_in_feasibility_region(self):
        assert (0 <= self.points).all() & (self.points <= 1).all(), "Points have to be in [0, 1]."

    # def copy(self):
    #     return ElementMarginalPolytope(weights=self.weights, points=self.points)


def bp(y: float, degree: int = 2):
    """Evaluates the Bernoulli polynomial of given degree at y.

    Args:
        y: float
            Point of evaluation.
        degree: integer, Optional
            The degree of the Bernoulli polynomial. (Default is 2).

    Returns:
        The evaluation of the Bernoulli polynomial of given degree at y.
    """
    if degree == 2:
        return (y ** 2 - y + 1 / 6)


def bp_fractional(y: float, w: float, degree: int = 2):
    """Evaluates the Bernoulli polynomial of given degree at the fractional part of y - w.

    Args:
        y: float
            The function evaluates the Bernoulli polynomial at the fractional part of y - w.
        w: float
            The function evaluates the Bernoulli polynomial at the fractional part of y - w.
        degree: integer, Optional
            The degree of the Bernoulli polynomial. (Default is 2).

    Returns:
        The evaluation of the Bernoulli polynomial of given degree at the fractional part of y - w.
    """
    fractional = y - w - np.floor(y - w)
    return bp(fractional, degree)


def kernel_evaluation(y: float, w: float, m: int = 1):
    """Evaluates the kernel k(y, w) = -(1)^(m-1)/((2 m)!) Bernoulli_2m(y - w - floor(y - w)).

    Args:
        y: float
            The function evaluates k(y, w).
        w: float
            The function evaluates k(y, w).
        m: integer
            Determines the degree of the Bernoulli polynomial. Depends on the Hilbert space. (Default is 1).

    Returns:
        The evaluation of the kernel k(y, w) = -(1)^(m-1)/((2 m)!) Bernoulli_2m(y - w - floor(y - w)).
    """
    if m == 1:
        return 1 / 2 * bp_fractional(y, w, degree=2)
    else:
        return (-1) ** (m - 1) / math.factorial(2 * m) * bp_fractional(y, w, degree=2 * m)

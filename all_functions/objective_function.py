from all_functions.auxiliary_functions import fd

from all_functions.hilbert_space_element import ElementMarginalPolytope
import numpy as np


class SquaredLoss:
    """Represents the loss function 1/2||x - mu||^2,.

    Attributes:
        mu: list, Optional
            The attribute mu represents a function of the form
                sum_j=1 to len(mu[0]) a_j cos(2pi j y) + b_j sin(2pi j y).
            If not specificed, i.e., if mu is None, we use the uniform distribution. If mu is not None, it has to be of
            the format mu = [np.ndarray, np.ndarray], where the two arrays are one-dimensional and of identical length.
            The first one corresponds to the cosine entries (a_j's) and the second one to the sine entries (b_j's).
            (Default is None.)

    Methods:
        evaluate_loss(x: ElementMarginalPolytope)
            Evaluates the loss at x.
        evaluate_gradient(x: ElementMarginalPolytope, w: ElementMarginalPolytope)
            Evaluates the gradient of x at w.
        compute_step_size(iteration: int, x: ElementMarginalPolytope, w: ElementMarginalPolytope,
                          step: dict, max_step: float)
            Computes the step-size given x in a certain direction.
    """

    def __init__(self, mu: list = None):
        self.mu = mu
        self.mu_squared = 0

        # needs to bring mu into the correct form and precompute various variables
        if self.mu is not None:
            self.mu_a = self.mu[0].flatten()
            self.mu_b = self.mu[1].flatten()
            max_length = max(len(self.mu_a), len(self.mu_b))
            if len(self.mu_a) < max_length:
                tmp = np.zeros(max_length).flatten()
                tmp[:len(self.mu_a)] = self.mu_a
                self.mu_a = tmp
            if len(self.mu_b) < max_length:
                tmp = np.zeros(max_length).flatten()
                tmp[:len(self.mu_b)] = self.mu_b
                self.mu_b = tmp
            self.mu[0] = self.mu_a
            self.mu[1] = self.mu_b
            for i in range(0, max_length):
                self.mu_squared += (2 * np.pi * (i + 1)) ** 2 * (self.mu_a[i] ** 2 + self.mu_b[i] ** 2)
            self.mu_squared *= 1 / 2

        self.mean_element_vals = None

    def evaluate_loss(self, x: ElementMarginalPolytope):
        """Evaluates the loss at x."""
        loss = x.compute_scalar_product(x)
        if self.mu is not None:
            loss += self.mu_squared - 2 * x.compute_scalar_product_with_mu(self.mu)
        return float(1 / 2 * loss)

    def evaluate_gradient(self, x: ElementMarginalPolytope, w: ElementMarginalPolytope):
        """Evaluates the gradient of x at w."""
        gradient = x.compute_scalar_product(w)
        if self.mu is not None:
            gradient -= w.compute_scalar_product_with_mu(self.mu)
        return float(gradient)

    def compute_step_size(self,
                          iteration: int,
                          x: ElementMarginalPolytope,
                          w: ElementMarginalPolytope,
                          step: dict,
                          max_step: float = 1):
        """Computes the step-size given x in a certain direction.

        Args:
            iteration: integer
                The current iteration of the algorithm. Needed for "open-loop".
            x: instance of ElementMarginalPolytope
                An instance of ElementMarginalPolytope.
            w: instance of ElementMarginalPolytope
                An instance of ElementMarginalPolytope.
            step: dict
                A dictionnary containing the information about the step type. The dictionary can have the following arg-
                uments:
                    "step type": Choose from "open-loop", "exact", "line-search", "short-step".
                Additional Arguments:
                    For "open-loop", provide float values for the keys "a", "b", "c", "d" that affect the step type
                    as follows: a / (b * iteration**d + c)
                    For "line-search", provide an integer for the number of exhaustive search steps for the key
                    "number of iterations for line-search".
            max_step: float, Optional
                Maximum step-size. (Default is 1.)

        Returns:
            optimal_distance: float
                The step-size computed according to the chosen method.
        """
        step_type = step["step type"]
        if step_type == "open-loop":
            a = step["a"]
            b = step["b"]
            c = step["c"]
            d = step["d"]
            optimal_distance = a / (b * iteration ** d + c)
        elif step_type in ["exact", "line-search"]:

            sp_x_x = x.compute_scalar_product(x)
            sp_x_w = x.compute_scalar_product(w)
            sp_w_w = w.compute_scalar_product(w)
            sp_x_mu = 0
            sp_w_mu = 0
            if self.mu is not None:
                sp_x_mu = x.compute_scalar_product_with_mu(self.mu)
                sp_w_mu = w.compute_scalar_product_with_mu(self.mu)

            numerator = sp_x_x - sp_x_w - sp_x_mu + sp_w_mu
            denominator = sp_x_x - 2 * sp_x_w + sp_w_w

            optimal_distance = numerator / denominator
        elif step_type == "short-step":
            sp_x_x = x.compute_scalar_product(x)
            sp_x_w = x.compute_scalar_product(w)
            sp_w_w = w.compute_scalar_product(w)
            sp_x_mu = 0
            sp_w_mu = 0
            if self.mu is not None:
                sp_x_mu = x.compute_scalar_product_with_mu(self.mu)
                sp_w_mu = w.compute_scalar_product_with_mu(self.mu)

            numerator = sp_x_x - sp_x_w - sp_x_mu + sp_w_mu
            denominator = sp_x_x - 2 * sp_x_w + sp_w_w

            L = 1
            optimal_distance = numerator / (L * denominator)

        if optimal_distance > max_step:
            optimal_distance = max_step
        if optimal_distance <= 0:
            optimal_distance = 0

        return optimal_distance


def mu_from_rho(rho: list):
    """Computes mu from rho.

    More specifically, given a distribution rho such that int_Y rho dy = 1, creates the associated
        mu(t) = int_Y Phi(y)(t)rho(y) dy
    and stores it in appropriate format, where Phi(y)(t) is the kernel described in kernel_evaluation().

    Args:
        rho: list
            The argument rho needs to be of the form [rho_a, rho_b], where rho_a and rho_b are np.ndarrays of
            dimension 1. The ith entry of rho_a represents alpha_i and the ith entry of rho_b represents beta_i in the
            representation rho = (sum_{i=1}^d alpha_i cos(2pi i y) + beta_i sin(2pi i y))**2, where
            d = max(len(rho_a), len(rho_b)).
    Returns:
        mu: list
            Returns mu = [mu_a, mu_b], where mu_a and mu_b are np.ndarrays of dimension 1. The ith entry of mu_a
            represents a_i and the ith entry of mu_b represents b_i in the representation
            mu = sum_{i=1}^k a_i cos(2pi i y) + b_i sin(2pi i y), where k depends on rho.
    """
    rho_a = rho[0]
    rho_b = rho[1]
    sum = 1 / 2 * (np.sum(rho_a ** 2) + np.sum(rho_b ** 2))
    assert abs(sum - 1) <= 10e-10, "Rho is not a distribution."
    l_rho_a = len(rho_a)
    l_rho_b = len(rho_b)
    max_length = int(2 * max(l_rho_a, l_rho_b))

    mu_a = np.zeros(int(max_length))
    mu_b = np.zeros(len(mu_a))

    for i in range(l_rho_a):
        # cos(2pi i x) cos(2pi j x)
        for j in range(l_rho_a):
            sum_i_j_in_mu = int(i + j + 1)
            mu_a[sum_i_j_in_mu] += 1 / 4 * rho_a[i] * rho_a[j] * 2 / ((2 * np.pi * (sum_i_j_in_mu + 1)) ** 2)
            diff_i_j_in_mu = int(abs(i - j) - 1)
            if diff_i_j_in_mu >= 0:
                mu_a[diff_i_j_in_mu] += 1 / 4 * rho_a[i] * rho_a[j] * 2 / ((2 * np.pi * (diff_i_j_in_mu + 1)) ** 2)
        # cos(2pi i x) sin(2pi k x) (These terms will pop up twice. We deal with both of them at once here.)
        for k in range(l_rho_b):
            sum_i_k_in_mu = int(i + k + 1)
            mu_b[sum_i_k_in_mu] -= 2 / 4 * rho_a[i] * rho_b[k] * 2 / ((2 * np.pi * (sum_i_k_in_mu + 1)) ** 2)
            diff_i_k_in_mu = int(abs(i - k) - 1)
            if diff_i_k_in_mu >= 0:
                if i > k:
                    mu_b[diff_i_k_in_mu] += 2 / 4 * rho_a[i] * rho_b[k] * 2 / ((2 * np.pi * (diff_i_k_in_mu + 1)) ** 2)
                elif i < k:
                    mu_b[diff_i_k_in_mu] -= 2 / 4 * rho_a[i] * rho_b[k] * 2 / ((2 * np.pi * (diff_i_k_in_mu + 1)) ** 2)
    # sin(2pi i x) sin(2pi j x)
    for i in range(l_rho_b):
        for j in range(l_rho_b):
            sum_i_j_in_mu = int(i + j + 1)
            mu_a[sum_i_j_in_mu] -= 1 / 4 * rho_b[i] * rho_b[j] * 2 / ((2 * np.pi * (sum_i_j_in_mu + 1)) ** 2)
            diff_i_j_in_mu = int(abs(i - j) - 1)
            if diff_i_j_in_mu >= 0:
                mu_a[diff_i_j_in_mu] += 1 / 4 * rho_b[i] * rho_b[j] * 2 / ((2 * np.pi * (diff_i_j_in_mu + 1)) ** 2)

    mu = [mu_a, mu_b]
    return mu


def confirm_rho_distribution(rho: list):
    """Check that rho is a distribution and if not, scale it such that it is a distribution.

    Checks whether int_Y rho(y) dy = 1 and if not, scales rho, such that this holds.

    Args:
        rho: list
            The argument rho needs to be of the form [rho_a, rho_b], where rho_a and rho_b are np.ndarrays of
            dimension 1. The ith entry of rho_a represents alpha_i and the ith entry of rho_b represents beta_i in the
            representation rho = (sum_{i=1}^d alpha_i cos(2pi i y) + beta_i sin(2pi i y))**2, where
            d = max(len(rho_a), len(rho_b)).

    Returns:
        rho_as_distribution: list
            The returned variable rho_as_distribution is a distribution of the form [rho_a, rho_b], where rho_a and
            rho_b are np.ndarrays of dimensions 1. The ith entry of rho_a represents a_i and the ith entry of rho_b
            represents b_i in the representation rho_as_distribution = (sum_i a_i cos(2pi i x) + b_i sin(2pi i x))**2.
    """
    rho_a = rho[0].flatten()
    rho_b = rho[1].flatten()

    sum = 1 / 2 * (np.sum(rho_a ** 2) + np.sum(rho_b ** 2))
    rho_a = rho_a / np.sqrt(sum)
    rho_b = rho_b / np.sqrt(sum)
    sum = 1 / 2 * (np.sum(rho_a ** 2) + np.sum(rho_b ** 2))
    assert abs(sum - 1) <= 10e-10, "Rho is still not a distribution, there is a problem."
    rho_as_distribution = [rho_a, rho_b]

    return rho_as_distribution


class SquaredLossFinDim:
    """Represents the loss function f(x) = 1/2||Ax - b||^2 + 1/2 lmbda ||x||^2.

    Attributes:
        A: np.ndarray
            A np.ndarray of dimension (m, n).
        b: np.ndarray
            A np.ndarray of dimension (m, 1).
        lmbda: float, Optional
            Regularization parameter. (Default is 0.0.)

    Methods:
        evaluate_loss(x: np.ndarray)
            Evaluates the loss of f at x.
        evaluate_gradient(x: np.ndarray)
            Evaluates the gradient of f at x.
        compute_step_size(iteration: int, x: np.ndarray, direction: np.ndarray, step: dict, max_step: float = 1)
            Computes the step-size for iterate x in a certain direction.
    """

    def __init__(self, A: np.ndarray, b: np.ndarray, lmbda: float = 0.0):

        self.A = A
        self.Asquared = self.A.T.dot(self.A)
        self.b = b.flatten()
        self.Ab = self.A.T.dot(self.b[:, np.newaxis]).flatten()
        self.m, self.n = self.A.shape
        eigenvalues, _ = np.linalg.eigh(self.Asquared)
        self.L = float(np.max([np.max(eigenvalues), 1]))

        assert self.b.shape[0] == self.m, "Arrays not of correct dimensions."
        self.lmbda = lmbda

    def evaluate_loss(self, x: np.ndarray):
        """Evaluates the loss at x."""
        x = x.flatten()
        return float(
            1 / 2 * (float(self.lmbda) * np.linalg.norm(x) ** 2 + np.linalg.norm(
                self.A.dot(x[:, np.newaxis]).flatten() - self.b) ** 2))

    def evaluate_gradient(self, x: np.ndarray):
        """Evaluates the gradient of f at x."""
        x = x.flatten()
        gradient = self.Asquared.dot(x[:, np.newaxis]).flatten() - self.Ab + self.lmbda * x
        return gradient

    def compute_step_size(self,
                          iteration: int,
                          x: np.ndarray,
                          direction: np.ndarray,
                          gradient: np.array,
                          step: dict,
                          max_step: float = 1):
        """Computes the step-size for iterate x in a certain direction.

        Args:
            iteration: integer
                The current iteration of the algorithm. Needed for "open-loop".
            x: np.ndarray
            direction: np.ndarray
                FW vertex.
            gradient: np.ndarray
            step: dict
                A dictionnary containing the information about the step type. The dictionary can have the following arg-
                uments:
                    "step type": Choose from "open-loop", "open-loop constant", "line-search", "line-search afw",
                    "short-step afw", "line-search difw probability simplex", "short-step", and
                    "short-step difw probability simplex".
                Additional Arguments:
                    For "open-loop", provide float values for the keys "a", "b", "c", "d" that affect the step type
                    as follows: a / (b * iteration**d + c)
            max_step: float, Optional
                Maximum step-size. (Default is 1.)

        Returns:
            optimal_distance: float
                The step-size computed according to the chosen method.
        """
        x = x.flatten()
        direction = direction.flatten()
        gradient = gradient.flatten()
        step_type = step["step type"]
        if step_type == "open-loop":
            a = step["a"]
            b = step["b"]
            c = step["c"]
            d = step["d"]
            optimal_distance = a / (b * iteration ** d + c)
        elif step_type == "open-loop constant":
            optimal_distance = step["cst"]

        elif step_type == "line-search":
            p_x = direction - x
            optimal_distance = float(-gradient.T.dot(p_x)) / (p_x.T.dot(self.Asquared).dot(p_x))

        elif step_type == "short-step":
            optimal_distance = gradient.dot(x - direction) / (self.L * np.linalg.norm(x - direction) ** 2)

        elif step_type == "line-search afw":
            optimal_distance = float(-gradient.T.dot(direction)) / (direction.T.dot(self.Asquared).dot(direction))

        elif step_type == "short-step afw":
            optimal_distance = float(-gradient.T.dot(direction)) / (self.L * np.linalg.norm(direction) ** 2)

        elif step_type == "line-search difw probability simplex":
            y_mod = direction.copy()[(direction < 0)]
            x_mod = x.copy()[(direction < 0)]
            y_mod[(y_mod == 0)] = 1
            max_step = min(max_step, np.max(-x_mod / y_mod))
            optimal_distance = float(-gradient.T.dot(direction)) / (direction.T.dot(self.Asquared).dot(direction))

        elif step_type == "short-step difw probability simplex":
            y_mod = direction.copy()[(direction < 0)]
            x_mod = x.copy()[(direction < 0)]
            y_mod[(y_mod == 0)] = 1

            try:
                max_step = min(max_step, np.max(-x_mod / y_mod))
            except ValueError:  # raised if `y` is empty.
                pass
            optimal_distance = float(-gradient.T.dot(direction)) / (self.L * np.linalg.norm(direction) ** 2)

        if optimal_distance > max_step:
            optimal_distance = max_step
        return float(optimal_distance)


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':

    x = np.array([[1, 0],
                 [np.log(2), np.log(3)]])
    print(sigmoid(x))


class LogisticLossFinDim:
    """Represents the logistic loss function f(x) = (1/m)sum_i log(1 + exp(-b_i * a_i^T * x)).

    Attributes:
        A: np.ndarray
            A np.ndarray of dimension (m, n).
        b: np.ndarray
            A np.ndarray of dimension (m, 1).
        m: int
            Number of data points.
        n: int
            Number of features.

    Methods:
        evaluate_loss(x: np.ndarray)
            Evaluates the loss of f at x.
        evaluate_gradient(x: np.ndarray)
            Evaluates the gradient of f at x.
        compute_step_size(iteration: int, x: np.ndarray, direction: np.ndarray, step: dict, max_step: float = 1)
            Computes the step-size for iterate x in a certain direction.
    """

    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = A
        self.b = b.flatten()
        self.m, self.n = A.shape

    def evaluate_loss(self, x: np.ndarray):
        """Evaluates the loss at x."""
        x = x.flatten()
        logits = self.A.dot(x) * self.b
        return float(np.mean(np.logaddexp(0, -logits)))

    def evaluate_gradient(self, x: np.ndarray):
        """Evaluates the gradient of f at x."""
        x = x.flatten()
        logits = self.A.dot(x) * self.b
        probs = sigmoid(-logits)
        grad = self.A.T.dot(-self.b * probs) / self.m
        return grad

    def compute_step_size(self,
                          iteration: int,
                          x: np.ndarray,
                          direction: np.ndarray,
                          gradient: np.array,
                          step: dict,
                          max_step: float = 1):
        """Computes the step-size for iterate x in a certain direction.

        Args:
            iteration: integer
                The current iteration of the algorithm. Needed for "open-loop".
            x: np.ndarray
            direction: np.ndarray
                FW vertex.
            gradient: np.ndarray
            step: dict
                A dictionnary containing the information about the step type. The dictionary can have the following arg-
                uments:
                    "step type": Choose from "open-loop", "open-loop constant"
                Additional Arguments:
                    For "open-loop", provide float values for the keys "a", "b", "c", "d" that affect the step type
                    as follows: a / (b * iteration**d + c)
            max_step: float, Optional
                Maximum step-size. (Default is 1.)

        Returns:
            optimal_distance: float
                The step-size computed according to the chosen method.
        """
        step_type = step["step type"]
        if step_type == "open-loop":
            a = step["a"]
            b = step["b"]
            c = step["c"]
            d = step["d"]
            optimal_distance = a / (b * iteration ** d + c)
        elif step_type == "open-loop constant":
            optimal_distance = step["cst"]

        if optimal_distance > max_step:
            optimal_distance = max_step

        return float(optimal_distance)

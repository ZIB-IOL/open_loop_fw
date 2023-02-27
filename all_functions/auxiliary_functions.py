import autograd.numpy as np


def fd(matrix, option="column"):
    """Guarantees that autograd.numpy aarray is 2d.

    Args:
        matrix: np.ndarray
            An array.
        option: string, Optional
            Either "column" or "row", determining whether we get a row or column vector. (Default is column.)

    Returns:
        A 2d version of matrix as a column or row vector.
    """
    if matrix.ndim == 1:
        if option == "column":
            matrix = matrix[:, np.newaxis]
        elif option == "row":
            matrix = matrix[np.newaxis, :]
    return matrix


def distribution_to_string(distribution):
    """Creates a string from a list of arrays."""
    string = ""
    for j in range(len(distribution)):
        if j > 0:
            string = string + "_"
        array = distribution[j].flatten()
        for i in range(len(array)):
            string = string + str(round(array[i], 2))
    return string


def get_non_zero_indices(x: np.ndarray):
    return np.where(x.flatten())[0].tolist()
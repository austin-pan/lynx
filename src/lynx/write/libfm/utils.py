import numpy as np
from scipy import sparse


def hack_sparse(mat: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Hack to get single-column sparse matrices to work with libFM. This is
    achieved by adding a new explicit column of 0s to the sparse matrix. Luckily
    only a single explicit 0 is required to achieve this.

    Args:
        mat (sparse.csr_matrix): Sparse matrix to add column to.

    Returns:
        sparse.csr_matrix: Provided matrix with an extra column of zeros which
        can be used for libFM.
    """
    row = np.array([0])
    col = np.array([0])
    data = np.array([1])
    extra_column = sparse.csr_matrix((data, (row, col)), shape=(mat.shape[0], 1))

    hacked_mat: sparse.csr_matrix = sparse.hstack([mat, extra_column], format="csr") # type: ignore
    hacked_mat[(0, -1)] = 0
    return hacked_mat

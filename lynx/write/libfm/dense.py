"""Writer to Dense Structure libFM files."""


from typing import Iterable
from sklearn import datasets

import lynx as lx
from lynx.write.libfm import utils


def write_libfm(
    table: lx.Table,
    target: Iterable[float],
    path: str,
    *,
    empty: bool = False
) -> None:
    """
    Write provided Table with the provided target values in materialized form to
    a single SVMLight file.

    Args:
        table (Table): Table to write.
        target (Iterable[float]): Target values to use.
        path (str): Output filepath.
        empty (bool, optional): Whether to write an empty file. Defaults to
        None.
    """
    if not empty:
        data = table.to_csr_matrix()
        if data.shape[1] == 1:
            print(
                "Warning: libFM may not work with a matrix of width 1. Adding " +
                "a column of zeros to make libFM happy."
            )
            data = utils.hack_sparse(data)

        datasets.dump_svmlight_file(X=data, y=target, f=path) # type: ignore
    else:
        with open(path, "w", encoding="utf-8"):
            pass

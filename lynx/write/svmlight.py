"""Writer for SVMLight sparse file type."""


from typing import Iterable
from sklearn import datasets

import lynx as lx


def write_svmlight(
    table: lx.Table,
    target: Iterable[float],
    path: str,
) -> None:
    """
    Write provided Table with the provided target values in materialized form to
    a single SVMLight file.

    Args:
        table (Table): Table to write.
        target (Iterable[float]): Target values to use.
        path (str): Output filepath.
    """
    data = table.to_csr_matrix()
    datasets.dump_svmlight_file(X=data, y=target, f=path) # type: ignore

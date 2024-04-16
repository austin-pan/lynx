import os
from typing import Iterable
from sklearn import datasets
import numpy as np

import lynx as lx

def write_libfm(
    table: lx.Table,
    target: Iterable[float],
    dir_path: str,
    phase: str,
    *,
    ignore_block: bool = False,
    empty_indices: bool = False,
    empty_targets: bool = False
) -> None:
    """
    Write provided Table with the provided target values as blocks in LibFM
    block structure format where each block has its own SVMLight file and an
    accompanying join mapping file.

    Args:
        table (Table): Table to write.
        target (Iterable[float]): Target values to use.
        path (str): Output filepath.
    """
    for block in table.blocks:
        if not ignore_block:
            data = block.get_block_csr_matrix()
            if data.shape[1] == 0:
                print(f"Warning: Empty block {block.name}")
            if data.shape[1] == 1:
                print(f"Warning: LibFM may not work with a block ({block.name}) of width 1")

            datasets.dump_svmlight_file(
                X=data,
                y=np.zeros(data.shape[0]), # type: ignore
                f=os.path.join(dir_path, f"{block.name}.libfm")
            )

        index_path = os.path.join(dir_path, f"{block.name}.{phase}")
        if not empty_indices:
            np.savetxt(
                fname=index_path,
                X=block.index, # type: ignore
                fmt="%i"
            )
        else:
            with open(index_path, "w", encoding="utf-8"):
                pass

    target_path = os.path.join(dir_path, phase)
    if not empty_targets:
        np.savetxt(
            fname=target_path,
            X=target, # type: ignore
            fmt="%f"
        )
    else:
        with open(target_path, "w", encoding="utf-8"):
            pass

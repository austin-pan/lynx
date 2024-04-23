"""Block that handles sparse data."""

from typing import Tuple, Union
import pandas as pd
from scipy import sparse

from lynx.table.block import Block


class SparseBlock(Block):
    """
    Block that uses a CSR sparse matrix to represent data along with a
    positional index.
    """

    def __init__(
        self,
        name: str,
        data: sparse.csr_matrix,
        index: Union[pd.Series, None] = None
    ):
        """
        Args:
            name (str): Name of this SparseBlock.
            data (sparse.csr_matrix): Internal sparse data.
            index (pd.Series | None, optional): Positional index for data. If
            None, initializes to list of 1 to N where N is the height of the
            provided data. Defaults to None.
        """
        if index is None:
            index = pd.Series(range(data.shape[0]))
        super().__init__(name, index)
        self.data = data

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.index), self.data.shape[1])

    @property
    def data_shape(self) -> Tuple[int, int]:
        return self.data.shape

    def to_dataframe(self) -> pd.DataFrame:
        dense = pd.DataFrame(self.data.todense())
        return dense.take(self.index) # type: ignore

    def to_csr_matrix(self) -> sparse.csr_matrix:
        return self.data[self.index]

    def get_block_csr_matrix(self) -> sparse.csr_matrix:
        return self.data

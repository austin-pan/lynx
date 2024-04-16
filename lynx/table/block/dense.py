"""Block that handles densed, named data."""

from copy import deepcopy
from typing import List, Tuple, Union
import pandas as pd
from scipy import sparse

from lynx.table.block import Block
from lynx.table.utils import dataframe_to_csr_matrix


class DenseBlock(Block):
    """
    Main block in a Table that enables merging and querying of columns. The
    underlying data is a pandas DataFrame.
    """

    def __init__(
        self,
        name: str,
        data: pd.DataFrame,
        index: Union[pd.Series, None] = None
    ):
        """
        Args:
            name (str): Name of this DenseBlock.
            data (pd.DataFrame): Internal data of this DenseBlock.
            index (pd.Series | None, optional): Positional index for data. If
            None, initializes to list of 1 to N where N is the height of the
            provided data. Defaults to None.
        """
        self.data = data.reset_index(drop=True)
        if index is None:
            index = self.data.index.to_series()
        super().__init__(name, index)

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.index), self.data.shape[1])

    @property
    def data_shape(self) -> Tuple[int, int]:
        return self.data.shape

    @property
    def columns(self) -> List[str]:
        """Returns the column names in this Block's data."""
        return self.data.columns.to_list()

    def to_dataframe(self) -> pd.DataFrame:
        return self.data.take(self.index) # type: ignore

    def to_csr_matrix(self) -> sparse.csr_matrix:
        return dataframe_to_csr_matrix(self.to_dataframe())

    def get_block_csr_matrix(self) -> sparse.csr_matrix:
        return dataframe_to_csr_matrix(self.data)

    def get(self, column: str) -> pd.Series:
        """Returns the requested column values of the materialized data."""
        return self.data[column].take(self.index).reset_index(drop=True) # type: ignore

    def pop(self, column: str) -> pd.Series:
        """
        Returns the requested column values of the materialized data and drops
        the column.
        """
        return self.data.pop(column).take(self.index).reset_index(drop=True) # type: ignore

    def drop(self, column: Union[str, List[str]]) -> Block:
        """Returns a new Block with the provided column removed."""
        copy = deepcopy(self)
        copy.data = self.data.drop(columns=column, errors="ignore")
        return copy

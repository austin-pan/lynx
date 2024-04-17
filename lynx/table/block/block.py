"""Block data structure."""

from copy import deepcopy
from typing import Sequence, Tuple, Union
import pandas as pd
from scipy import sparse


class Block:
    """
    Base blocks that make up Tables. Every Block has a name and index for
    materializing the block's data when materializing the Table.

    Blocks are lazy representations of data where the internal data is not the
    entire block's data. The materialized Block is the internal data
    positionally indexed by the index.
    """

    def __init__(self, name: str, index: pd.Series):
        """
        Args:
            name (str): Name of Block.
            index (pd.Series): Index of Block.
        """
        self.name = name
        self.index = index

    @property
    def shape(self) -> Tuple[int, int]:
        """Height and width of this Block after materialization."""
        raise NotImplementedError()

    @property
    def data_shape(self) -> Tuple[int, int]:
        """Height and width of the internal data of this Block."""
        raise NotImplementedError()

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the DataFrame representation of this materialized Block."""
        raise NotImplementedError()

    def to_csr_matrix(self) -> sparse.csr_matrix:
        """Returns the sparse matrix representation of this materialized Block."""
        raise NotImplementedError()

    def get_block_csr_matrix(self) -> sparse.csr_matrix:
        """Returns the sparse matrix representation of this Block's internal data."""
        raise NotImplementedError()

    def get_index_name(self, index_prefix: str) -> str:
        """Returns the name to use to track this block's index."""
        return f"{index_prefix}_{self.name}"

    def take(self, indices: Union[int, Sequence[int]]) -> "Block":
        """Returns a new Block with the requested positionally indexed rows."""
        if isinstance(indices, int):
            indices = [indices]
        block = deepcopy(self)
        block.index = self.index.take(indices)
        return block

    def reindex(self, index: pd.Series) -> "Block":
        """Returns a new Block with the index set to the provided index."""
        block = deepcopy(self)
        block.index = index
        return block

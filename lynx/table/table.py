"""Table class and utilities."""

from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from scipy import sparse

from lynx.table import block as B
from lynx.table import utils as U


class Table:
    """
    A string of blocks of data where each block is an independent representation
    of tabular data.
    """

    def __init__(
        self,
        data: Union[List[B.Block], pd.DataFrame, sparse.csr_matrix],
        block_name: Union[str, None] = None):
        """
        Args:
            data (list[B.Block] | pd.DataFrame | sparse.csr_matrix): Data to
            initialize this Table with. Can be Blocks that make up this Table or
            tabular structure to create the initial Block of this Table. If data
            is a tabular structure, then an associated name must be provided as
            well.
            name (str | None): If data is a DataFrame or Sparse Matrix, then
            this is the name to use for the initial Block. Defaults to None.
        """
        if isinstance(data, pd.DataFrame):
            assert block_name is not None, "name must be supplied with DataFrame."
            data = [B.DenseBlock(block_name, data)]
        if isinstance(data, sparse.csr_matrix):
            assert block_name is not None, "name must be supplied with Sparse Matrix."
            data = [B.SparseBlock(block_name, data)]
        blocks: list[B.Block] = data

        # Remove blocks with no columns
        blocks = [block for block in blocks if block.shape[1] > 0]
        assert len(blocks) == len(set(block.name for block in blocks)), (
            f"All blocks must have different names: {[block.name for block in blocks]}"
        )
        self.height = blocks[0].shape[0]
        assert len(blocks) > 0 and self.height > 0, "There must be at least one non-empty block"

        for block in blocks[1:]:
            assert block.shape[0] == self.height, (
                f"All blocks must have the same height: {[block.shape[0] for block in blocks]}"
            )

        self.blocks = blocks

    def __len__(self) -> int:
        return self.height

    def __getitem__(self, key: Union[int, List[int]]) -> Union[pd.Series, pd.DataFrame, "Table"]:
        """Necessary for `sklearn.datasets.train_test_split` to work."""
        return self.take(key)

    @property
    def width(self) -> int:
        """Sum of the widths of all of the blocks in this Table."""
        return sum(block.shape[1] for block in self.blocks)

    @property
    def shape(self) -> Tuple[int, int]:
        """Height and width of this table when all blocks are materialized."""
        return (self.height, self.width)

    @property
    def block_shapes(self) -> Dict[str, Tuple[int, int]]:
        """Mapping of block names to the shape of each block's internal data."""
        return { block.name: block.data_shape for block in self.blocks }

    @property
    def block_names(self) -> List[str]:
        """Block names."""
        return [block.name for block in self.blocks]

    def has_dense_block(self) -> bool:
        """Returns whether this Table has a DenseBlock."""
        return any(isinstance(block, B.DenseBlock) for block in self.blocks)

    def to_dataframe(self) -> pd.DataFrame:
        """Returns a DataFrame representation of this Table."""
        table = pd.concat(
            [block.to_dataframe().reset_index(drop=True) for block in self.blocks],
            axis=1
        )
        return table

    def to_csr_matrix(self) -> sparse.csr_matrix:
        """Returns a sparse matrix representation of this Table."""
        blocks = [block.to_csr_matrix() for block in self.blocks]
        return sparse.hstack(blocks, format="csr") # type: ignore

    def get_indexed_columns(
        self,
        columns: Union[str, List[str]],
        index_prefix: Union[str, None] = None
    ) -> pd.DataFrame:
        """
        Returns specified columns of this table, concatenated with the indices
        of each block.

        Args:
            columns (str | list[str]): Columns to return.
            index_prefix (str | None, optional): String prefix to add to each index
            column. Defaults to None.

        Returns:
            pd.DataFrame: Columns concatenated with block indices.
        """
        if index_prefix is None:
            index_prefix = ""

        column_value = self.get(columns)
        indices = self._get_indices(index_prefix)
        return pd.concat([indices, column_value], axis=1)

    def _get_indices(self, index_prefix: Union[str, None] = None) -> pd.DataFrame:
        """
        Returns the indices of each block in this Table.

        Args:
            index_prefix (str | None, optional): String to prefix index column
            names with. Defaults to None.

        Returns:
            pd.DataFrame: B.Block indices concatenated together.
        """
        if index_prefix is None:
            index_prefix = ""

        indices = [
            block.index
            .to_frame(name=block.get_index_name(index_prefix))
            .reset_index(drop=True)
            for block in self.blocks
        ]
        return pd.concat(indices, axis=1)

    def _get_block(self, column: str) -> B.DenseBlock:
        """
        Returns the B.DenseBlock in this Table that contains the specified column.

        Args:
            column (str): Column that block should contain.

        Raises:
            ValueError: Requested column doesn't exist in any blocks of this
            Table.

        Returns:
            B.DenseBlock: B.Block that contains the specified column.
        """
        assert self.has_dense_block(), "Table needs a dense block to look for a column"
        for block in self.blocks:
            if isinstance(block, B.DenseBlock) and column in block.data.columns:
                return block
        raise ValueError(f"{column} not found")

    def get(self, columns: Union[str, List[str]]) -> pd.DataFrame:
        """
        Returns column values of this Table.

        Args:
            columns (str | list[str]): Columns to get.

        Returns:
            pd.DataFrame: Concatenatation of column values.
        """
        if isinstance(columns, str):
            columns = [columns]

        return pd.concat([
            self._get_block(column).get(column)
            for column in columns
        ], axis=1)

    def pop(self, column: str) -> pd.Series:
        """
        Returns the values of the specified column and drops the column from
        this table.

        Note: If there is only one column in this table, the table can
        potentially be left empty which would make it unusable.

        Args:
            column (str): Column to pop.

        Returns:
            pd.Series: Popped column values.
        """
        block = self._get_block(column)
        values = block.pop(column)
        self.blocks = [block for block in self.blocks if block.shape[1] > 0]
        return values

    def take(self, indices: Union[int, List[int]], axis=None) -> "Table":
        """
        Returns the rows in the given positional indices as a new Table.

        Args:
            indices (list[int]): Positional indices of rows to return.
            axis (None): Ignored. Necessary to camouflage as Pandas DataFrame
            for `sklearn.model_selection.train_test_split`.

        Returns:
            Table: New Table with rows taken from this Table.
        """
        blocks = [block.take(indices) for block in self.blocks]
        return Table(blocks)

    def drop(self, columns: Union[str, List[str]]) -> "Table":
        """
        Returns a copy of this Table with the given columns removed.

        Args:
            columns (str | list[str]): Columns to drop.

        Returns:
            Table: New Table with columns dropped from this Table.
        """
        assert self.has_dense_block(), "Table needs a dense block to drop a column from"
        if isinstance(columns, str):
            columns = [columns]
        blocks = [
            block.drop(columns) if isinstance(block, B.DenseBlock) else block
            for block in self.blocks
        ]
        return Table(blocks)

    def concat(self, other: "Table") -> "Table":
        """
        Returns a new Table with this and the other Tables' blocks concatenated.

        Args:
            other (Table): Table to concatenate to this Table.

        Returns:
            Table: New Table with concatenated blocks.
        """
        return self.extend(other.blocks)

    def extend(self, blocks: List[B.Block]) -> "Table":
        """
        Returns a new Table with this Table's blocks concatenated with the
        provided blocks.

        Args:
            blocks (list[B.Block]): Blocks to extend.

        Returns:
            Table: New Table with concatenated blocks.
        """
        extended_blocks = self.blocks + blocks
        return Table(extended_blocks)

    def merge(
        self,
        other: "Table",
        left_on: Union[str, List[str]],
        right_on: Union[str, List[str]],
        *,
        drop_left_on: bool = False
    ) -> "Table":
        return merge(self, other, left_on, right_on, drop_left_on=drop_left_on)

    def onehot(
        self,
        columns: Union[str, List[str]],
        block_name: Union[str, None] = None,
        *,
        drop: bool = True
    ) -> "Table":
        return onehot(self, columns, block_name, drop=drop)

    def normalize(
        self,
        columns: Union[str, List[str]],
        block_name: Union[str, None] = None,
        *,
        drop: bool = True
    ) -> "Table":
        return normalize(self, columns, block_name, drop=drop)

    def explode(
        self,
        column: str,
        block_name: Union[str, None] = None,
        *,
        drop: bool = True
    ) -> "Table":
        return explode(self, column, block_name, drop=drop)

    def model_interactions(
        self,
        index_features: Union[str, List[str]],
        value_features: Union[str, List[str]],
        block_name: Union[str, None] = None,
    ) -> "Table":
        return model_interactions(
            self,
            index_features,
            value_features,
            block_name,
        )

def merge(
    left: Table,
    right: Table,
    left_on: Union[str, List[str]],
    right_on: Union[str, List[str]],
    *,
    drop_left_on: bool = False
) -> Table:
    """
    Merges two Tables with a database-style inner join.

    Args:
        left (Table): Left Table.
        right (Table): Right Table.
        left_on (str | list[str]): Columns in the left Table to merge on.
        right_on (str | list[str]): Columns in the right Table to merge on.
        drop_left_on (bool, optional): Whether to drop the merging columns in
        left Table. Otherwise drops the merging columns in the right Table.
        Defaults to False.

    Returns:
        Table: New Table of the merged Tables.
    """
    # TODO support non-inner joins
    # TODO handle overlapping column names
    assert left.has_dense_block() and right.has_dense_block(), (
        "Both tables must have a dense block to be merged"
    )
    left_index_prefix = "_left_merge_index_"
    right_index_prefix = "_right_merge_index_"
    merge_mapping = pd.merge(
        left=left.get_indexed_columns(left_on, left_index_prefix),
        right=right.get_indexed_columns(right_on, right_index_prefix),
        left_on=left_on,
        right_on=right_on,
        how="inner"
    )

    merged_blocks = []
    for block in left.blocks:
        block = block.reindex(
            merge_mapping[block.get_index_name(left_index_prefix)] # type: ignore
        )
        # Drop left key if specified
        if drop_left_on and isinstance(block, B.DenseBlock) and left_on in block.columns:
            block = block.drop(left_on)
        merged_blocks.append(block)
    for block in right.blocks:
        block = block.reindex(
            merge_mapping[block.get_index_name(right_index_prefix)] # type: ignore
        )
        # Drop right key if specified
        if not drop_left_on and isinstance(block, B.DenseBlock) and right_on in block.columns:
            block = block.drop(right_on)
        merged_blocks.append(block)
    return Table(merged_blocks)

def onehot(
    table: Table,
    columns: Union[str, List[str]],
    block_name: Union[str, None] = None,
    *,
    drop: bool = True
) -> Table:
    """
    Returns a new Table with an additional block that is the one-hot encoding
    of the provided columns.

    Args:
        table (Table): Table with columns to one-hot encode.
        columns (str | list[str]): Columns to one-hot encode.
        block_name (str | None, optional): Name to use for the new one-hot
        encoded block. If None, then joins the provided column names with an
        underscore and appends "_onehot". Defaults to None.
        drop (bool, optional): Whether to drop the provided columns. Defaults to
        True.

    Returns:
        Table: Table with one-hot encoded columns.
    """
    assert table.height > 0, "Table height must be greater than zero"

    if isinstance(columns, str):
        columns = [columns]
    if block_name is None:
        block_name = f"{'_'.join(columns)}_onehot"

    mapped_index, unique_df = U.get_unique_mapping(table.get(columns), columns)
    ohe = U.onehot_dataframe(unique_df[columns])

    if drop:
        table = table.drop(columns)
    return table.extend([B.SparseBlock(block_name, ohe, index=mapped_index)])

def normalize(
    table: Table,
    columns: Union[str, List[str]],
    block_name: Union[str, None] = None,
    *,
    drop: bool = True
) -> Table:
    """
    Returns a new Table with an additional block that is the database
    normalization of the provided columns.

    Args:
        table (Table): Table with columns to normalize.
        columns (str | list[str]): Columns to normalize.
        block_name (str | None, optional): Name to use for the new one-hot
        encoding block. If None, then joins the provided column names with an
        underscore and appends "_normalized". Defaults to None.
        drop (bool, optional): Whether to drop the provided columns. Defaults to
        True.

    Returns:
        Table: Table with normalized columns.
    """
    if isinstance(columns, str):
        columns = [columns]
    if block_name is None:
        block_name = f"{'_'.join(columns)}_normalized"

    mapped_index, unique_df = U.get_unique_mapping(table.get(columns), columns)
    normalized_block = B.DenseBlock(block_name, unique_df[columns], index=mapped_index)

    if drop:
        table = table.drop(columns)
    return table.extend([normalized_block])

def explode(
    table: Table,
    column: str,
    block_name: Union[str, None] = None,
    *,
    drop: bool = True
) -> Table:
    """
    Returns a new Table with an additional block that is the multi-hot encoding
    of the provided columns.

    Args:
        table (Table): Table with the column to explode.
        column (str): Column to explode.
        block_name (str | None, optional): Name to use for the new multi-hot
        encoded block. If None, then appends "_exploded" to the provided column
        name. Defaults to None.
        drop (bool, optional): Whether to drop the provided column. Defaults to
        True.

    Returns:
        Table: Table with the exploded column.
    """
    if block_name is None:
        block_name = f"{column}_exploded"

    mapped_index, unique_df = U.get_unique_mapping(table.get(column), column)
    multihot = U.multihot_series(unique_df[column])
    exploded_block = B.SparseBlock(block_name, multihot, index=mapped_index)

    if drop:
        table = table.drop(column)
    return table.extend([exploded_block])

def model_interactions(
    table: Table,
    index_features: Union[str, List[str]],
    value_features: Union[str, List[str]],
    block_name: Union[str, None] = None,
    use_onehot: bool = False,
) -> Table:
    """
    Returns a new Table with an additional block that is the interaction matrix
    between the provided index and value features.

    Args:
        table (Table): Table with index and value columns.
        index_features (str | list[str]): Index columns.
        value_features (str | list[str]): Value columns.
        block_name (str | None, optional): Name to use for the new interactions
        block. If None, then joins the provided column names with an
        underscore and appends "_interactions". Defaults to None.

    Returns:
        Table: Table with the feature interactions.
    """
    if isinstance(index_features, str):
        index_features = [index_features]
    if isinstance(value_features, str):
        value_features = [value_features]
    features = index_features + value_features

    if block_name is None:
        block_name = f"{'_'.join(features)}_interactions"

    mapped_index, _ = U.get_unique_mapping(table.get(index_features), index_features)

    interactions = U.sparse_pivot(
        table.get(features).drop_duplicates(),
        index_features,
        value_features
    )
    if not use_onehot:
        # Divide rows by non-zero row sums to get proportional votes
        row_sums = np.asarray(interactions.sum(axis=1)).squeeze()
        interactions.data = interactions.data / row_sums[interactions.nonzero()[0]]
    interactions_block = B.SparseBlock(block_name, interactions, index=mapped_index)

    return table.extend([interactions_block])

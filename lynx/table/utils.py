"""Utilities for data wrangling."""

from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder


def stringify_series(series: pd.Series) -> str:
    """
    Returns a hashable representation of a series by stringifying the series
    values.

    Args:
        series (pd.Series): Series to stringify.

    Returns:
        str: Stringified version of the values of the series.
    """
    return '_'.join(series.map(str))

def get_unique_mapping(
    dataframe: pd.DataFrame,
    columns: Union[str, List[str]],
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Returns the index mapping of the unique values of the provided dataframe to
    the original values. Also returns the dataframe with duplicates dropped with
    an id column used for de-duping.

    Args:
        dataframe (pd.DataFrame): DataFrame with values to de-dupe.
        columns (str | list[str]): Columns in `dataframe` to de-dupe.

    Returns:
        tuple[pd.Series, pd.DataFrame]: Index mapping of unique values and the
        unique values of `dataframe`.
    """
    if isinstance(columns, str):
        columns = [columns]
    hashed_column = "unique_id"

    dataframe[hashed_column] = dataframe[columns].apply(stringify_series, axis=1, result_type=None)
    unique_df = dataframe.drop_duplicates(subset=hashed_column)

    unique_hashes = unique_df[hashed_column]
    mapping_series = pd.Series(range(len(unique_hashes)), index=unique_hashes)
    mapped_index = dataframe[hashed_column].map(mapping_series)
    return (mapped_index, unique_df)

def onehot_dataframe(dataframe: pd.DataFrame) -> sparse.csr_matrix:
    """
    Returns the one-hot encoding of the provided dataframe in a sparse matrix.

    Args:
        dataframe (pd.DataFrame): DataFrame to one-hot encode.

    Returns:
        sparse.csr_matrix: Sparse representation of the one-hot encoding.
    """
    assert dataframe.shape[0] > 0, "DataFrame must be non-empty"

    ohe_encoder = OneHotEncoder()
    ohe: sparse.csr_matrix = ohe_encoder.fit_transform(dataframe).astype(int) # type: ignore
    return ohe

def multihot_series(series: pd.Series) -> sparse.csr_matrix:
    """
    Returns the multi-hot encoding of the provided series where the values of
    the series are lists.

    Args:
        series (pd.Series): Series to multi-hot encode.

    Returns:
        sparse.csr_matrix: Sparse representation of the multi-hot encoding.
    """
    assert len(series) > 0, "Series must be non-empty"

    mlb = MultiLabelBinarizer(sparse_output=True)
    multihot_features: sparse.csr_matrix = mlb.fit_transform(series) # type: ignore
    return multihot_features

def sparse_pivot(
    dataframe: pd.DataFrame,
    idx_vars: Union[str, List[str]],
    value_vars: Union[str, List[str]]
) -> sparse.csr_matrix:
    """
    Pivots the provided dataframe to the indexes (rows) and values (columns)
    provided and returns the pivoted data as a sparse matrix.

    Args:
        dataframe (pd.DataFrame): Dataframe to pivot.
        idx_vars (str | list[str]): Index variables to use as the rows in the
        pivot table.
        value_vars (str | list[str]): Value variables to use as the column pivot
        tables.

    Returns:
        sparse.csr_matrix: Sparse matrix representation of the pivot table.
    """
    # `group_info` returns a tuple of
    #   1. group number that each row belongs to
    #   2. all unique group numbers
    #   3. number of unique groups
    idx_rows, _, num_idx_groups = dataframe.groupby(idx_vars).grouper.group_info
    val_cols, _, num_vars_groups = dataframe.groupby(value_vars).grouper.group_info

    data_pivot = sparse.csr_matrix(
        (np.ones(len(dataframe)), (idx_rows, val_cols)),
        shape=(num_idx_groups, num_vars_groups)
    )
    return data_pivot

def dataframe_to_csr_matrix(data: pd.DataFrame) -> sparse.csr_matrix:
    """
    Returns a sparse matrix representation of the provided DataFrame.

    Args:
        data (pd.DataFrame): DataFrame to make sparse.

    Returns:
        sparse.csr_matrix: Sparse matrix representation of the DataFrame.
    """
    sparse_data = data.astype(pd.SparseDtype("float64", 0))
    return sparse_data.sparse.to_coo().tocsr()

def normalize_sparse_rows(mat: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Returns a copy of the provided sparse matrix where each row has been divided
    by the row sum.

    Args:
        mat (sparse.csr_matrix): Sparse matrix to normalize.

    Returns:
        sparse.csr_matrix: Normalized sparse matrix.
    """
    mat = mat.copy()
    # Divide rows by non-zero row sums to get proportional votes
    row_sums = np.asarray(mat.sum(axis=1)).squeeze()
    mat.data = mat.data / row_sums[mat.nonzero()[0]]
    return mat

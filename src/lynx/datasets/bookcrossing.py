

import os
from typing import List, Union

import pandas as pd


def download() -> None:
    """
    Please download the BookCrossing dataset from
    https://www.kaggle.com/datasets/syedjaferk/book-crossing-dataset.

    The file tree should look something like:

    ```
    .../dataset_path/
        - BX-Book-Ratings.csv
        - BX-Books.csv
        - BX-Users.csv
    ```
    """
    raise NotImplementedError(
        "Please download the dataset from " +
        "https://www.kaggle.com/datasets/syedjaferk/book-crossing-dataset"
    )

def load_books(
    dataset_path: str,
    usecols: Union[List[str], None] = None,
    nrows: Union[int, None] = None
) -> pd.DataFrame:
    """
    Load books CSV as a pandas DataFrame.

    Args:
        dataset_path (str): Directory with BX data.
        usecols (List[str] | None, optional): Which columns to use. Can choose
        from. Defaults to None.
        nrows (int | None, optional): Number of rows to load. Defaults to None.

    Returns:
        pd.DataFrame: Books dataframe.
    """
    dataset_path = os.path.expanduser(dataset_path)
    return pd.read_csv(
        os.path.join(dataset_path, "BX-Books.csv"),
        usecols=usecols,
        sep=";",
        encoding="cp1252",
        on_bad_lines='skip',
        escapechar='\\',
        nrows=nrows
    )

def load_users(
    dataset_path: str,
    usecols: Union[List[str], None] = None,
    nrows: Union[int, None] = None
) -> pd.DataFrame:
    """
    Load users CSV as a pandas DataFrame.

    Args:
        dataset_path (str): Directory with BX data.
        usecols (List[str] | None, optional): Which columns to use. Defaults to
        None.
        nrows (int | None, optional): Number of rows to load. Defaults to None.

    Returns:
        pd.DataFrame: Users dataframe.
    """
    dataset_path = os.path.expanduser(dataset_path)
    users = pd.read_csv(
        os.path.join(dataset_path, "BX-Users.csv"),
        usecols=usecols,
        sep=";",
        encoding="cp1252",
        nrows=nrows
    )
    users["location"] = users["location"].str.split(", ")
    return users

def load_ratings(
    dataset_path: str,
    usecols: Union[List[str], None] = None,
    nrows: Union[int, None] = None
) -> pd.DataFrame:
    """
    Load ratings CSV as a pandas DataFrame.

    Args:
        dataset_path (str): Directory with BX data.
        usecols (List[str] | None, optional): Which columns to use. Defaults to
        None.
        nrows (int | None, optional): Number of rows to load. Defaults to None.

    Returns:
        pd.DataFrame: Ratings dataframe.
    """
    dataset_path = os.path.expanduser(dataset_path)
    return pd.read_csv(
        os.path.join(dataset_path, "BX-Book-Ratings.csv"),
        usecols=usecols,
        sep=";",
        encoding="cp1252",
        nrows=nrows
    )

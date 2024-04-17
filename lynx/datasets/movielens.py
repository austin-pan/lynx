"""http://files.grouplens.org/datasets/movielens/ml-1m.zip"""

import os
import subprocess
from typing import List, Union
import pandas as pd

def download_1m(destination: str) -> None:
    """
    Download MovieLens-1M to the provided destination.

    Args:
        destination (str): Directory to download dataset to.
    """
    destination = os.path.expanduser(destination)
    os.makedirs(destination, exist_ok=True)
    archive_path = os.path.join(destination, 'ml-1m.zip')
    commands = [
        f"curl -o {archive_path} http://files.grouplens.org/datasets/movielens/ml-1m.zip",
        f"unzip {archive_path} -d {destination}"
    ]
    for command in commands:
        subprocess.call(command.split(" "))
    os.remove(archive_path)

def load_users(
    dataset_path: str,
    nrows: Union[int, None] = None,
    usecols: Union[List[str], None] = None
) -> pd.DataFrame:
    """
    Load user data as pandas DataFrame.

    Args:
        dataset_path (str): Directory with MovieLens data.
        nrows (int | None, optional): Number of rows to read. Defaults to
        None.
        usecols (List[str] | None, optional): Columns to use. Chosen from
        ["user_id", "gender", "age", "occupation", "zipcode"]. Defaults to None.

    Returns:
        pd.DataFrame: User data.
    """
    dataset_path = os.path.expanduser(dataset_path)
    return pd.read_csv(
        os.path.join(dataset_path, "users.dat"),
        names=["user_id", "gender", "age", "occupation", "zipcode"],
        usecols=usecols,
        sep="::",
        engine="python",
        encoding="latin-1",
        nrows=nrows
    )

def load_movies(
    dataset_path: str,
    nrows: Union[int, None] = None,
    usecols: Union[List[str], None] = None
) -> pd.DataFrame:
    """
    Load movie data as pandas DataFrame.

    Args:
        dataset_path (str): Directory with MovieLens data.
        nrows (int | None, optional): Number of rows to read. Defaults to
        None.
        usecols (List[str] | None, optional): Columns to use. Chosen from
        ["movie_id", "title", "genres"]. Defaults to None.

    Returns:
        pd.DataFrame: Movie data.
    """
    dataset_path = os.path.expanduser(dataset_path)
    return pd.read_csv(
        os.path.join(dataset_path, "movies.dat"),
        names=["movie_id", "title", "genres"],
        usecols=usecols,
        sep="::",
        engine="python",
        encoding="latin-1",
        nrows=nrows
    )

def load_ratings(
    dataset_path: str,
    nrows: Union[int, None] = None,
    usecols: Union[List[str], None] = None
) -> pd.DataFrame:
    """
    Load Rating data as pandas DataFrame.

    Args:
        dataset_path (str): Directory with MovieLens data.
        nrows (int | None, optional): Number of rows to read. Defaults to
        None.
        usecols (List[str] | None, optional): Columns to use. Chosen from
        ["user_id", "movie_id", "rating", "timestamp"]. Defaults to None.

    Returns:
        pd.DataFrame: Rating data.
    """
    dataset_path = os.path.expanduser(dataset_path)
    return pd.read_csv(
        os.path.join(dataset_path, "ratings.dat"),
        names=["user_id", "movie_id", "rating", "timestamp"],
        usecols=usecols,
        sep="::",
        engine="python",
        encoding="latin-1",
        nrows=nrows
    )

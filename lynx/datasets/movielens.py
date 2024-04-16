"""http://files.grouplens.org/datasets/movielens/ml-1m.zip"""

import os
import subprocess
import pandas as pd

def download(destination: str) -> None:
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
    nrows: int | None = None,
    usecols: list[str] | None = None
) -> pd.DataFrame:
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
    nrows: int | None = None,
    usecols: list[str] | None = None
) -> pd.DataFrame:
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
    nrows: int | None = None,
    usecols: list[str] | None = None
) -> pd.DataFrame:
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

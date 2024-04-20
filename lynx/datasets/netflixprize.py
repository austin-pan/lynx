"""https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data"""

import os
import subprocess
from typing import Union
import pandas as pd


def download(destination: str) -> None:
    """
    Download and extract Netflix Prize dataset from
    https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz.
    """
    destination = os.path.expanduser(destination)
    os.makedirs(destination, exist_ok=True)
    archive_path = os.path.join(destination, "nf_prize_dataset.tar.gz")
    commands = [
        f"curl -o {archive_path} -L https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz",
        f"tar -xvzf {os.path.join(destination, 'download')} -C {destination}",
        f"tar -xvzf {os.path.join(destination, 'download', 'training_set.tar')} -C {os.path.join(destination, 'download')}"
    ]
    for command in commands:
        subprocess.call(command.split(" "))
    os.remove(archive_path)
    os.remove(os.path.join(destination, "download", "training_set.tar"))

def load_data(dataset_path: str, nrows: Union[int, None] = None) -> pd.DataFrame:
    """
    Load Netflix Prize training data.

    Args:
        dataset_path (str): Path to dataset directory.
        nrows (int | None, optional): Number of rows to load. Defaults to
        None.

    Returns:
        pd.DataFrame: Netflix Prize training data.
    """
    dataset_path = os.path.expanduser(dataset_path)
    movies_path = os.path.join(dataset_path, "training_set")
    movie_files = [f for f in os.listdir(movies_path) if f.endswith(".txt")]
    movie_files = movie_files if nrows is None else movie_files[:nrows]
    rows = []
    for mf in movie_files:
        with open(os.path.join(movies_path, mf), "r", encoding="utf-8") as f:
            line = f.readline().strip()
            movie_id = line.rstrip(":")
            while True:
                line = f.readline().strip()
                if not line:
                    break

                user_id, rating, date = line.split(",")
                row = {
                    "movie_id": movie_id,
                    "user_id": user_id,
                    "rating": int(rating),
                    "date": date
                }
                rows.append(row)
    return pd.DataFrame(rows)

"""https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data"""

import os
import pandas as pd


def load_data_part(
    dataset_path: str,
    part_file: str,
    nrows: int | None = None
) -> pd.DataFrame:
    curr_movie_id = None
    rows = []
    with open(os.path.join(dataset_path, part_file), "r", encoding="utf-8") as f:
        while nrows is None or len(rows) < nrows:
            line = f.readline()
            if not line:
                break

            line = line.strip()
            if line.endswith(":"):
                curr_movie_id = line[:-1]
            else:
                user_id, rating, date = line.split(",")
                row = {
                    "movie_id": curr_movie_id,
                    "user_id": user_id,
                    "rating": rating,
                    "date": date
                }
                rows.append(row)
    return pd.DataFrame(rows)

def load_data(dataset_path: str) -> pd.DataFrame:
    dataset_path = os.path.expanduser(dataset_path)
    parts: list[pd.DataFrame] = []
    for i in range(1, 5):
        parts.append(load_data_part(dataset_path, f"combined_data_{i}.txt", nrows=100))
    return pd.concat(parts, ignore_index=True)

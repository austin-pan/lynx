from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import lynx as lx
from lynx.datasets import movielens


DATASET_PATH = "~/Downloads/lynx/datasets/movielens/ml-1m"
SEED = 0
VERBOSE = False

def load_train_test() -> Tuple[lx.Table, lx.Table, pd.Series, lx.Table, pd.Series]:
    ratings_data = movielens.load_ratings(
        DATASET_PATH,
        usecols=["user_id", "movie_id", "rating"],
        nrows=1000
    )
    movielens_table = (
        lx.Table(ratings_data, "ratings")
        .model_interactions("user_id", "movie_id")
        .onehot("movie_id")
        .onehot("user_id")
    )

    y = movielens_table.pop("rating")
    X_train, X_test, y_train, y_test = train_test_split(
        movielens_table, y,
        train_size=0.8,
        test_size=0.2,
        random_state=SEED
    )

    return movielens_table, X_train, y_train, X_test, y_test

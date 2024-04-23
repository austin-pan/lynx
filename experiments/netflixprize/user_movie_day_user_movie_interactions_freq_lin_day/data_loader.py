from typing import Tuple
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import lynx as lx
from lynx.datasets import netflixprize


DATASET_PATH = "~/Downloads/lynx/datasets/netflix/download"
SEED = 0
VERBOSE = False

def load_train_test() -> Tuple[lx.Table, lx.Table, pd.Series, lx.Table, pd.Series]:
    ratings_data = netflixprize.load_data(
        DATASET_PATH,
        num_movies=10
    )
    ratings_data["date"] = ratings_data["date"].apply(lambda d: datetime.strptime(d, "%Y-%m-%d"))
    ratings_data["day"] = ratings_data["date"].apply(lambda d: d.weekday())
    min_date = min(ratings_data["date"])
    ratings_data["lin_day"] = (
        ratings_data["date"]
        .sub(min_date)
        .apply(lambda d: d.days)
    )
    netflix_table = (
        lx.Table(ratings_data, "ratings")
        .model_interactions("user_id", "movie_id")
        .model_interactions("date", "user_id", freq=True)
        .onehot("movie_id")
        .onehot("user_id")
        .drop("date")
    )

    y = netflix_table.pop("rating")
    X_train, X_test, y_train, y_test = train_test_split(
        netflix_table, y,
        train_size=0.8,
        test_size=0.2,
        random_state=SEED
    )
    return netflix_table, X_train, y_train, X_test, y_test

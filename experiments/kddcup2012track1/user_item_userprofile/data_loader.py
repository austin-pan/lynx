from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import lynx as lx
from lynx.datasets import kddcup2012


DATASET_PATH = "~/Downloads/lynx/datasets/kddcup2012/kddcup2012-track1"
SEED = 0
VERBOSE = False

def load_train_test() -> Tuple[lx.Table, lx.Table, pd.Series, lx.Table, pd.Series]:
    train_data = kddcup2012.load_rec_log_train(
        DATASET_PATH,
        usecols=["user_id", "item_id", "result"],
        nrows=10000
    )
    train_table = lx.Table(train_data, "train")

    user_profiles_data = kddcup2012.load_user_profiles(
        DATASET_PATH,
        usecols=["user_id", "year_of_birth", "gender", "tag_ids", "num_tweets"],
        nrows=10000
    )
    user_profiles_table = (
        lx.Table(user_profiles_data, "user_profiles")
        .manyhot("tag_ids")
        .onehot("gender")
        .onehot("year_of_birth")
    )

    train_table = (
        train_table
        .merge(user_profiles_table, left_on="user_id", right_on="user_id")
        .onehot("user_id")
        .onehot("item_id")
    )
    y = train_table.pop("result")
    X_train, X_test, y_train, y_test = train_test_split(
        train_table, y,
        train_size=0.8,
        test_size=0.2,
        random_state=SEED
    )

    return train_table, X_train, y_train, X_test, y_test

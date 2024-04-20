from typing import Tuple
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
import lynx as lx
from lynx.table import block as B
from lynx.datasets import kddcup2012


DATASET_PATH = "~/Downloads/lynx/datasets/kddcup2012/kddcup2012-track1"
SEED = 0
VERBOSE = False

def load_train_test() -> Tuple[lx.Table, lx.Table, pd.Series, lx.Table, pd.Series]:
    train_data = kddcup2012.load_rec_log_train(
        DATASET_PATH,
        usecols=["user_id", "item_id", "result"],
        nrows=100000
    )
    train_table = lx.Table(train_data, "train")

    user_snses = kddcup2012.load_user_snses(
        DATASET_PATH,
        nrows=10000
    )
    friends = user_snses.groupby("follower_user_id").aggregate(list).reset_index()
    friends_table = (
        lx.Table(friends, "followers")
        .manyhot("followee_user_id", "friends")
    )

    train_table = (
        train_table
        .merge(friends_table, left_on="user_id", right_on="follower_user_id")
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

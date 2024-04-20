from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import lynx as lx
from lynx.datasets import bookcrossing


DATASET_PATH = "~/Downloads/lynx/datasets/bookcrossing/archive"
SEED = 0
VERBOSE = False

def load_train_test() -> Tuple[lx.Table, lx.Table, pd.Series, lx.Table, pd.Series]:
    ratings_data = bookcrossing.load_ratings(
        DATASET_PATH,
        nrows=1000
    )
    bx_table = (
        lx.Table(ratings_data, "ratings")
        .model_interactions("ISBN", "User-ID")
        .onehot("ISBN")
        .onehot("User-ID")
    )

    y = bx_table.pop("Book-Rating")
    X_train, X_test, y_train, y_test = train_test_split(
        bx_table, y,
        train_size=0.8,
        test_size=0.2,
        random_state=SEED
    )
    return bx_table, X_train, y_train, X_test, y_test

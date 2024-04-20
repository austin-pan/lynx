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
        nrows=10000
    )
    train_table = lx.Table(train_data, "train")

    user_keywords = kddcup2012.load_user_keywords(
        DATASET_PATH,
        nrows=10000
    )
    keywords_block = B.SparseBlock("keywords", _expand_keywords(user_keywords["keywords"]))
    user_id_block = B.DenseBlock("user_id", user_keywords["user_id"].to_frame())
    user_keywords_table = lx.Table([user_id_block, keywords_block])

    train_table = (
        train_table
        .merge(user_keywords_table, left_on="user_id", right_on="user_id")
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

def _expand_keywords(keywords: pd.Series) -> sparse.csr_matrix:
    keys = set()
    for key_weights in keywords:
        for key, weight in key_weights.items():
            keys.add(key)
    col_mapping = { v: k for k, v in enumerate(keys) }

    mats = []
    for key_weights in keywords:
        cols = []
        data = []
        for key, weight in key_weights.items():
            cols.append(col_mapping[key])
            data.append(weight)
        rows = [0]*len(cols)
        mat = sparse.csr_matrix((data, (rows, cols)), shape=(1, len(col_mapping)))
        mats.append(mat)
    return sparse.vstack(mats, format="csr") # type: ignore

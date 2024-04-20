"""
Produce RMSE and timing statistics for FastFM. Note: FastFM only works with
Python 3.6 or lower.
"""

import time
from typing import Union

import pandas as pd
from fastFM import mcmc
from sklearn.metrics import mean_squared_error

import lynx as lx


def run(
    X_train: lx.Table,
    y_train: pd.Series,
    X_test: lx.Table,
    y_test: pd.Series,
    *,
    seed: Union[int, None] = None,
    outpath: Union[str, None] = None
) -> None:
    X_train_csr = X_train.to_csr_matrix()
    X_test_csr = X_test.to_csr_matrix()

    results = []
    for k in [1, 2, 4, 8]:
        print(f"k={k}")

        fm = mcmc.FMRegression(n_iter=50, rank=k, init_stdev=0.1, random_state=seed)

        start_time = time.perf_counter()
        predictions = fm.fit_predict(X_train_csr, y_train, X_test_csr)
        end_time = time.perf_counter()

        rmse = mean_squared_error(predictions, y_test, squared=False)
        train_time = end_time - start_time

        row = {"k": k, "train_time": train_time, "rmse": rmse}
        results.append(row)
        print(row)
    results_frame = pd.DataFrame(results)
    print(results_frame)
    if outpath is not None:
        results_frame.to_csv(outpath)

import time
from typing import Union

import pandas as pd
from sklearn.metrics import mean_squared_error

import lynx as lx
from lynx.libfm.bs import mcmc


def run(
    data: lx.Table,
    X_train: lx.Table,
    y_train: pd.Series,
    X_test: lx.Table,
    y_test: pd.Series,
    *,
    seed: Union[int, None] = None,
    verbose: bool = False,
    outpath: Union[str, None] = None
) -> None:
    print(f"NNZ(BS): {data.block_nnz:,}")

    results = []
    for k in [1, 2, 4, 8]:
        print(f"k={k}")

        fm = mcmc.FMRegression(iter_num=50, dim=(1, 1, k), seed=seed)

        fm.write(X_train, y_train, X_test, verbose=verbose)

        start_time = time.perf_counter()
        cmd_output = fm.train(verbose=verbose)
        end_time = time.perf_counter()

        predictions = fm.get_predictions()

        rmse = mean_squared_error(predictions, y_test, squared=False)
        train_time = end_time - start_time

        row = {"k": k, "train_time": train_time, "rmse": rmse}
        results.append(row)
        print(row)
        fm.flush()
    results_frame = pd.DataFrame(results)
    print(results_frame)
    if outpath is not None:
        results_frame.to_csv(outpath)

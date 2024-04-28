import time

import data_loader
import pandas as pd
from sklearn.metrics import mean_squared_error

from lynx.libfm.bs import mcmc


OUTPATH = None

rating_table, X_train, y_train, X_test, y_test = data_loader.load_train_test()

print(f"NNZ(BS): {rating_table.block_nnz:,}")

results = []
for k in [128]:
    print(f"k={k}")

    fm = mcmc.FMRegression(iter_num=100, dim=(1, 1, k), seed=data_loader.SEED)

    fm.write(X_train, y_train, X_test, verbose=data_loader.VERBOSE)

    start_time = time.perf_counter()
    cmd_output = fm.train(verbose=data_loader.VERBOSE)
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
if OUTPATH is not None:
    results_frame.to_csv(OUTPATH)

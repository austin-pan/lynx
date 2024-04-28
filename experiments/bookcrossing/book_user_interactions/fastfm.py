"""
Produce RMSE and timing statistics for FastFM. Note: FastFM only works with
Python 3.6 or lower.
"""
import time

import data_loader
import pandas as pd
from fastFM import mcmc
from sklearn.metrics import mean_squared_error


OUTPATH = None

_, X_train, y_train, X_test, y_test = data_loader.load_train_test()

X_train_csr = X_train.to_csr_matrix()
X_test_csr = X_test.to_csr_matrix()

results = []
for k in [1, 2, 4, 8]:
    print(f"k={k}")

    fm = mcmc.FMRegression(n_iter=50, rank=k, init_stdev=0.1, random_state=data_loader.SEED)

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
if OUTPATH is not None:
    results_frame.to_csv(OUTPATH)

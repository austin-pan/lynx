import time
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from fastFM import mcmc
import lynx as lx
from lynx.datasets import movielens


DATASET_PATH = "~/Downloads/lynx/datasets/movielens/ml-1m"
SEED = 0

# users_table = lx.Table(movielens.load_users(DATASET_PATH), "users")
# movies_table = lx.Table(movielens.load_movies(DATASET_PATH), "movies")

ratings_data = movielens.load_ratings(
    DATASET_PATH,
    usecols=["user_id", "movie_id", "rating"],
    # nrows=1000
)
ratings_table = (
    lx.Table(ratings_data, "ratings")
    .model_interactions("user_id", "movie_id")
    .onehot("movie_id")
    .onehot("user_id")
)

y = ratings_table.pop("rating")
X_train, X_test, y_train, y_test = train_test_split(
    ratings_table, y,
    train_size=0.8,
    test_size=0.2,
    random_state=SEED
)

results = []
for k in [1, 2, 4, 8]:
    print(f"k={k}")

    fm = mcmc.FMRegression(rank=k, init_stdev=0.1)
    start_time = time.perf_counter()
    predictions = fm.fit_predict(
        X_train.to_csr_matrix(),
        y_train,
        X_test.to_csr_matrix()
    )
    end_time = time.perf_counter()
    rmse = mean_squared_error(predictions, y_test, squared=False)

    results.append({"k": k, "train_time": end_time - start_time, "rmse": rmse})
print(pd.DataFrame(results))

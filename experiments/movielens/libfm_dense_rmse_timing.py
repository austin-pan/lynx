import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import lynx as lx
from lynx.datasets import movielens
from lynx.libfm import mcmc


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

    fm = mcmc.FMRegression(dim=(1, 1, k))
    predictions_path = os.path.join(fm.mat_dir, "predictions.txt")

    fm.write(X_train, y_train, X_test, verbose=True)
    train_time = fm.train(predictions_path, verbose=True)
    predictions = fm.get_predictions(predictions_path)
    rmse = mean_squared_error(predictions, y_test, squared=False)

    results.append({"k": k, "train_time": train_time, "rmse": rmse})
    fm.flush()
print(pd.DataFrame(results))

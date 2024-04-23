# `lynx`

`lynx` is a library that simplifies taking advantage of relational data in machine learning. Behind the curtains, this is achieved by not materializing joins in memory and only keeping track of row mappings of join operations. This format allows certain algorithms to take advantage of the smaller data footprint to speed up learning while retaining prediction performance.

`lynx` also provides a user-friendly interface for working with `libFM` (http://www.libfm.org/), a factorization machine utility that can take advantage of relations using its own block structure schema and learning algorithm.

For ease of use and experiment reproducibility, `lynx` also provides loaders for popular datasets such as MovieLens and BookCrossing.

## Requirements

* Unix OS
* Python 3.6+

## Installation

We recommend using `conda` (https://docs.conda.io/en/latest/) environments to keep library environments clean.

```sh
pip install -e 'git+https://github.com/austin-pan/lynx#egg=lynx'
```

### libFM

To use libFM functionalities, you must install libFM by doing the following:

1. Install libFM either from the [repository](https://github.com/srendle/libfm) (more up-to-date) or their [website](https://www.libfm.org)

```sh
git clone https://github.com/srendle/libfm.git
cd libfm
make all # Build latest libFM from source
```

2. Set the environment variable `LIBFM_HOME` to the base directory of the LibFM directory in your preferred run commands file, e.g. `.zshrc`, `.bashrc`, `.bash_profile`, `.profile`, etc.

```sh
echo "export LIBFM_HOME=$(pwd)" >> ~/.zshrc # Use your preferred terminal initialization file
```

## Example

A detailed tutorial can be viewed at [examples/tutorial.ipynb](examples/tutorial.ipynb). And extra examples can be viewed in the [examples](examples) directory.

`lynx` introduces a new data structure, `Table`, for working with tabular data while retaining relational information which can then be used by specialized algorithms that take advantage of relations. Here's a quick example for working with relational data and using it with libFM's block structure algorithm.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

import lynx as lx
from lynx.libfm.bs import als

# Create ratings table
movie_ratings = pd.DataFrame(
    [
        ["Alice", "TI", 5],
        ["Alice", "NH", 3],
        ["Bob", "SW", 4],
        ["Bob", "ST", 5],
    ],
    columns=["user_id", "movie_id", "rating"]
)
movie_ratings_table = lx.Table(movie_ratings, "movie_ratings")

# Create users table
users = pd.DataFrame(
    [
        ["Alice", 20, "F"],
        ["Bob", 35, "M"]
    ],
    columns=["user_id", "age_group", "gender"]
)
users_table = lx.Table(users, "users")

# Merge ratings and users tables
merged_table = (
    movie_ratings_table
    .merge(users_table, left_on="user_id", right_on="user_id")
)
print(merged_table.to_dataframe())
#   user_id movie_id  rating  age_group gender
# 0   Alice       TI       5         20      F
# 1   Alice       NH       3         20      F
# 2     Bob       SW       4         35      M
# 3     Bob       ST       5         35      M

# One-hot encode features in the merged table
merged_table = (
    merged_table
    .onehot("age_group")
    .onehot("gender")
    .onehot("user_id")
    .onehot("movie_id")
)
print(merged_table.to_dataframe())
#    rating  0  1  0  1  0  1  0  1  2  3
# 0       5  1  0  1  0  1  0  0  0  0  1
# 1       3  1  0  1  0  1  0  1  0  0  0
# 2       4  0  1  0  1  0  1  0  0  1  0
# 3       5  0  1  0  1  0  1  0  1  0  0
###
### Same as
###         | age_group | gender | user_id | movie_id
#    rating | 0  1      | 0  1   | 0  1    | 0  1  2  3
# ----------|-----------|--------|---------|-----------
# 0       5 | 1  0      | 1  0   | 1  0    | 0  0  0  1
# 1       3 | 1  0      | 1  0   | 1  0    | 1  0  0  0
# 2       4 | 0  1      | 0  1   | 0  1    | 0  0  1  0
# 3       5 | 0  1      | 0  1   | 0  1    | 0  1  0  0

# Create train-test split
y = merged_table.pop("rating")
X_train, X_test, y_train, y_test = train_test_split(
    merged_table, y,
    random_state=0
)

# Run libFM regression task using the ALS block structure algorithm
fm = als.FMRegression(seed=0)
fm.fit(X_train, y_train)
predictions = fm.predict(X_test)
# Delete temporary libFM files
fm.flush()

print(X_test.to_dataframe())
#    0  1  0  1  0  1  0  1  2  3
# 0  0  1  0  1  0  1  0  0  1  0
###
### Same as
### age_group | gender | user_id | movie_id
#    0  1     | 0  1   | 0  1    | 0  1  2  3
# ------------|--------|---------|-----------
# 0  0  1     | 0  1   | 0  1    | 0  0  1  0

print(predictions)
# [5.0]
```

## Experiments

Files to reproduce some of the experiments in "[Scaling Factorization Machines to Relational Data](https://www.vldb.org/pvldb/vol6/p337-rendle.pdf)" using `lynx` can be found in the [experiments](experiments) directory.

## Contributions

All forms of contributions are very welcome! Anything from documentation and tests to more dataset loaders and feature expansion would be appreciated!

## TODOs

- [ ] Unit tests (yikes)

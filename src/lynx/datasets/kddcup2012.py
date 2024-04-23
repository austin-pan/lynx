"""https://www.kaggle.com/competitions/kddcup2012-track1/data"""


import os
from typing import List, Tuple, Union

import pandas as pd


def download() -> None:
    """
    Dataset is hosted on Kaggle at https://www.kaggle.com/competitions/kddcup2012-track1/data.
    Please download and unzip the zip files:
        * `rec_log_test.txt.zip`
        * `track1.zip`

    The resulting file tree should look something like:

    ```
    .../dataset_path/
        - rec_log_test.txt
        - track1/
            - item.txt
            - rec_log_train.txt
            - user_action.txt
            - user_key_word.txt
            - user_profile.txt
            - user_sns.txt
    ```
    """
    raise NotImplementedError(
        "Please download the dataset zip files from " +
        "https://www.kaggle.com/competitions/kddcup2012-track1/data"
    )

def load_rec_log_train(
    dataset_path: str,
    usecols: Union[List[str], None] = None,
    nrows: Union[int, None] = None
) -> pd.DataFrame:
    dataset_path = os.path.expanduser(dataset_path)
    train_file = os.path.join(dataset_path, "track1", "rec_log_train.txt")
    return _load_dataset(train_file, usecols=usecols, nrows=nrows)

def load_rec_log_test(
    dataset_path: str,
    usecols: Union[List[str], None] = None,
    nrows: Union[int, None] = None
) -> pd.DataFrame:
    dataset_path = os.path.expanduser(dataset_path)
    train_file = os.path.join(dataset_path, "rec_log_test.txt")
    return _load_dataset(train_file, usecols=usecols, nrows=nrows)

def _load_dataset(
    filepath: str,
    usecols: Union[List[str], None] = None,
    nrows: Union[int, None] = None
) -> pd.DataFrame:
    return pd.read_csv(
        filepath,
        sep="\t",
        names=["user_id", "item_id", "result", "timestamp"],
        usecols=usecols,
        nrows=nrows
    )

def load_user_profiles(
    dataset_path: str,
    usecols: Union[List[str], None] = None,
    nrows: Union[int, None] = None
) -> pd.DataFrame:
    dataset_path = os.path.expanduser(dataset_path)
    profiles = pd.read_csv(
        os.path.join(dataset_path, "track1", "user_profile.txt"),
        sep="\t",
        names=["user_id", "year_of_birth", "gender", "num_tweets", "tag_ids"],
        usecols=usecols,
        nrows=nrows
    )
    profiles["tag_ids"] = profiles["tag_ids"].str.split(";")
    profiles["tag_ids"] = profiles["tag_ids"].apply(lambda x: [] if x == ["0"] else x)
    return profiles

def load_items(
    dataset_path: str,
    usecols: Union[List[str], None] = None,
    nrows: Union[int, None] = None
) -> pd.DataFrame:
    dataset_path = os.path.expanduser(dataset_path)
    items = pd.read_csv(
        os.path.join(dataset_path, "track1", "item.txt"),
        sep="\t",
        names=["item_id", "item_category", "item_keywords"],
        usecols=usecols,
        nrows=nrows
    )
    items["item_keywords"] = items["item_keywords"].str.split(";")
    return items

def load_user_actions(
    dataset_path: str,
    usecols: Union[List[str], None] = None,
    nrows: Union[int, None] = None
) -> pd.DataFrame:
    dataset_path = os.path.expanduser(dataset_path)
    return pd.read_csv(
        os.path.join(dataset_path, "track1", "user_action.txt"),
        sep="\t",
        names=[
            "user_id",
            "action_destination_user_id",
            "num_at_actions",
            "num_retweets",
            "num_comments"
        ],
        usecols=usecols,
        nrows=nrows
    )

def load_user_snses(
    dataset_path: str,
    usecols: Union[List[str], None] = None,
    nrows: Union[int, None] = None
) -> pd.DataFrame:
    dataset_path = os.path.expanduser(dataset_path)
    return pd.read_csv(
        os.path.join(dataset_path, "track1", "user_sns.txt"),
        sep="\t",
        names=["follower_user_id", "followee_user_id"],
        usecols=usecols,
        nrows=nrows
    )

def load_user_keywords(
    dataset_path: str,
    usecols: Union[List[str], None] = None,
    nrows: Union[int, None] = None
) -> pd.DataFrame:
    def parse_key_weight(key_weight: str) -> Tuple[str, float]:
        key, weight = key_weight.split(":")
        return (key, float(weight))

    dataset_path = os.path.expanduser(dataset_path)
    keywords = pd.read_csv(
        os.path.join(dataset_path, "track1", "user_key_word.txt"),
        sep="\t",
        names=["user_id", "keywords"],
        usecols=usecols,
        nrows=nrows
    )
    keywords["keywords"] = (
        keywords["keywords"]
        .str.split(";")
        .apply(lambda key_weights: dict([parse_key_weight(kw) for kw in key_weights]))
    )
    return keywords

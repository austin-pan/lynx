# Experiments

This directory contains files for reproducing some of the experiments mentioned in "[Scaling Factorization Machines to Relational Data](https://www.vldb.org/pvldb/vol6/p337-rendle.pdf)". The subdirectories are the names of the datasets that the experiments are for.

The experiments here are for examining the performance of `libFM`, specifically the effects of using block structure which takes advantage of relational data.

## Datasets

`lynx` provides loaders for popular datasets which these experiments use. Currently, the datasets that are supported are:

* MovieLens-1M (http://files.grouplens.org/datasets/movielens/ml-1m.zip)
* BookCrossing (https://www.kaggle.com/datasets/syedjaferk/book-crossing-dataset)
* Netflix Prize (https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)
* KDDCup 2012 Track 1 (https://www.kaggle.com/competitions/kddcup2012-track1/data)

## `fastFM`

There are also files for using `fastFM` (https://ibayer.github.io/fastFM/) as comparisons in the experiments. `fastFM` will need to be installed for those experiments to run successfully. Note that `fastFM` can only run in `python3.6`.

A simple way to install `python3.6` is to create a `conda` environment and install `python3.6` in it. The following commands assume that `conda` is installed (https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

```sh
conda create -n fastfm
conda activate fastfm

CONDA_SUBDIR=osx-64 conda install python=3.6
pip install fastfm
```

Note that `lynx` is compatible with `python3.6`.

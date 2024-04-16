


import os
import shutil
import time
from typing import Iterable

import lynx as lx
from lynx import libfm


class LibFMTask:

    def __init__(
        self,
        method: str,
        task: str,
        train_file: str,
        test_file: str,
        *,
        cache_size: int | None = None,
        dim: tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        learn_rate: float | tuple[float, float, float] | None = None, # SGD only
        meta: str | None = None,
        regularizations: float | tuple[float, float, float] | None = None, # Only SGD and ALS
        rlog: str | None = None,
        seed: int | None = None,
        validation: str | None = None, # Only SGDA
        verbosity: int | None = None,

        mat_dir: str | None = None
    ):
        self.method = method
        self.task = task
        self.train_file = train_file
        self.test_file = test_file

        self.cache_size = cache_size
        self.dim = dim
        self.init_stdev = init_stdev
        self.iter_num = iter_num
        self.learn_rate = learn_rate
        self.meta = meta
        self.regularizations = regularizations
        self.rlog = rlog
        self.seed = seed
        self.validation = validation
        self.verbosity = verbosity

        if mat_dir is None:
            dir_name = f"libfm_{self.method}_{self.task}_{str(time.time()).replace('.', '')}"
            self.mat_dir = os.path.join("/tmp", dir_name)
        else:
            self.mat_dir = mat_dir
        os.makedirs(self.mat_dir, exist_ok=True)

    def run(self, **kwargs) -> None:
        """Run LibFM with specified flags and values."""
        args = {
            "method": self.method,
            "task": self.task,
            "train": self.train_file,
            "test": self.test_file,
            "cache_size": self.cache_size,
            "dim": self.dim,
            "init_stdev": self.init_stdev,
            "iter_num": self.iter_num,
            "learn_rate": self.learn_rate,
            "meta": self.meta,
            "regular": self.regularizations,
            "rlog": self.rlog,
            "seed": self.seed,
            "validation": self.validation,
            "verbosity": self.verbosity
        }
        args.update(kwargs)
        libfm.run(self.mat_dir, **args)

    def flush(self) -> None:
        """Remove LibFM files."""
        shutil.rmtree(self.mat_dir)

class StatelessLibFMTask(LibFMTask):

    def __init__(
        self,
        method: str,
        task: str,
        train_file: str,
        test_file: str,
        *,
        cache_size: int | None = None,
        dim: tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        learn_rate: float | tuple[float, float, float] | None = None, # SGD only
        meta: str | None = None,
        regularizations: float | tuple[float, float, float] | None = None, # Only SGD and ALS
        rlog: str | None = None,
        seed: int | None = None,
        validation: str | None = None, # Only SGDA
        verbosity: int | None = None,

        mat_dir: str | None = None
    ):
        super().__init__(
            method,
            task,
            train_file,
            test_file,
            cache_size=cache_size,
            dim=dim,
            init_stdev=init_stdev,
            iter_num=iter_num,
            learn_rate=learn_rate,
            meta=meta,
            regularizations=regularizations,
            rlog=rlog,
            seed=seed,
            validation=validation,
            verbosity=verbosity,

            mat_dir = mat_dir
        )

    def write(
        self,
        X_train: lx.Table,
        y_train: Iterable[float | int],
        X_test: lx.Table,
        verbose: bool = False
    ) -> None:
        raise NotImplementedError()

    def train(
        self,
        outpath: str,
        verbose: bool = False,
        time_only: bool = False
    ) -> float:
        raise NotImplementedError()

    def fit_predict(
        self,
        X_train: lx.Table,
        y_train: Iterable[float | int],
        X_test: lx.Table,
        verbose: bool = False
    ) -> list[float]:
        raise NotImplementedError()

class StatefulLibFMTask(LibFMTask):

    def __init__(
        self,
        method: str,
        task: str,
        train_file: str,
        test_file: str,
        *,
        cache_size: int | None = None,
        dim: tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        learn_rate: float | tuple[float, float, float] | None = None, # SGD only
        load_model: str | None = None,
        meta: str | None = None,
        regularizations: float | tuple[float, float, float] | None = None, # Only SGD and ALS
        rlog: str | None = None,
        seed: int | None = None,
        validation: str | None = None, # Only SGDA
        verbosity: int | None = None,

        mat_dir: str | None = None
    ):
        super().__init__(
            method,
            task,
            train_file,
            test_file,
            cache_size=cache_size,
            dim=dim,
            init_stdev=init_stdev,
            iter_num=iter_num,
            learn_rate=learn_rate,
            meta=meta,
            regularizations=regularizations,
            rlog=rlog,
            seed=seed,
            validation=validation,
            verbosity=verbosity,

            mat_dir = mat_dir
        )
        self.load_model = load_model
        self.model_name = f"{method}_model"

    def write(
        self,
        X_train: lx.Table,
        y_train: Iterable[float | int],
        verbose: bool = False
    ) -> None:
        raise NotImplementedError()

    def train(self, verbose: bool = False, time_only: bool = False) -> float:
        raise NotImplementedError()

    def fit(
        self,
        X_train: lx.Table,
        y_train: Iterable[float | int],
        verbose: bool = False
    ) -> None:
        raise NotImplementedError()

    def predict(self, X_test: lx.Table, verbose: bool = False) -> list[float]:
        raise NotImplementedError()

    def save_model(self, path: str) -> None:
        # self.run(save_model=path, iter_num=0, verbose=verbose)
        model_path = os.path.join(self.mat_dir, self.model_name)
        shutil.copy(model_path, path)

class DenseStatefulLibFMTask(StatefulLibFMTask):

    def __init__(
        self,
        method: str,
        task: str,
        train_file: str,
        test_file: str,
        *,
        cache_size: int | None = None,
        dim: tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        learn_rate: float | tuple[float, float, float] | None = None, # SGD only
        load_model: str | None = None,
        meta: str | None = None,
        regularizations: float | tuple[float, float, float] | None = None, # Only SGD and ALS
        rlog: str | None = None,
        seed: int | None = None,
        validation: str | None = None, # Only SGDA
        verbosity: int | None = None,

        mat_dir: str | None = None
    ):
        super().__init__(
            method,
            task,
            train_file,
            test_file,
            cache_size=cache_size,
            dim=dim,
            init_stdev=init_stdev,
            iter_num=iter_num,
            learn_rate=learn_rate,
            meta=meta,
            regularizations=regularizations,
            rlog=rlog,
            seed=seed,
            validation=validation,
            verbosity=verbosity,

            mat_dir = mat_dir
        )
        self.load_model = load_model
        self.model_name = f"{method}_model"

    def write(
        self,
        X_train: lx.Table,
        y_train: Iterable[float | int],
        verbose: bool = False
    ) -> None:
        # Write train libfm files
        libfm_train_file = f"{self.train_file}.libfm"
        train_file_path = os.path.join(self.mat_dir, libfm_train_file)
        lx.write.libfm.dense(X_train, y_train, train_file_path)
        libfm.create_dense_binaries(self.mat_dir, libfm_train_file, verbose=verbose)
        # Write empty test libfm files
        libfm_test_file = f"{self.test_file}.libfm"
        test_file_path = os.path.join(self.mat_dir, libfm_test_file)
        lx.write.libfm.dense(X_train, y_train, test_file_path, empty=True)
        libfm.create_dense_binaries(self.mat_dir, libfm_test_file, verbose=verbose)

    def train(self, verbose: bool = False, time_only: bool = False) -> float:
        """time_only = False is necessary for `predict` to work!"""
        start_time = time.perf_counter()
        if not time_only:
            self.run(save_model=self.model_name, seed=self.seed, verbose=verbose)
        else:
            self.run(seed=self.seed, verbose=verbose)
        end_time = time.perf_counter()
        return end_time - start_time

    def fit(
        self,
        X_train: lx.Table,
        y_train: Iterable[float | int],
        verbose: bool = False
    ) -> None:
        self.write(X_train, y_train, verbose=verbose)
        self.train(verbose=verbose)

    def predict(self, X_test: lx.Table, verbose: bool = False) -> list[float]:
        # NOTE `predict` is not reproducible unless a seed has been used because it
        # requires 1 iteration of training to get predictions
        outpath = os.path.join(self.mat_dir, "predictions.txt")
        # Dummy y_test
        y_test = [0] * X_test.height
        libfm_test_file = f"{self.test_file}.libfm"
        test_file_path = os.path.join(self.mat_dir, libfm_test_file)
        lx.write.libfm.dense(X_test, y_test, test_file_path)
        libfm.create_dense_binaries(self.mat_dir, libfm_test_file, verbose=verbose)

        self.run(
            load_model=self.model_name,
            seed=self.seed,
            iter_num=1,
            out=outpath,
            verbose=verbose
        )

        predictions = []
        with open(outpath, "r", encoding="utf-8") as f:
            predictions = f.readlines()
        return list(map(float, predictions))

"""Block structure ALS tasks."""

import os
from typing import Iterable

import lynx as lx
from lynx import libfm
from lynx.libfm import tasks


class SGDATask(tasks.DenseStatefulLibFMTask):

    def __init__(
        self,
        task: str,
        learn_rate: float | tuple[float, float, float], # Only SGD and SGDA
        X_validation: lx.Table, # Only SGDA
        y_validation: Iterable[float | int], # Only SGDA
        *,
        regularizations: float | tuple[float, float, float] | None = None, # Only SGD and ALS
        cache_size: int | None = None,
        dim: tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        load_model: str | None = None,
        meta: str | None = None,
        rlog: str | None = None,
        seed: int | None = None,
        verbosity: int | None = None
    ):
        super().__init__(
            method="sgda",
            task=task,
            train_file="train",
            test_file="test",
            cache_size=cache_size,
            dim=dim,
            init_stdev=init_stdev,
            iter_num=iter_num,
            learn_rate=learn_rate,
            load_model=load_model,
            meta=meta,
            regularizations=regularizations,
            rlog=rlog,
            seed=seed,
            validation="validation",
            verbosity=verbosity
        )
        self.write_validation(X_validation, y_validation)

    def write_validation(
        self,
        X_validation: lx.Table,
        y_validation: Iterable[float | int],
        verbose: bool = False
    ) -> None:
        validation_path = os.path.join(self.mat_dir, f"{self.validation}.libfm")
        lx.write.libfm.dense(X_validation, y_validation, validation_path)
        libfm.create_dense_binaries(self.mat_dir, validation_path, verbose=verbose)

    def fit_validation(
        self,
        X_train: lx.Table,
        y_train: Iterable[float | int],
        X_validation: lx.Table,
        y_validation: Iterable[float | int],
        verbose: bool = False
    ) -> None:
        self.write_validation(X_validation, y_validation, verbose=verbose)
        super().fit(X_train, y_train, verbose=verbose)

    def fit(
        self,
        X_train: lx.Table,
        y_train: Iterable[float | int],
        verbose: bool = False
    ) -> None:
        raise AttributeError("SGDA uses `fit_validation`")

class FMRegression(SGDATask):

    def __init__(
        self,
        X_validation: lx.Table, # Only SGDA
        y_validation: Iterable[float | int], # Only SGDA
        learn_rate: float | tuple[float, float, float], # Only SGD and SGDA
        *,
        regularizations: float | tuple[float, float, float] | None = None, # Only SGD and ALS
        cache_size: int | None = None,
        dim: tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        load_model: str | None = None,
        meta: str | None = None,
        rlog: str | None = None,
        seed: int | None = None,
        verbosity: int | None = None
    ):
        super().__init__(
            X_validation=X_validation,
            y_validation=y_validation,
            task="r",
            cache_size=cache_size,
            dim=dim,
            init_stdev=init_stdev,
            iter_num=iter_num,
            learn_rate=learn_rate,
            load_model=load_model,
            meta=meta,
            regularizations=regularizations,
            rlog=rlog,
            seed=seed,
            verbosity=verbosity
        )

class FMClassification(SGDATask):

    def __init__(
        self,
        learn_rate: float | tuple[float, float, float], # Only SGD and SGDA
        X_validation: lx.Table, # Only SGDA
        y_validation: Iterable[float | int], # Only SGDA
        *,
        regularizations: float | tuple[float, float, float] | None = None, # Only SGD and ALS
        cache_size: int | None = None,
        dim: tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        load_model: str | None = None,
        meta: str | None = None,
        rlog: str | None = None,
        seed: int | None = None,
        verbosity: int | None = None
    ):
        super().__init__(
            X_validation=X_validation,
            y_validation=y_validation,
            task="c",
            cache_size=cache_size,
            dim=dim,
            init_stdev=init_stdev,
            iter_num=iter_num,
            learn_rate=learn_rate,
            load_model=load_model,
            meta=meta,
            regularizations=regularizations,
            rlog=rlog,
            seed=seed,
            verbosity=verbosity
        )

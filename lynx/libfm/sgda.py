"""Dense structure SGDA tasks."""

import os
from typing import Iterable, Tuple, Union

import lynx as lx
from lynx import libfm
from lynx.libfm import tasks


class SGDATask(tasks.DenseStatefulLibFMTask):
    """Adaptive Stochastic Gradient Descent task."""

    def __init__(
        self,
        task: str,
        learn_rate: Union[float, Tuple[float, float, float]], # Only SGD and SGDA
        *,
        regularizations: Union[float, Tuple[float, float, float], None] = None, # Only SGD and ALS
        cache_size: Union[int, None] = None,
        dim: Tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        load_model: Union[str, None] = None,
        meta: Union[str, None] = None,
        rlog: Union[str, None] = None,
        seed: Union[int, None] = None,
        verbosity: Union[int, None] = None
    ):
        """
        Args:
            task (str): "r"=regression, "c"=binary classification.
            learn_rate (float | None, optional): learn_rate for SGD. Defaults to
            None.
            regularizations (int | Tuple[int, int, int] | None, optional): (r0,r1,r2) for
            SGD and ALS: r0=bias regularization, r1=1-way regularization, r2=2-way
            regularization. Defaults to None.
            cache_size (int | None, optional): Cache size for data storage (only
            applicable if data is in binary format). Defaults to None.
            dim (Tuple[int, int, int], optional): (k0,k1,k2): k0=use bias,
            k1=use 1-way interactions, k2=dim of 2-way interactions. Defaults to
            (1, 1, 8).
            init_stdev (float, optional): Standard deviation for initialization of
            2-way factors. Defaults to 0.1.
            iter_num (int, optional): number of iterations. Defaults to 100.
            load_model (str | None, optional): Filename with saved model to load.
            Defaults to None.
            meta (str | None, optional): Filename for meta (group) information about
            data set. Defaults to None.
            rlog (str | None): Filename to write iterative measurements to.
            Defaults to None.
            seed (int | None, optional): Random state seed. Defaults to None.
            verbosity (int | None, optional): How much info to output to internal
            command line.
        """
        super().__init__(
            method="sgda",
            task=task,
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
            validation="_libfm_validation",
            verbosity=verbosity
        )

    def write_validation(
        self,
        X_validation: lx.Table,
        y_validation: Iterable[Union[float, int]],
        verbose: bool = False
    ) -> None:
        """
        Write out validation data in libFM SVMLight format and produce
        corresponding binaries.

        Args:
            X_validation (lx.Table): Validation data.
            y_validation (Iterable[Union[float, int]]): Validation targets.
            verbose (bool, optional): _description_. Defaults to False.
        """
        validation_path = os.path.join(self.mat_dir, f"{self.validation}.libfm")
        lx.write.libfm.dense(X_validation, y_validation, validation_path)
        libfm.create_dense_binaries(self.mat_dir, validation_path, verbose=verbose)

    def fit_validation(
        self,
        X_train: lx.Table,
        y_train: Iterable[Union[float, int]],
        X_validation: lx.Table,
        y_validation: Iterable[Union[float, int]],
        verbose: bool = False
    ) -> None:
        """
        Fit FM using validation data. Convenience function for
        `write_validation`, `write`, and then `train`.

        Do not use for timing. To get training time, run `write_validation` and
        `write` and then time `train`.

        Args:
            X_train (lx.Table): Train data.
            y_train (Iterable[float | int]]): Train data targets.
            X_validatoin (lx.Table): Validation data.
            y_validation (Iterable[float | int]): validation targets.
            verbose (bool, optional): Whether to print train progress. Train
            statistics should be ignored. Defaults to False.
        """
        self.write_validation(X_validation, y_validation, verbose=verbose)
        self.write(X_train, y_train, verbose=verbose)
        self.train(verbose=verbose)

    def fit(
        self,
        X_train: lx.Table,
        y_train: Iterable[Union[float, int]],
        verbose: bool = False
    ) -> None:
        raise AttributeError("SGDA uses `fit_validation`")

class FMRegression(SGDATask):
    """Adaptive Stochastic Gradient Descent regression task."""

    def __init__(
        self,
        learn_rate: Union[float, Tuple[float, float, float]], # Only SGD and SGDA
        *,
        regularizations: Union[float, Tuple[float, float, float], None] = None, # Only SGD and ALS
        cache_size: Union[int, None] = None,
        dim: Tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        load_model: Union[str, None] = None,
        meta: Union[str, None] = None,
        rlog: Union[str, None] = None,
        seed: Union[int, None] = None,
        verbosity: Union[int, None] = None
    ):
        """
        Args:
            learn_rate (float | None, optional): learn_rate for SGD. Defaults to
            None.
            regularizations (int | Tuple[int, int, int] | None, optional): (r0,r1,r2) for
            SGD and ALS: r0=bias regularization, r1=1-way regularization, r2=2-way
            regularization. Defaults to None.
            cache_size (int | None, optional): Cache size for data storage (only
            applicable if data is in binary format). Defaults to None.
            dim (Tuple[int, int, int], optional): (k0,k1,k2): k0=use bias,
            k1=use 1-way interactions, k2=dim of 2-way interactions. Defaults to
            (1, 1, 8).
            init_stdev (float, optional): Standard deviation for initialization of
            2-way factors. Defaults to 0.1.
            iter_num (int, optional): number of iterations. Defaults to 100.
            load_model (str | None, optional): Filename with saved model to load.
            Defaults to None.
            meta (str | None, optional): Filename for meta (group) information about
            data set. Defaults to None.
            rlog (str | None): Filename to write iterative measurements to.
            Defaults to None.
            seed (int | None, optional): Random state seed. Defaults to None.
            verbosity (int | None, optional): How much info to output to internal
            command line.
        """
        super().__init__(
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
    """Adaptive Stochastic Gradient Descent regression task."""

    def __init__(
        self,
        learn_rate: Union[float, Tuple[float, float, float]], # Only SGD and SGDA
        *,
        regularizations: Union[float, Tuple[float, float, float], None] = None, # Only SGD and ALS
        cache_size: Union[int, None] = None,
        dim: Tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        load_model: Union[str, None] = None,
        meta: Union[str, None] = None,
        rlog: Union[str, None] = None,
        seed: Union[int, None] = None,
        verbosity: Union[int, None] = None
    ):
        """
        Args:
            learn_rate (float | None, optional): learn_rate for SGD. Defaults to
            None.
            regularizations (int | Tuple[int, int, int] | None, optional): (r0,r1,r2) for
            SGD and ALS: r0=bias regularization, r1=1-way regularization, r2=2-way
            regularization. Defaults to None.
            cache_size (int | None, optional): Cache size for data storage (only
            applicable if data is in binary format). Defaults to None.
            dim (Tuple[int, int, int], optional): (k0,k1,k2): k0=use bias,
            k1=use 1-way interactions, k2=dim of 2-way interactions. Defaults to
            (1, 1, 8).
            init_stdev (float, optional): Standard deviation for initialization of
            2-way factors. Defaults to 0.1.
            iter_num (int, optional): number of iterations. Defaults to 100.
            load_model (str | None, optional): Filename with saved model to load.
            Defaults to None.
            meta (str | None, optional): Filename for meta (group) information about
            data set. Defaults to None.
            rlog (str | None): Filename to write iterative measurements to.
            Defaults to None.
            seed (int | None, optional): Random state seed. Defaults to None.
            verbosity (int | None, optional): How much info to output to internal
            command line.
        """
        super().__init__(
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

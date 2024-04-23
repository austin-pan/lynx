"""libFM task base classes."""


import os
import shutil
import time
from typing import Iterable, List, Tuple, Union

import lynx as lx
from lynx import libfm


class LibFMTask:
    """libFM task base."""

    def __init__(
        self,
        method: str,
        task: str,
        *,
        cache_size: Union[int, None] = None,
        dim: Tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        learn_rate: Union[float, Tuple[float, float, float], None] = None, # SGD only
        meta: Union[str, None] = None,
        regularizations: Union[float, Tuple[float, float, float], None] = None, # Only SGD and ALS
        rlog: Union[str, None] = None,
        seed: Union[int, None] = None,
        validation: Union[str, None] = None, # Only SGDA
        verbosity: Union[int, None] = None,

        # Only used by lynx, not by the libFM tool
        mat_dir: Union[str, None] = None
    ):
        """
        Args:
            method (str): Learning method ("sgd", "sgda", "als", "mcmc").
            task (str): "r"=regression, "c"=binary classification.
            cache_size (int | None, optional): Cache size for data storage (only
            applicable if data is in binary format). Defaults to None.
            dim (Tuple[int, int, int], optional): (k0,k1,k2): k0=use bias,
            k1=use 1-way interactions, k2=dim of 2-way interactions. Defaults to
            (1, 1, 8).
            init_stdev (float, optional): Standard deviation for initialization of
            2-way factors. Defaults to 0.1.
            iter_num (int, optional): number of iterations. Defaults to 100.
            learn_rate (float | None, optional): learn_rate for SGD. Defaults to
            None.
            meta (str | None, optional): Filename for meta (group) information about
            data set. Defaults to None.
            regular (int | Tuple[int, int, int] | None, optional): (r0,r1,r2) for
            SGD and ALS: r0=bias regularization, r1=1-way regularization, r2=2-way
            regularization. Defaults to None.
            rlog (str | None): Filename to write iterative measurements to.
            Defaults to None.
            seed (int | None, optional): Random state seed. Defaults to None.
            validation (str | None, optional): Filename for validation data (only
            for SGDA). Defaults to None.
            verbosity (int | None, optional): How much info to output to internal
            command line. Use `verbose` to print the output. Defaults to None.
            mat_dir (str | None, optional): Directory with libFM files. Defaults
            to creating a directory in `/tmp`. Only used by `lynx`, not passed to
            libFM.
            **kwargs: Additional flag-value pair to pass to libFM.
        """
        self.method = method
        self.task = task
        self.train_file = libfm.TRAIN_FILE
        self.test_file = libfm.TEST_FILE

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

        self.out = os.path.join(self.mat_dir, "predictions.txt")
        os.makedirs(self.mat_dir, exist_ok=True)

    def run(self, **kwargs) -> List[str]:
        """Run LibFM with specified flags and values."""
        libfm_args = {
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
        defined_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        libfm_args.update(defined_kwargs)
        return libfm.run(self.mat_dir, **libfm_args)

    def get_predictions(self) -> List[float]:
        """Read written predictions after FM has been fitted."""
        predictions = []
        with open(self.out, "r", encoding="utf-8") as f:
            predictions = f.readlines()
        return list(map(float, predictions))

    def flush(self) -> None:
        """Remove LibFM files."""
        shutil.rmtree(self.mat_dir)

class StatelessLibFMTask(LibFMTask):
    """libFM task without state base. Allows saving and loading of models."""

    def __init__(
        self,
        method: str,
        task: str,
        *,
        cache_size: Union[int, None] = None,
        dim: Tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        learn_rate: Union[float, Tuple[float, float, float], None] = None, # SGD only
        meta: Union[str, None] = None,
        regularizations: Union[float, Tuple[float, float, float], None] = None, # Only SGD and ALS
        rlog: Union[str, None] = None,
        seed: Union[int, None] = None,
        validation: Union[str, None] = None, # Only SGDA
        verbosity: Union[int, None] = None,

        mat_dir: Union[str, None] = None
    ):
        """
        Args:
            method (str): Learning method ("sgd", "sgda", "als", "mcmc").
            task (str): "r"=regression, "c"=binary classification.
            cache_size (int | None, optional): Cache size for data storage (only
            applicable if data is in binary format). Defaults to None.
            dim (Tuple[int, int, int], optional): (k0,k1,k2): k0=use bias,
            k1=use 1-way interactions, k2=dim of 2-way interactions. Defaults to
            (1, 1, 8).
            init_stdev (float, optional): Standard deviation for initialization of
            2-way factors. Defaults to 0.1.
            iter_num (int, optional): number of iterations. Defaults to 100.
            learn_rate (float | None, optional): learn_rate for SGD. Defaults to
            None.
            meta (str | None, optional): Filename for meta (group) information about
            data set. Defaults to None.
            regular (int | Tuple[int, int, int] | None, optional): (r0,r1,r2) for
            SGD and ALS: r0=bias regularization, r1=1-way regularization, r2=2-way
            regularization. Defaults to None.
            rlog (str | None): Filename to write iterative measurements to.
            Defaults to None.
            seed (int | None, optional): Random state seed. Defaults to None.
            validation (str | None, optional): Filename for validation data (only
            for SGDA). Defaults to None.
            verbosity (int | None, optional): How much info to output to internal
            command line. Use `verbose` to print the output. Defaults to None.
            mat_dir (str | None, optional): Directory with libFM files. Only used
            by `lynx`, not passed to libFM. Defaults to creating a directory in `/tmp`.
            **kwargs: Additional flag-value pair to pass to libFM.
        """
        super().__init__(
            method,
            task,
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
        y_train: Iterable[Union[float, int]],
        X_test: lx.Table,
        verbose: bool = False
    ) -> None:
        """
        Write libFM SVMLight files and create corresponding libFM binaries.

        Args:
            X_train (lx.Table): Train data.
            y_train (Iterable[Union[float, int]]): Train targets.
            X_test (lx.Table): Test data.
            verbose (bool, optional): Whether to print progress. Defaults to
            False.
        """
        raise NotImplementedError()

    def train(
        self,
        verbose: bool = False,
        no_output: bool = False
    ) -> List[str]:
        """
        Run libFM command for fitting and predicting FM.

        If `no_output` is set to True, then no predictions will be output which
        will cause `get_predictions` to fail.

        Args:
            verbose (bool, optional): Whether to print train progress. Train
            statistics should be ignored. Defaults to False.
            no_output (bool, optional): Whether to ignore reading files to get
            most accurate timings. Defaults to False.

        Returns:
            List[str]: Command line output.
        """
        raise NotImplementedError()

    def fit_predict(
        self,
        X_train: lx.Table,
        y_train: Iterable[Union[float, int]],
        X_test: lx.Table,
        verbose: bool = False
    ) -> List[float]:
        """
        Fit and predict FM. Convenience function for `write` then `train`.

        Do not use for timing. To get training time, run `write` and then time
        `train`. Afterwards, to get predictions, use `get_predictions`.

        Args:
            X_train (lx.Table): Train data.
            y_train (Iterable[Union[float, int]]): Train data targets.
            X_test (lx.Table): Test data.
            verbose (bool, optional): Whether to print train progress. Train
            statistics should be ignored. Defaults to False.

        Returns:
            List[float]: Test predictions.
        """
        self.write(X_train, y_train, X_test, verbose=verbose)
        self.train(verbose=verbose)
        return self.get_predictions()

class StatefulLibFMTask(LibFMTask):
    """libFM task with state base. Allows saving and loading of models."""

    def __init__(
        self,
        method: str,
        task: str,
        *,
        cache_size: Union[int, None] = None,
        dim: Tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        learn_rate: Union[float, Tuple[float, float, float], None] = None, # SGD only
        load_model: Union[str, None] = None,
        meta: Union[str, None] = None,
        regularizations: Union[float, Tuple[float, float, float], None] = None, # Only SGD and ALS
        rlog: Union[str, None] = None,
        seed: Union[int, None] = None,
        validation: Union[str, None] = None, # Only SGDA
        verbosity: Union[int, None] = None,

        mat_dir: Union[str, None] = None
    ):
        """
        Args:
            method (str): Learning method ("sgd", "sgda", "als", "mcmc").
            task (str): "r"=regression, "c"=binary classification.
            cache_size (int | None, optional): Cache size for data storage (only
            applicable if data is in binary format). Defaults to None.
            dim (Tuple[int, int, int], optional): (k0,k1,k2): k0=use bias,
            k1=use 1-way interactions, k2=dim of 2-way interactions. Defaults to
            (1, 1, 8).
            init_stdev (float, optional): Standard deviation for initialization of
            2-way factors. Defaults to 0.1.
            iter_num (int, optional): number of iterations. Defaults to 100.
            learn_rate (float | None, optional): learn_rate for SGD. Defaults to
            None.
            load_model (str | None, optional): Filename with saved model to load.
            Defaults to None.
            meta (str | None, optional): Filename for meta (group) information about
            data set. Defaults to None.
            regular (int | Tuple[int, int, int] | None, optional): (r0,r1,r2) for
            SGD and ALS: r0=bias regularization, r1=1-way regularization, r2=2-way
            regularization. Defaults to None.
            rlog (str | None): Filename to write iterative measurements to.
            Defaults to None.
            seed (int | None, optional): Random state seed. Defaults to None.
            validation (str | None, optional): Filename for validation data (only
            for SGDA). Defaults to None.
            verbosity (int | None, optional): How much info to output to internal
            command line.
            mat_dir (str | None, optional): Directory with libFM files. Only used
            by `lynx`, not passed to libFM. Defaults to creating a directory in `/tmp`.
            **kwargs: Additional flag-value pair to pass to libFM.
        """
        super().__init__(
            method,
            task,
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
        y_train: Iterable[Union[float, int]],
        verbose: bool = False
    ) -> None:
        """
        Write libFM SVMLight files and create corresponding libFM binaries.

        Args:
            X_train (lx.Table): Train data.
            y_train (Iterable[float | int]): Train targets.
            verbose (bool, optional): Whether to print progress. Defaults to
            False.
        """
        raise NotImplementedError()

    def train(
        self,
        verbose: bool = False,
        no_output: bool = False
    ) -> List[str]:
        """
        Run libFM command for fitting FM.

        If `no_output` is set to True, then no model state will be saved which
        will cause `predict` to fail.

        Args:
            verbose (bool, optional): Whether to print train progress. Train
            statistics should be ignored. Defaults to False.
            no_output (bool, optional): Whether to ignore reading files to get
            most accurate timings. Defaults to False.

        Returns:
            List[str]: Command line output.
        """
        raise NotImplementedError()

    def fit(
        self,
        X_train: lx.Table,
        y_train: Iterable[Union[float, int]],
        verbose: bool = False
    ) -> None:
        """
        Fit FM. Convenience function for `write` then `train`.

        Do not use for timing. To get training time, run `write` and then time
        `train`.

        Args:
            X_train (lx.Table): Train data.
            y_train (Iterable[float | int]): Train data targets.
            verbose (bool, optional): Whether to print train progress. Train
            statistics should be ignored. Defaults to False.
        """
        self.write(X_train, y_train, verbose=verbose)
        self.train(verbose=verbose)

    def predict(self, X_test: lx.Table, verbose: bool = False) -> List[float]:
        """
        Returns libFM task predictions for the rows in the provided test data.

        NOTE: `predict` is not reproducible unless a seed has been used because
        libFM requires 1 additional iteration of training to produce
        predictions.

        Args:
            X_test (lx.Table): Test data.
            verbose (bool, optional): Whether to print prediction progress.
            Defaults to False.

        Returns:
            List[float]: Test predictions.
        """
        raise NotImplementedError()

    def save_model(self, path: str) -> None:
        """
        Save FM model to a specified filepath.

        Args:
            path (str): Filepath to save model.
        """
        model_path = os.path.join(self.mat_dir, self.model_name)
        shutil.copy(model_path, path)

class DenseStatefulLibFMTask(StatefulLibFMTask):
    """Dense libFM task that allows saving and loading of models."""

    def __init__(
        self,
        method: str,
        task: str,
        *,
        cache_size: Union[int, None] = None,
        dim: Tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        learn_rate: Union[float, Tuple[float, float, float], None] = None, # SGD only
        load_model: Union[str, None] = None,
        meta: Union[str, None] = None,
        regularizations: Union[float, Tuple[float, float, float], None] = None, # Only SGD and ALS
        rlog: Union[str, None] = None,
        seed: Union[int, None] = None,
        validation: Union[str, None] = None, # Only SGDA
        verbosity: Union[int, None] = None,

        mat_dir: Union[str, None] = None
    ):
        """
        Args:
            method (str): Learning method ("sgd", "sgda", "als", "mcmc").
            task (str): "r"=regression, "c"=binary classification.
            cache_size (int | None, optional): Cache size for data storage (only
            applicable if data is in binary format). Defaults to None.
            dim (Tuple[int, int, int], optional): (k0,k1,k2): k0=use bias,
            k1=use 1-way interactions, k2=dim of 2-way interactions. Defaults to
            (1, 1, 8).
            init_stdev (float, optional): Standard deviation for initialization of
            2-way factors. Defaults to 0.1.
            iter_num (int, optional): number of iterations. Defaults to 100.
            learn_rate (float | None, optional): learn_rate for SGD. Defaults to
            None.
            load_model (str | None, optional): Filename with saved model to load.
            Defaults to None.
            meta (str | None, optional): Filename for meta (group) information about
            data set. Defaults to None.
            out (str | None, optional): Filename to write prediction output to.
            Defaults to None.
            regular (int | Tuple[int, int, int] | None, optional): (r0,r1,r2) for
            SGD and ALS: r0=bias regularization, r1=1-way regularization, r2=2-way
            regularization. Defaults to None.
            rlog (str | None): Filename to write iterative measurements to.
            Defaults to None.
            seed (int | None, optional): Random state seed. Defaults to None.
            verbosity (int | None, optional): How much info to output to internal
            command line.
            mat_dir (str | None, optional): Directory with libFM files. Only used
            by `lynx`, not passed to libFM. Defaults to creating a directory in `/tmp`.
            **kwargs: Additional flag-value pair to pass to libFM.
        """
        super().__init__(
            method,
            task,
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
        y_train: Iterable[Union[float, int]],
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

    def train(self, verbose: bool = False, no_output: bool = False) -> List[str]:
        save_model = None if no_output else self.model_name
        return self.run(save_model=save_model, seed=self.seed, verbose=verbose)

    def predict(self, X_test: lx.Table, verbose: bool = False) -> List[float]:
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
            out=self.out,
            verbose=verbose
        )
        return self.get_predictions()

"""Dense structure MCMC tasks."""

import os
from typing import Iterable, List, Tuple, Union

import lynx as lx
from lynx import libfm
from lynx.libfm import tasks


class MCMCTask(tasks.StatelessLibFMTask):
    """Monte-Carlo Markov Chain task."""

    def __init__(
        self,
        task: str,
        *,
        cache_size: Union[int, None] = None,
        dim: Tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        meta: Union[str, None] = None,
        rlog: Union[str, None] = None,
        seed: Union[int, None] = None,
        verbosity: Union[int, None] = None
    ):
        """
        Args:
            task (str): "r"=regression, "c"=binary classification.
            cache_size (int | None, optional): Cache size for data storage (only
            applicable if data is in binary format). Defaults to None.
            dim (Tuple[int, int, int], optional): (k0,k1,k2): k0=use bias,
            k1=use 1-way interactions, k2=dim of 2-way interactions. Defaults to
            (1, 1, 8).
            init_stdev (float, optional): Standard deviation for initialization of
            2-way factors. Defaults to 0.1.
            iter_num (int, optional): number of iterations. Defaults to 100.
            meta (str | None, optional): Filename for meta (group) information about
            data set. Defaults to None.
            rlog (str | None): Filename to write iterative measurements to.
            Defaults to None.
            seed (int | None, optional): Random state seed. Defaults to None.
            verbosity (int | None, optional): How much info to output to internal
            command line.
        """
        super().__init__(
            method="mcmc",
            task=task,
            train_file="_libfm_train",
            test_file="_libfm_test",
            cache_size=cache_size,
            dim=dim,
            init_stdev=init_stdev,
            iter_num=iter_num,
            meta=meta,
            rlog=rlog,
            seed=seed,
            verbosity=verbosity
        )

    def write(
        self,
        X_train: lx.Table,
        y_train: Iterable[Union[float, int]],
        X_test: lx.Table,
        verbose: bool = False
    ) -> None:
        # Write train libfm files
        libfm_train_file = f"{self.train_file}.libfm"
        train_target_path = os.path.join(self.mat_dir, libfm_train_file)
        lx.write.libfm.dense(X_train, y_train, train_target_path)
        libfm.create_dense_binaries(self.mat_dir, libfm_train_file, verbose=verbose)
        # Write test libfm files
        libfm_test_file = f"{self.test_file}.libfm"
        test_target_path = os.path.join(self.mat_dir, libfm_test_file)
        # Dummy y_test
        y_test = list(range(X_test.height))
        lx.write.libfm.dense(X_test, y_test, test_target_path)
        libfm.create_dense_binaries(self.mat_dir, libfm_test_file, verbose=verbose)

    def train(
        self,
        verbose: bool = False,
        no_output: bool = False
    ) -> List[str]:
        out = None if no_output else self.out
        return self.run(seed=self.seed, out=out, verbose=verbose)

class FMRegression(MCMCTask):
    """Monte-Carlo Markov Chain regression task."""

    def __init__(
        self,
        *,
        cache_size: Union[int, None] = None,
        dim: Tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        meta: Union[str, None] = None,
        rlog: Union[str, None] = None,
        seed: Union[int, None] = None,
        verbosity: Union[int, None] = None
    ):
        """
        Args:
            cache_size (int | None, optional): Cache size for data storage (only
            applicable if data is in binary format). Defaults to None.
            dim (Tuple[int, int, int], optional): (k0,k1,k2): k0=use bias,
            k1=use 1-way interactions, k2=dim of 2-way interactions. Defaults to
            (1, 1, 8).
            init_stdev (float, optional): Standard deviation for initialization of
            2-way factors. Defaults to 0.1.
            iter_num (int, optional): number of iterations. Defaults to 100.
            meta (str | None, optional): Filename for meta (group) information about
            data set. Defaults to None.
            rlog (str | None): Filename to write iterative measurements to.
            Defaults to None.
            seed (int | None, optional): Random state seed. Defaults to None.
            verbosity (int | None, optional): How much info to output to internal
            command line.
        """
        super().__init__(
            task = "r",
            cache_size=cache_size,
            dim=dim,
            init_stdev=init_stdev,
            iter_num=iter_num,
            meta=meta,
            rlog=rlog,
            seed=seed,
            verbosity=verbosity
        )

class FMClassification(MCMCTask):
    """Monte-Carlo Markov Chain classification task."""

    def __init__(
        self,
        *,
        cache_size: Union[int, None] = None,
        dim: Tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        meta: Union[str, None] = None,
        rlog: Union[str, None] = None,
        seed: Union[int, None] = None,
        verbosity: Union[int, None] = None
    ):
        """
        Args:
            cache_size (int | None, optional): Cache size for data storage (only
            applicable if data is in binary format). Defaults to None.
            dim (Tuple[int, int, int], optional): (k0,k1,k2): k0=use bias,
            k1=use 1-way interactions, k2=dim of 2-way interactions. Defaults to
            (1, 1, 8).
            init_stdev (float, optional): Standard deviation for initialization of
            2-way factors. Defaults to 0.1.
            iter_num (int, optional): number of iterations. Defaults to 100.
            meta (str | None, optional): Filename for meta (group) information about
            data set. Defaults to None.
            rlog (str | None): Filename to write iterative measurements to.
            Defaults to None.
            seed (int | None, optional): Random state seed. Defaults to None.
            verbosity (int | None, optional): How much info to output to internal
            command line.
        """
        super().__init__(
            task = "c",
            cache_size=cache_size,
            dim=dim,
            init_stdev=init_stdev,
            iter_num=iter_num,
            meta=meta,
            rlog=rlog,
            seed=seed,
            verbosity=verbosity
        )

"""Block structure MCMC tasks."""

import os
import time
from typing import Iterable

import lynx as lx
from lynx import libfm
from lynx.libfm import tasks


class MCMCTask(tasks.StatelessLibFMTask):

    def __init__(
        self,
        task: str,
        *,
        cache_size: int | None = None,
        dim: tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        meta: str | None = None,
        rlog: str | None = None,
        seed: int | None = None,
        verbosity: int | None = None
    ):
        super().__init__(
            method="mcmc",
            task=task,
            train_file="train",
            test_file = "test",
            cache_size=cache_size,
            dim=dim,
            init_stdev=init_stdev,
            iter_num=iter_num,
            meta=meta,
            rlog=rlog,
            seed=seed,
            verbosity=verbosity
        )
        self.relation = []

    def write(
        self,
        X_train: lx.Table,
        y_train: Iterable[float | int],
        X_test: lx.Table,
        verbose: bool = False
    ) -> None:
        # Write train libfm files
        lx.write.libfm.bs(X_train, y_train, self.mat_dir, phase="train")
        # Write test libfm files
        # Dummy y_test
        y_test = list(range(X_test.height))
        lx.write.libfm.bs(X_test, y_test, self.mat_dir, phase="test",
                          ignore_block=True)

        block_names = X_train.block_names
        self.relation = block_names
        libfm.create_bs_binaries(self.mat_dir, block_names, verbose=verbose)

    def train(
        self,
        outpath: str,
        verbose: bool = False,
        time_only: bool = False
    ) -> float:
        start_time = time.perf_counter()
        if not time_only:
            self.run(
                relation=self.relation,
                seed=self.seed,
                out=outpath,
                verbose=verbose
            )
        else:
            self.run(
                relation=self.relation,
                seed=self.seed,
                verbose=verbose
            )
        end_time = time.perf_counter()
        return end_time - start_time

    def get_predictions(self, outpath: str) -> list[float]:
        predictions = []
        with open(outpath, "r", encoding="utf-8") as f:
            predictions = f.readlines()
        return list(map(float, predictions))

    def fit_predict(
        self,
        X_train: lx.Table,
        y_train: Iterable[float | int],
        X_test: lx.Table,
        verbose: bool = False
    ) -> list[float]:
        """Stateless so can only do both fit and predict together."""
        self.write(X_train, y_train, X_test, verbose=verbose)
        outpath = os.path.join(self.mat_dir, "predictions.txt")
        self.train(outpath, verbose=verbose)

        return self.get_predictions(outpath)

class FMRegression(MCMCTask):

    def __init__(
        self,
        *,
        cache_size: int | None = None,
        dim: tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        meta: str | None = None,
        rlog: str | None = None,
        seed: int | None = None,
        verbosity: int | None = None
    ):
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

    def __init__(
        self,
        *,
        cache_size: int | None = None,
        dim: tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        meta: str | None = None,
        rlog: str | None = None,
        seed: int | None = None,
        verbosity: int | None = None
    ):
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

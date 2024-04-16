import os
import time
from typing import Iterable

import lynx as lx
from lynx import libfm
from lynx.libfm import tasks

class BlockStatefulLibFMTask(tasks.StatefulLibFMTask):

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
        learn_rate: float | None = None, # Only SGD
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
        self.relation = []

    def write(
        self,
        X_train: lx.Table,
        y_train: Iterable[float | int],
        verbose: bool = False
    ) -> None:
        # Write train libfm files
        lx.write.libfm.bs(X_train, y_train, self.mat_dir, phase="train")
        # Write empty test libfm files
        lx.write.libfm.bs(X_train, y_train, self.mat_dir, phase="test",
                          ignore_block=True, empty_indices=True, empty_targets=True)
        self.relation = X_train.block_names
        libfm.create_bs_binaries(self.mat_dir, X_train.block_names, verbose=verbose)

    def train(self, verbose: bool = False, time_only: bool = False) -> float:
        start_time = time.perf_counter()
        if not time_only:
            self.run(
                relation=self.relation,
                save_model=self.model_name,
                seed=self.seed,
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
        assert X_test.block_names == self.relation, "Tables must have the same schema (block names)"

        outpath = os.path.join(self.mat_dir, "predictions.txt")
        # Dummy y_test
        y_test = [0] * X_test.height
        lx.write.libfm.bs(X_test, y_test, self.mat_dir, phase="test", ignore_block=True)
        self.run(
            relation=self.relation,
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

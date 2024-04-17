"""Blocks structure task base with state."""


from typing import Iterable, List, Tuple, Union

import lynx as lx
from lynx import libfm
from lynx.libfm import tasks

class BlockStatefulLibFMTask(tasks.StatefulLibFMTask):
    """Block structure task with state."""

    def __init__(
        self,
        method: str,
        task: str,
        train_file: str,
        test_file: str,
        *,
        cache_size: Union[int, None] = None,
        dim: Tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        learn_rate: Union[float, None] = None, # Only SGD
        load_model: Union[str, None] = None,
        meta: Union[str, None] = None,
        regularizations: Union[float, Tuple[float, float, float], None] = None, # Only SGD and ALS
        rlog: Union[str, None] = None,
        seed: Union[int, None] = None,
        validation: Union[str, None] = None, # Only SGDA
        verbosity: Union[int, None] = None,

        mat_dir: Union[str, None] = None
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
        y_train: Iterable[Union[float, int]],
        verbose: bool = False
    ) -> None:
        # Write train libfm files
        lx.write.libfm.bs(X_train, y_train, self.mat_dir, self.train_file, phase="train")
        # Write empty test libfm files
        lx.write.libfm.bs(X_train, y_train, self.mat_dir, self.test_file, phase="test",
                          ignore_block=True, empty_indices=True, empty_targets=True)
        self.relation = X_train.block_names
        libfm.create_bs_binaries(self.mat_dir, X_train.block_names, verbose=verbose)

    def train(self, verbose: bool = False, no_output: bool = False) -> List[str]:
        save_model = None if no_output else self.model_name
        return self.run(
            relation=self.relation,
            save_model=save_model,
            seed=self.seed,
            verbose=verbose
        )

    def predict(self, X_test: lx.Table, verbose: bool = False) -> List[float]:
        assert X_test.block_names == self.relation, "Tables must have the same schema (block names)"

        # Dummy y_test
        y_test = [0] * X_test.height
        lx.write.libfm.bs(X_test, y_test, self.mat_dir, self.test_file, phase="test",
                          ignore_block=True)
        self.run(
            relation=self.relation,
            load_model=self.model_name,
            seed=self.seed,
            iter_num=1,
            out=self.out,
            verbose=verbose
        )

        predictions = []
        with open(self.out, "r", encoding="utf-8") as f:
            predictions = f.readlines()
        return list(map(float, predictions))

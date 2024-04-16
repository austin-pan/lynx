"""Block structure ALS tasks."""

from lynx.libfm import tasks


class SGDTask(tasks.DenseStatefulLibFMTask):

    def __init__(
        self,
        task: str,
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
            method="sgd",
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
            verbosity=verbosity
        )

class FMRegression(SGDTask):

    def __init__(
        self,
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

class FMClassification(SGDTask):

    def __init__(
        self,
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

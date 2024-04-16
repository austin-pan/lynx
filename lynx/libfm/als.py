"""Dense ALS tasks."""

from lynx.libfm import tasks


class ALSTask(tasks.DenseStatefulLibFMTask):

    def __init__(
        self,
        task: str,
        *,
        cache_size: int | None = None,
        dim: tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        load_model: str | None = None,
        meta: str | None = None,
        regularizations: float | tuple[float, float, float] | None = None, # Only SGD and ALS
        rlog: str | None = None,
        seed: int | None = None,
        verbosity: int | None = None
    ):
        super().__init__(
            method="als",
            task=task,
            train_file="train",
            test_file="test",
            cache_size=cache_size,
            dim=dim,
            init_stdev=init_stdev,
            iter_num=iter_num,
            load_model=load_model,
            meta=meta,
            regularizations=regularizations,
            rlog=rlog,
            seed=seed,
            verbosity=verbosity
        )

class FMRegression(ALSTask):

    def __init__(
        self,
        *,
        cache_size: int | None = None,
        dim: tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        load_model: str | None = None,
        meta: str | None = None,
        regularizations: float | tuple[float, float, float] | None = None, # Only SGD and ALS
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
            load_model=load_model,
            meta=meta,
            regularizations=regularizations,
            rlog=rlog,
            seed=seed,
            verbosity=verbosity
        )

class FMClassification(ALSTask):

    def __init__(
        self,
        *,
        cache_size: int | None = None,
        dim: tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        load_model: str | None = None,
        meta: str | None = None,
        regularizations: float | tuple[float, float, float] | None = None, # Only SGD and ALS
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
            load_model=load_model,
            meta=meta,
            regularizations=regularizations,
            rlog=rlog,
            seed=seed,
            verbosity=verbosity
        )

"""Dense structure ALS tasks."""

from typing import Tuple, Union
from lynx.libfm import tasks


class ALSTask(tasks.DenseStatefulLibFMTask):
    """Alternating Least Squares task."""

    def __init__(
        self,
        task: str,
        *,
        cache_size: Union[int, None] = None,
        dim: Tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        load_model: Union[str, None] = None,
        meta: Union[str, None] = None,
        regularizations: Union[float, Tuple[float, float, float], None] = None, # Only SGD and ALS
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
            load_model (str | None, optional): Filename with saved model to load.
            Defaults to None.
            meta (str | None, optional): Filename for meta (group) information about
            data set. Defaults to None.
            regularizations (int | Tuple[int, int, int] | None, optional): (r0,r1,r2) for
            SGD and ALS: r0=bias regularization, r1=1-way regularization, r2=2-way
            regularization. Defaults to None.
            rlog (str | None): Filename to write iterative measurements to.
            Defaults to None.
            seed (int | None, optional): Random state seed. Defaults to None.
            verbosity (int | None, optional): How much info to output to internal
            command line.
        """
        super().__init__(
            method="als",
            task=task,
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
    """Alternating Least Squares regression task."""

    def __init__(
        self,
        *,
        cache_size: Union[int, None] = None,
        dim: Tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        load_model: Union[str, None] = None,
        meta: Union[str, None] = None,
        regularizations: Union[float, Tuple[float, float, float], None] = None, # Only SGD and ALS
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
            load_model (str | None, optional): Filename with saved model to load.
            Defaults to None.
            meta (str | None, optional): Filename for meta (group) information about
            data set. Defaults to None.
            regularizations (int | Tuple[int, int, int] | None, optional): (r0,r1,r2) for
            SGD and ALS: r0=bias regularization, r1=1-way regularization, r2=2-way
            regularization. Defaults to None.
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
            load_model=load_model,
            meta=meta,
            regularizations=regularizations,
            rlog=rlog,
            seed=seed,
            verbosity=verbosity
        )

class FMClassification(ALSTask):
    """Alternating Least Squares classification task."""

    def __init__(
        self,
        *,
        cache_size: Union[int, None] = None,
        dim: Tuple[int, int, int] = (1, 1, 8),
        init_stdev: float = 0.1,
        iter_num: int = 100,
        load_model: Union[str, None] = None,
        meta: Union[str, None] = None,
        regularizations: Union[float, Tuple[float, float, float], None] = None, # Only SGD and ALS
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
            load_model (str | None, optional): Filename with saved model to load.
            Defaults to None.
            meta (str | None, optional): Filename for meta (group) information about
            data set. Defaults to None.
            regularizations (int | Tuple[int, int, int] | None, optional): (r0,r1,r2) for
            SGD and ALS: r0=bias regularization, r1=1-way regularization, r2=2-way
            regularization. Defaults to None.
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
            load_model=load_model,
            meta=meta,
            regularizations=regularizations,
            rlog=rlog,
            seed=seed,
            verbosity=verbosity
        )

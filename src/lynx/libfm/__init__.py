"""LibFM tools."""

import os
import subprocess
from typing import List, Tuple, Union


LIBFM_HOME = os.environ["LIBFM_HOME"]
APPS = {
    tool: f"{LIBFM_HOME}/bin/{tool}"
    for tool in ["libFM", "transpose", "convert"]
}

TRAIN_FILE = "_libfm_train"
TEST_FILE = "_libfm_test"
VALIDATION_FILE = "_libfm_validation"

def run(
    mat_dir: str,
    *,
    method: str,
    task: str,
    train: str,
    test: str,
    cache_size: Union[int, None] = None,
    dim: Tuple[int, int, int] = (1, 1, 8),
    init_stdev: float = 0.1,
    iter_num: int = 100,
    learn_rate: Union[float, None] = None, # Only SGD
    load_model: Union[str, None] = None,
    meta: Union[str, None] = None,
    out: Union[str, None] = None,
    regular: Union[int, Tuple[int, int, int], None] = None, # Only SGD and ALS
    relation: Union[List[str], None] = None, # Only block structure
    rlog: Union[str, None] = None,
    save_model: Union[str, None] = None,
    seed: Union[int, None] = None,
    validation: Union[str, None] = None, # Only SGDA
    verbosity: Union[int, None] = None,
    verbose: bool = False,
    **kwargs
) -> List[str]:
    """
    Run LibFM on the files in the provided directory.

    Args:
        mat_dir (str): Directory with libFM files.
        method (str): Learning method ("sgd", "sgda", "als", "mcmc").
        task (str): "r"=regression, "c"=binary classification.
        train (str): Filename for training data.
        test (str): Filename for test data.
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
        load_model (str | None, optional): Filename with saved model to load
        before running. Defaults to None.
        meta (str | None, optional): Filename for meta (group) information about
        data set. Defaults to None.
        out (str | None, optional): Filename to write prediction output to.
        Defaults to None.
        regular (int | Tuple[int, int, int] | None, optional): (r0,r1,r2) for
        SGD and ALS: r0=bias regularization, r1=1-way regularization, r2=2-way
        regularization. Defaults to None.
        relation (List[str] | None, optional): Names of relation files. Block
        structure only. Defaults to None.
        rlog (str | None): Filename to write iterative measurements to. Defaults to None.
        save_model (str | None, optional): Filename to save model weights to.
        Only supported by ALS, SGD, and SGDA. Defaults to None.
        seed (int | None, optional): Random state seed. Defaults to None.
        validation (str | None, optional): Filename for validation data (only
        for SGDA). Defaults to None.
        verbosity (int | None, optional): How much info to output to internal
        command line. Use `verbose` to print the output. Defaults to None.
        verbose (bool, optional): Whether to display libFM command line output.
        Defaults to False.
        **kwargs: Additional flag-value pair to pass to libFM.

    Return:
        List[str]: Command line output
    """
    args = {
        "method": method,
        "task": task,
        "train": train,
        "test": test,
        "dim": ",".join(str(d) for d in dim),
        "iter": iter_num,
        "init_stdev": init_stdev
    }
    if cache_size is not None:
        args["cache_size"] = cache_size
    if learn_rate is not None: # Only SGD
        args["learn_rate"] = learn_rate
    if load_model is not None:
        args["load_model"] = load_model
    if meta is not None:
        args["meta"] = meta
    if out is not None:
        args["out"] = out
    if regular is not None: # Only SGD and ALS
        reg_value = regular
        if isinstance(reg_value, Tuple):
            reg_value = ",".join(str(r) for r in reg_value)
        args["regular"] = reg_value
    if relation is not None: # Block Structure mode
        args["relation"] = ",".join(relation)
    if rlog is not None:
        args["rlog"] = rlog
    if save_model is not None:
        args["save_model"] = save_model
    if seed is not None:
        args["seed"] = seed
    if validation is not None: # Only SGDA
        args["validation"] = validation
    if verbosity is not None:
        args["verbosity"] = verbosity
    args.update(kwargs)

    cmd_args = []
    for arg, val in args.items():
        if val is not None:
            cmd_args += [f"-{arg}", val]
    cmd_args = list(map(str, cmd_args))

    # Change directory to directory with matrices to speed up LibFM file reading
    cwd = os.getcwd()
    os.chdir(mat_dir)

    cmd = [APPS["libFM"]] + cmd_args
    if verbose:
        print(" ".join(cmd))

    cmd_output = execute(cmd, verbose=verbose)
    os.chdir(cwd)
    return cmd_output

def convert(mat_filename: str, *, verbose: bool = False):
    """Converts LibFM text file to binary file."""
    mat_name = mat_filename.split(".")[0]
    if verbose:
        print(f"Converting {mat_name}")
    subprocess.run(
        (
            f"{APPS['convert']} --ifile {mat_name}.libfm --ofilex {mat_name}.x "
            f"--ofiley {mat_name}.y"
            .split(" ")
        ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

def transpose(mat_filename: str, *, verbose: bool = False):
    """Transposes LibFM binary file."""
    mat_name = mat_filename.split(".")[0]
    if verbose:
        print(f"Transposing {mat_name}")
    subprocess.run(
        (
            f"{APPS['transpose']} --ifile {mat_name}.x --ofile {mat_name}.xt"
            .split(" ")
        ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )


def execute(cmd, *, verbose: bool = False) -> List[str]:
    """Run a command in the command line and return stdout."""
    def exec_cmd():
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True) as popen:
            if popen.stdout is not None:
                for line in iter(popen.stdout.readline, ""):
                    yield line
            return_code = popen.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, cmd)

    stdout_lines = []
    for stdout_line in exec_cmd():
        if verbose:
            print(stdout_line.rstrip())
        stdout_lines.append(stdout_line)

    return stdout_lines

def create_dense_binaries(
    mat_dir: str,
    path: str,
    *,
    verbose: bool = False
) -> None:
    train_path = os.path.join(mat_dir, path)
    convert(train_path, verbose=verbose)
    transpose(train_path, verbose=verbose)

def create_bs_binaries(
    mat_dir: str,
    block_names: List[str],
    *,
    verbose: bool = False
) -> None:
    for block_name in block_names:
        block_path = os.path.join(mat_dir, f"{block_name}.libfm")
        convert(block_path, verbose=verbose)
        transpose(block_path, verbose=verbose)

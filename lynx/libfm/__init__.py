"""LibFM tools."""

from __future__ import annotations

import os
import subprocess


LIBFM_HOME = os.environ["LIBFM_HOME"]
APPS = {
    tool: f"{LIBFM_HOME}/bin/{tool}"
    for tool in ["libFM", "transpose", "convert"]
}

def run(
    mat_dir: str,
    *,
    method: str,
    task: str,
    train: str,
    test: str,
    cache_size: int | None = None,
    dim: tuple[int, int, int] = (1, 1, 8),
    init_stdev: float = 0.1,
    iter_num: int = 100,
    learn_rate: float | None = None, # Only SGD
    load_model: str | None = None,
    meta: str | None = None,
    out: str | None = None,
    regular: int | tuple[int, int, int] | None = None, # Only SGD and ALS
    relation: list[str] | None = None, # Only block structure
    rlog: str | None = None,
    save_model: str | None = None,
    seed: int | None = None,
    validation: str | None = None, # Only SGDA
    verbosity: int | None = None,
    verbose: bool = False,
    **kwargs
) -> None:
    """
    Run LibFM on the files in the provided directory.

    Args:
        mat_dir (str): Directory with LibFM files.
        train (str): Filename for train data.
        test (str): Filename for test file.
        task (str, optional): r=regression, c=binary classification. Defaults to
        "r".
        dim (tuple[int, int, int], optional): (k0,k1,k2): k0=use bias,
        k1=use 1-way interactions, k2=dim of 2-way interactions. Defaults to
        (1, 1, 8).
        num_iter (int, optional): Number of iterations. Defaults to 100.
        method (str, optional): Learning method (SGD, SGDA, ALS, MCMC). Defaults
        to "mcmc".
        init_stdev (float, optional): stdev for initialization of 2-way factors.
        Defaults to 0.1.
        relation (list[str] | None, optional): Block Structure: filenames for
        the relations.  Defaults to None.
        verbose (bool, optional): Whether to print command line output. Defaults
        to False.
        **kwargs: Key-value pair that get passed to libFM as flag-value pairs,
        e.g. `-key value`

    Returns:
        tuple[list[float], float]: Test accuracy of each iteration and total
        runtime.
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
        if isinstance(reg_value, tuple):
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

    execute(cmd, verbose=verbose)
    os.chdir(cwd)

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


def execute(cmd, *, verbose: bool = False) -> list[str]:
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
    block_names: list[str],
    *,
    verbose: bool = False
) -> None:
    for block_name in block_names:
        block_path = os.path.join(mat_dir, f"{block_name}.libfm")
        convert(block_path, verbose=verbose)
        transpose(block_path, verbose=verbose)

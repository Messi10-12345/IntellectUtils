# intellect_utils/blas_utils.py

import os

def set_blas_threads(n: int):
    """
    Set number of threads used by various BLAS libraries.
    Call this before importing NumPy, SciPy, etc.

    Args:
        n (int): Number of threads to use.
    """
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    os.environ["BLIS_NUM_THREADS"] = str(n)

def show_blas_threads():
    """
    Return current thread-related environment settings.
    """
    keys = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ]
    return {key: os.environ.get(key, "default") for key in keys}
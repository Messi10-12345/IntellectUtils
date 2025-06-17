# intellect_utils/mkl_utils.py

import os
import numpy as np
import platform
import io
import sys

def is_mkl_available():
    """
    Check if MKL is being used by NumPy by inspecting numpy config info.
    Returns True if MKL is detected, else False.
    """
    buf = io.StringIO()
    # Capture the output of np.__config__.show()
    sys_stdout = sys.stdout
    sys.stdout = buf
    try:
        np.__config__.show()
    finally:
        sys.stdout = sys_stdout
    config_str = buf.getvalue().lower()
    return 'mkl' in config_str

def get_mkl_num_threads():
    """
    Get the number of MKL threads from environment variables.
    Returns int, default 1 if not set.
    """
    # MKL can use these env vars to control threads
    mkl_threads = os.environ.get('MKL_NUM_THREADS')
    omp_threads = os.environ.get('OMP_NUM_THREADS')

    # Prefer MKL_NUM_THREADS if set, else fallback to OMP_NUM_THREADS
    if mkl_threads is not None:
        return int(mkl_threads)
    elif omp_threads is not None:
        return int(omp_threads)
    else:
        return 1

def set_mkl_num_threads(n):
    """
    Set number of MKL threads by setting environment variables.
    Note: Should be set before importing numpy for best effect.
    """
    os.environ['MKL_NUM_THREADS'] = str(n)
    os.environ['OMP_NUM_THREADS'] = str(n)

def get_cpu_info():
    """
    Return basic CPU info to detect Intel CPUs.
    """
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'intel_cpu': 'intel' in platform.processor().lower(),
    }
    return info

def recommend_mkl_settings():
    """
    Return recommended environment variable settings for MKL tuning.
    """
    cpu_cores = os.cpu_count() or 1
    return {
        'MKL_NUM_THREADS': str(cpu_cores),
        'OMP_NUM_THREADS': str(cpu_cores),
        'MKL_DYNAMIC': 'FALSE',
        'OMP_DYNAMIC': 'FALSE',
    }
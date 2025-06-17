# intellect_utils/__init__.py

"""
IntellectUtils: Accelerated CPU Utilities for AI Workloads

Provides easy access to optimized computation tools using:
- Numba for JIT acceleration
- Joblib & concurrent.futures for parallelism
- Dask for task graphs
- Intel MKL / OpenBLAS tuning
- ONNX Runtime for efficient model inference
- PyTorch / TensorFlow CPU tuning
- Ray for distributed processing
- BLAS environment tuning

Version: Beta
"""

__version__ = "0.1.0-beta"

# Import submodules for easier access
from .jit_utils import jit_compile, vectorize_function
from .parallel_utils import run_parallel_threadpool, run_parallel_processpool
from .dask_utils import dask_parallel, start_dask_cluster
from .mkl_utils import mkl_get_version, mkl_set_num_threads
from .onnx_utils import run_onnx_model, load_onnx_model
from .torch_tf_utils import (
    set_torch_num_threads,
    get_torch_num_threads,
    set_tf_num_threads,
    get_tf_num_threads,
    torch_cpu_info,
    tf_cpu_info,
)
from .ray_utils import init_ray, shutdown_ray, run_parallel as ray_run_parallel, available_resources
from .blas_utils import set_blas_threads, show_blas_threads

# Optional: print a startup message
def startup_message():
    import platform
    print(f"[IntellectUtils v{__version__}] CPU Optimization Initialized ({platform.processor()})")

# Automatically show message on import
startup_message()
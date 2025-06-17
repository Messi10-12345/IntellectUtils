# intellect_utils/parallel.py

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

import multiprocessing
from tqdm import tqdm

def get_num_cores():
    """Return number of CPU cores available."""
    try:
        return multiprocessing.cpu_count()
    except NotImplementedError:
        return 1

def parallel_map(func, iterable, n_jobs=None, use_threads=False, show_progress=False, batch_size=None):
    """
    Run func over iterable in parallel using joblib.

    Args:
        func (callable): Function to apply.
        iterable (iterable): Input data iterable.
        n_jobs (int): Number of parallel jobs (defaults to CPU core count).
        use_threads (bool): Use threads instead of processes.
        show_progress (bool): Show progress bar.
        batch_size (int): Chunk size for batching.

    Returns:
        list: List of results.
    """
    if not JOBLIB_AVAILABLE:
        if show_progress:
            return [func(x) for x in tqdm(iterable)]
        else:
            return list(map(func, iterable))

    if n_jobs is None:
        n_jobs = get_num_cores()

    backend = 'threading' if use_threads else 'multiprocessing'

    if show_progress:
        results = []
        with Parallel(n_jobs=n_jobs, backend=backend, batch_size=batch_size) as parallel:
            tasks = (delayed(func)(x) for x in iterable)
            for r in tqdm(parallel(tasks), total=len(iterable)):
                results.append(r)
        return results

    else:
        with Parallel(n_jobs=n_jobs, backend=backend, batch_size=batch_size) as parallel:
            return parallel(delayed(func)(x) for x in iterable)
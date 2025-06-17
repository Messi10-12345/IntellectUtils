# intellect_utils/ray_utils.py

import ray

def init_ray(num_cpus=None, ignore_reinit_error=True):
    """
    Initialize Ray runtime with optional CPU count limit.

    Args:
        num_cpus (int or None): Number of CPU cores Ray can use.
        ignore_reinit_error (bool): Suppress error if Ray is already initialized.

    Returns:
        None
    """
    ray.init(num_cpus=num_cpus, ignore_reinit_error=ignore_reinit_error)

def shutdown_ray():
    """Shutdown Ray runtime."""
    ray.shutdown()

def run_parallel(func, args_list):
    """
    Run a function in parallel on Ray.

    Args:
        func (callable): Function to execute remotely.
        args_list (list of tuples): List of argument tuples to pass to func.

    Returns:
        list: Results from each remote execution.
    """
    remote_func = ray.remote(func)
    futures = [remote_func.remote(*args) for args in args_list]
    results = ray.get(futures)
    return results

def available_resources():
    """Return Ray cluster resource info."""
    return ray.available_resources()
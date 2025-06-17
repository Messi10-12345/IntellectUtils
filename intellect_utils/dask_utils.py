# intellect_utils/dask_helpers.py

try:
    import dask.array as da
    import dask.dataframe as dd
    from dask.distributed import Client, LocalCluster
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

def create_local_cluster(n_workers=None, threads_per_worker=1):
    """
    Create a local Dask cluster for parallel computation.

    Args:
        n_workers (int): Number of workers (defaults to CPU cores).
        threads_per_worker (int): Threads per worker.

    Returns:
        Client: Dask distributed client connected to cluster.
    """
    if not DASK_AVAILABLE:
        raise ImportError("Dask is not installed. Please install with `pip install dask[distributed]`.")

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
    client = Client(cluster)
    return client

def to_dask_array(np_array, chunks="auto"):
    """
    Convert a NumPy array to a Dask array.

    Args:
        np_array (np.ndarray): Input NumPy array.
        chunks (str or tuple): Chunk size, default 'auto'.

    Returns:
        dask.array.Array: Dask array.
    """
    if not DASK_AVAILABLE:
        raise ImportError("Dask is not installed.")
    return da.from_array(np_array, chunks=chunks)

def to_dask_dataframe(pd_df, npartitions=None):
    """
    Convert a Pandas DataFrame to a Dask DataFrame.

    Args:
        pd_df (pd.DataFrame): Input Pandas DataFrame.
        npartitions (int): Number of partitions.

    Returns:
        dask.dataframe.DataFrame: Dask dataframe.
    """
    if not DASK_AVAILABLE:
        raise ImportError("Dask is not installed.")
    return dd.from_pandas(pd_df, npartitions=npartitions)
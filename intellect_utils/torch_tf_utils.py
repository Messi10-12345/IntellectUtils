# intellect_utils/torch_tf_utils.py

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

def set_torch_num_threads(n):
    if torch is None:
        raise ImportError("PyTorch not installed.")
    torch.set_num_threads(n)

def get_torch_num_threads():
    if torch is None:
        raise ImportError("PyTorch not installed.")
    return torch.get_num_threads()

def set_tf_num_threads(intra=None, inter=None):
    if tf is None:
        raise ImportError("TensorFlow not installed.")
    if intra is not None:
        tf.config.threading.set_intra_op_parallelism_threads(intra)
    if inter is not None:
        tf.config.threading.set_inter_op_parallelism_threads(inter)

def get_tf_num_threads():
    if tf is None:
        raise ImportError("TensorFlow not installed.")
    intra = tf.config.threading.get_intra_op_parallelism_threads()
    inter = tf.config.threading.get_inter_op_parallelism_threads()
    return intra, inter

def torch_cpu_info():
    if torch is None:
        return "PyTorch not installed."
    return {
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "is_mkldnn_enabled": torch.backends.mkldnn.is_available(),
    }

def tf_cpu_info():
    if tf is None:
        return "TensorFlow not installed."
    # TensorFlow doesn't provide an easy CPU info API, so just return threading config.
    return {
        "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
        "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
        "is_mkl_enabled": tf.test.is_built_with_mkl(),
    }
# intellect_utils/onnx_utils.py

import onnxruntime as ort
import time
import numpy as np

def create_session(model_path, providers=None, intra_op_num_threads=None):
    """
    Create an ONNX Runtime InferenceSession with optional CPU optimizations.

    Args:
        model_path (str): Path to ONNX model file.
        providers (list): Execution providers, default to ['CPUExecutionProvider'].
        intra_op_num_threads (int): Number of threads for intra-op parallelism.

    Returns:
        onnxruntime.InferenceSession
    """
    if providers is None:
        providers = ['CPUExecutionProvider']

    sess_options = ort.SessionOptions()
    if intra_op_num_threads:
        sess_options.intra_op_num_threads = intra_op_num_threads

    session = ort.InferenceSession(model_path, sess_options, providers=providers)
    return session

def run_inference(session, input_feed):
    """
    Run inference on the ONNX Runtime session.

    Args:
        session: ONNX Runtime InferenceSession.
        input_feed (dict): Dictionary mapping input names to numpy arrays.

    Returns:
        list: Outputs of the model.
    """
    outputs = session.run(None, input_feed)
    return outputs

def benchmark_inference(session, input_feed, runs=10):
    """
    Benchmark inference speed.

    Args:
        session: ONNX Runtime InferenceSession.
        input_feed (dict): Model inputs.
        runs (int): Number of times to run inference.

    Returns:
        float: Average inference time per run (seconds).
    """
    # Warmup run
    session.run(None, input_feed)

    start = time.time()
    for _ in range(runs):
        session.run(None, input_feed)
    end = time.time()

    avg_time = (end - start) / runs
    return avg_time

def get_providers():
    """
    Return available ONNX Runtime execution providers on the current machine.
    """
    return ort.get_available_providers()
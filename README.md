# IntellectUtils 0.1.0 BETA

**🚀 IntellectUtils** is a high-performance Python utility library built to **accelerate AI workloads on CPU**.  
Designed by **ARCX Studios** and the **Intellect** team — led by [Rayhan Aman](https://github.com/rayhanAman) and [Cooper Johnson](https://github.com/) — this toolkit simplifies and supercharges training, preprocessing, and inference on machines without powerful GPUs.

> 🎯 “AI shouldn’t be slow just because you don’t have a GPU.”

---

## ✨ Features

- 🔥 JIT acceleration with **Numba**
- ⚙️ Multi-threaded & multi-process parallelism with **Joblib** and **concurrent.futures**
- 🔗 Distributed task execution using **Ray**
- 🧩 Graph-based task scheduling using **Dask**
- 💡 CPU tuning via **OpenBLAS**, **BLIS**, and **Intel MKL**
- 🧠 Efficient model inference using **ONNX Runtime**
- 📦 Built-in helpers for **TensorFlow** and **PyTorch** CPU optimization
- 🧪 Auto-tuning of environment variables for threading & performance

---

## 🏁 Quick Start

```bash

git clone https://github.com/LaserRay1234/IntellectUtils
cd IntellectUtils
pip install .

1. JIT Acceleration with Numba

    from intellect_utils.jit_utils import jit_compile

    @jit_compile
    def add_loop(arr):
        total = 0
        for x in arr:
            total += x
        return total

    print(add_loop([1, 2, 3, 4]))

2. Parallel Task Execution (CPU)

    from intellect_utils.parallel_utils import run_parallel_threadpool

    def slow_add(x, y):
        return x + y

    tasks = [(i, i * 2) for i in range(10)]
    results = run_parallel_threadpool(slow_add, tasks)
    print(results)

3. Distributed Computing with Ray

    from intellect_utils.ray_utils import init_ray, run_parallel, shutdown_ray

    def square(x):
        return x * x

    init_ray(num_cpus=4)
    tasks = [(i,) for i in range(8)]
    results = run_parallel(square, tasks)
    shutdown_ray()

    print(results)

4. ONNX Model Inference (CPU Optimized)

    from intellect_utils.onnx_utils import run_onnx_model

    output = run_onnx_model("model.onnx", input_dict={"input": input_array})

5. BLAS / MKL Thread Optimization
    from intellect_utils.blas_utils import set_blas_threads

    set_blas_threads(4)  # Apply to NumPy, SciPy, PyTorch, etc.

6. PyTorch & TensorFlow CPU Tuning

    from intellect_utils.torch_tf_utils import set_torch_num_threads, set_tf_num_threads

    set_torch_num_threads(4)
    set_tf_num_threads(4)

📁 Project Structure

intellect_utils/
├── jit_utils.py
├── parallel_utils.py
├── dask_utils.py
├── mkl_utils.py
├── onnx_utils.py
├── torch_tf_utils.py
├── ray_utils.py
├── blas_utils.py
└── __init__.py

🔧 Configuration
Use these utilities before importing NumPy/TensorFlow/PyTorch:

from intellect_utils.blas_utils import set_blas_threads
set_blas_threads(4)

🧑‍💻 About the Creators
ARCX Studios

Where AI meets speed.

💡 Built by Rayhan Aman, founder of Intellect, developer and AI optimization engineer.

💼 Co-created with Cooper Johnson, CFO, co-CEO of ARCX Studios, and systems strategist.

🧠 Designed to help students, developers, and startups train AI faster using just CPUs.

🪪 License
MIT License

🌍 Vision
Making AI development faster, more accessible, and GPU-free — powered by Intellect and ARCX Studios.

📫 Contact
GitHub: LaserRay1234

GitHub: ARCX-OFFICIAL

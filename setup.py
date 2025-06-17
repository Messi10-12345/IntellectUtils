from setuptools import setup, find_packages

setup(
    name="IntellectUtils",
    version="0.1.0-beta",
    author="Your Name",
    author_email="your.email@example.com",
    description="High-performance CPU utilities for AI training and parallel computation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/LaserRay1234/IntellectUtils",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires=">=3.7",
    install_requires=[
        "numba",
        "joblib",
        "dask",
        "ray",
        "onnxruntime",
        "torch",
        "tensorflow",
        "threadpoolctl",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    include_package_data=True,
    zip_safe=False,
)
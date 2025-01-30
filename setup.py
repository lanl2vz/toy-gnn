
from setuptools import setup, find_packages

setup(
    name="toy_gnn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torch_geometric",
    ],
    author="lanl2vz",
    description="A toy implementation of Graph Neural Networks",
    python_requires=">=3.7",
)
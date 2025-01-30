from setuptools import setup, find_packages

setup(
    name="toy_gnn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torch_geometric>=1.0.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
    ],
    author="lanl2vz",
    author_email="author@example.com",
    description="A toy implementation of Graph Neural Networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lanl2vz/toy-gnn",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "toy-gnn-train=toy_gnn.train:train_model",
            "toy-gnn-test=toy_gnn.test:main",
        ],
    },
)
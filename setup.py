"""Install the oamp package.

    pip install -e .

The optional CUDA extension for the max-abs anchor variant is built separately,
inside oamp_cuda/, with `python setup.py build_ext --inplace`. oamp.cuda_ops
falls back to PyTorch without it, and the training runs in the paper never use
it.
"""

from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
long_description = (ROOT / "README.md").read_text() if (ROOT / "README.md").exists() else ""

setup(
    name="oamp",
    version="0.1.0",
    description="Backward-only activation compression for LoRA fine-tuning, with the run records behind the paper",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["oamp", "oamp.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.40",
        "peft>=0.10",
        "bitsandbytes",
        "torchao>=0.17",
        "datasets",
        "numpy",
        "safetensors",
    ],
    extras_require={
        "viz": ["matplotlib"],
        "dev": ["pytest", "ruff"],
    },
    zip_safe=False,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

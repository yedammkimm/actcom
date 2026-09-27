"""Install the oamp package and, when torch and nvcc are available, build the CUDA extension.

    pip install -e .

The extension is optional: oamp.cuda_ops falls back to PyTorch without it, and
the training runs in the paper never use it.
"""

from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent

# Optional CUDA extension
ext_modules = []
cmdclass = {}
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        arch = [f"--generate-code=arch=compute_{cap[0]}{cap[1]},code=sm_{cap[0]}{cap[1]}"]
        ext_modules.append(
            CUDAExtension(
                name="oamp_cuda",
                sources=[
                    "oamp_cuda/oamp_ops.cpp",
                    "oamp_cuda/oamp_kernels.cu",
                    "oamp_cuda/oamp_bilevel_ops.cpp",
                    "oamp_cuda/oamp_bilevel_kernels.cu",
                ],
                include_dirs=["oamp_cuda"],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": ["-O3", "--use_fast_math", "-std=c++17"] + arch,
                },
            )
        )
        cmdclass["build_ext"] = BuildExtension.with_options(use_ninja=False)
except ImportError:
    pass  # torch is not installed yet; install it first, then re-run

long_description = (ROOT / "README.md").read_text() if (ROOT / "README.md").exists() else ""

setup(
    name="oamp",
    version="0.1.0",
    description="Outlier-Aware Mixed-Precision Activation Compression for LLM Fine-tuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["oamp", "oamp.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.40",
        "peft>=0.10",
        "datasets",
        "numpy",
        "tqdm",
    ],
    extras_require={
        "viz": ["matplotlib"],
        "dev": ["pytest", "ruff"],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

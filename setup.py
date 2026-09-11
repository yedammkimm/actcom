"""
OAMP — Outlier-Aware Mixed Precision

Minimal setup for installing the Python package and (optionally) building
the CUDA extension for FP8/FP4 bi-level activation packing.

Usage
-----
    pip install -e .              # Python only
    pip install -e .[cuda]        # Build CUDA kernels (requires nvcc)
"""

from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent

# ── Optional CUDA extension ──────────────────────────────────────────────────
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
    pass  # torch not installed yet; user must install torch first

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
#!/usr/bin/env python3
"""
OAMP (Hierarchical Memory Adaptation) Setup
==========================================

Professional-grade setup.py for pip installation and Docker deployment.
Supports both development and production installation modes.
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch
from pathlib import Path

# Read version from version file
VERSION_FILE = Path(__file__).parent / "oamp" / "version.py"
if VERSION_FILE.exists():
    exec(open(VERSION_FILE).read())
else:
    __version__ = "0.1.0"

# Read README for long description
README_PATH = Path(__file__).parent / "README.md"
long_description = README_PATH.read_text() if README_PATH.exists() else ""

# CUDA extension setup
def get_cuda_extension():
    """Build CUDA extension with robust error handling"""
    cuda_sources = [
        'oamp_cuda/oamp_ops.cpp',
        'oamp_cuda/oamp_kernels.cu',
    ]
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Building CPU-only version.")
        return None
    
    # Verify source files exist
    for source in cuda_sources:
        if not Path(source).exists():
            print(f"WARNING: Source file {source} not found. Skipping CUDA extension.")
            return None
    
    # Get CUDA compute capability
    compute_cap = torch.cuda.get_device_capability()
    arch_list = [f'--generate-code=arch=compute_{compute_cap[0]}{compute_cap[1]},code=sm_{compute_cap[0]}{compute_cap[1]}']
    
    return CUDAExtension(
        name='oamp_cuda',
        sources=cuda_sources,
        extra_compile_args={
            'cxx': ['-O3', '-std=c++14'],
            'nvcc': [
                '-O3', 
                '--use_fast_math',
                '--expt-extended-lambda',
                '-std=c++14'
            ] + arch_list
        },
        include_dirs=[
            'oamp_cuda',
            torch.utils.cpp_extension.include_paths()
        ]
    )

# Build extension list
ext_modules = []
cuda_ext = get_cuda_extension()
if cuda_ext:
    ext_modules.append(cuda_ext)

setup(
    # Package metadata
    name='oamp-kv-cache',
    version=__version__,
    author='OAMP Research Team',
    author_email='research@oamp.ai',
    description='Hierarchical Memory Adaptation for Large Language Model KV Cache Compression',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/oamp-research/oamp-kv-cache',
    
    # Package structure
    packages=find_packages(),
    package_data={
        'oamp': ['cuda/*.h', 'cuda/*.cuh'],
        'oamp_cuda': ['*.h']
    },
    include_package_data=True,
    
    # Dependencies
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.9.0',
        'transformers>=4.20.0',
        'numpy>=1.20.0',
        'tqdm',
        'matplotlib',
        'seaborn'
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'black',
            'flake8',
            'mypy',
            'jupyter'
        ],
        'eval': [
            'lm-eval>=0.3.0',
            'datasets'
        ],
        'viz': [
            'plotly',
            'pandas',
            'scipy'
        ]
    },
    
    # CUDA Extension
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    },
    zip_safe=False,
    
    # Entry points for CLI
    entry_points={
        'console_scripts': [
            'oamp-benchmark=experiments.benchmarks.benchmark_memory_smart:main',
            'oamp-latency=experiments.benchmarks.benchmark_latency:main',
            'oamp-eval=experiments.benchmarks.benchmark_downstream:main',
        ],
    },
    
    # Classification
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Hardware :: Symmetric Multi-processing'
    ],
    
    # Keywords for discovery
    keywords='llm, kv-cache, compression, cuda, memory-optimization, transformers'
)
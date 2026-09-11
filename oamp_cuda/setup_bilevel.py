# Copyright (c) 2025 OAMP Research Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
setup_bilevel.py
================
OAMP Bi-Level CUDA Extension build script.

Build
---------
  cd /path/to/oamp_cuda
  python setup_bilevel.py build_ext --inplace

Or install from project root:
  pip install -e . --no-build-isolation

Output: oamp_bilevel.cpython-*.so  (in oamp_cuda/ directory)

Architecture Strategy
-------------
  System nvcc(12.1) supports: up to sm_90
  Current GPU (GB10, sm_121): requires CUDA 12.8+

  -> Generate sm_90 PTX with nvcc 12.1, CUDA 12.8 runtime JIT-compiles for sm_121.
  -> PTX is forward-compatible: sm_90 PTX runs directly on sm_121.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import subprocess


def get_nvcc_max_arch() -> str:
    """
    Return the highest compute capability supported by the system nvcc.
    nvcc 12.1  -> sm_90
    nvcc 12.8+ -> sm_121 etc.
    """
    try:
        out = subprocess.check_output(
            ["nvcc", "--list-gpu-arch"], stderr=subprocess.STDOUT, text=True
        )
        # Example output: "compute_50\ncompute_52\n...compute_90\n"
        archs = [
            line.strip().replace("compute_", "")
            for line in out.strip().splitlines()
            if line.strip().startswith("compute_")
        ]
        if archs:
            return max(archs, key=lambda a: int(a))
    except Exception:
        pass
    return "90"  # Conservative default


def get_cuda_arch_flags() -> list:
    """
    GPU arch <= nvcc max supported  -> compile natively for that arch
    GPU arch  > nvcc max supported  -> generate PTX at nvcc max (forward-compatible JIT)
    """
    nvcc_max = get_nvcc_max_arch()

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        gpu_arch = f"{major}{minor}"
    else:
        gpu_arch = "80"

    # If nvcc supports the GPU arch directly, generate both native binary + PTX
    if int(gpu_arch) <= int(nvcc_max):
        return [
            f"-gencode=arch=compute_{gpu_arch},code=sm_{gpu_arch}",
            f"-gencode=arch=compute_{gpu_arch},code=compute_{gpu_arch}",
        ]
    else:
        # GPU arch exceeds nvcc range -> generate max PTX, runtime JIT handles the rest
        print(
            f"[setup_bilevel] GPU sm_{gpu_arch} > nvcc max sm_{nvcc_max}. "
            f"PTX fallback: computing PTX for sm_{nvcc_max}, "
            f"JIT will compile for sm_{gpu_arch} at runtime."
        )
        return [
            f"-gencode=arch=compute_{nvcc_max},code=compute_{nvcc_max}",
        ]


cuda_flags = [
    "-O3",
    "-use_fast_math",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-Xcompiler=-fPIC",
    "-lineinfo",
    "--ptxas-options=-v",
] + get_cuda_arch_flags()

cxx_flags = [
    "-O3",
    "-std=c++17",
    "-fPIC",
    "-Wno-deprecated-declarations",
    "-Wno-unused-function",
]

setup(
    name="oamp_bilevel",
    version="1.0.0",
    description="OAMP Bi-Level FP8/FP4 CUDA Kernels (Nibble Packing + Fused Quantize)",
    ext_modules=[
        CUDAExtension(
            name="oamp_bilevel",
            sources=[
                "oamp_bilevel_ops.cpp",
                "oamp_bilevel_kernels.cu",
            ],
            extra_compile_args={
                "cxx":  cxx_flags,
                "nvcc": cuda_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

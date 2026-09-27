# Copyright 2026 OAMP Authors. Licensed under the Apache License, Version 2.0.

"""Build the anchor-variant CUDA kernels as the extension `oamp_bilevel`.

    cd oamp_cuda
    python setup_bilevel.py build_ext --inplace

writes oamp_bilevel.cpython-*.so into this directory, where oamp/cuda_ops.py
looks for it.

Architecture flags: when nvcc supports the GPU's compute capability the
extension is compiled for it natively, with PTX as well. When the GPU is
newer than nvcc (the GB10 is sm_121 and nvcc 12.1 stops at sm_90), only PTX
for the highest supported architecture is generated, and the CUDA runtime
compiles it for the GPU the first time the extension loads.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import subprocess


def get_nvcc_max_arch() -> str:
    """Return the highest compute capability nvcc lists ('90' for nvcc 12.1), or
    '90' when nvcc cannot be queried.
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
    """Return the -gencode flags: native code plus PTX when nvcc supports the
    GPU, otherwise PTX for nvcc's highest architecture.
    """
    nvcc_max = get_nvcc_max_arch()

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        gpu_arch = f"{major}{minor}"
    else:
        gpu_arch = "80"

    # nvcc supports the GPU directly: native code plus PTX
    if int(gpu_arch) <= int(nvcc_max):
        return [
            f"-gencode=arch=compute_{gpu_arch},code=sm_{gpu_arch}",
            f"-gencode=arch=compute_{gpu_arch},code=compute_{gpu_arch}",
        ]
    else:
        # GPU newer than nvcc: PTX for the highest supported architecture, compiled at load time
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

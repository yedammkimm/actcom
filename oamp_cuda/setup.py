# Copyright 2026 OAMP Authors. Licensed under the Apache License, Version 2.0.
"""Build the earlier three-level packer (oamp_ops.cpp, oamp_kernels.cu) as the
extension `oamp_cuda`. Kept for reference; the anchor variant is built by
setup_bilevel.py, and the training runs in the paper use neither.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cuda_flags = [
    '-O3',
    '-use_fast_math',
    '--ptxas-options=-v',
    '-lineinfo',
    '-std=c++17',
    '-gencode=arch=compute_80,code=sm_80',
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda',
    '-Xcompiler=-fPIC',
]

cxx_flags = [
    '-O3',
    '-std=c++17',
    '-fPIC',
    '-Wno-deprecated-declarations',
    '-Wno-unused-function',
]

setup(
    name='oamp_cuda',
    ext_modules=[
        CUDAExtension(
            name='oamp_cuda',
            sources=[
                'oamp_ops.cpp',
                'oamp_kernels.cu',
            ],
            extra_compile_args={
                'cxx': cxx_flags,
                'nvcc': cuda_flags,
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

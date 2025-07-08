from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

setup(
    name='gelu_cuda',
    ext_modules=[
        CUDAExtension(
            name='gelu_cuda',
            sources=['gelu_cuda.cu'],
            include_dirs=torch.utils.cpp_extension.include_paths(),
            extra_compile_args={
                'cxx': ['/O2'],
                'nvcc': ['-O2']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)


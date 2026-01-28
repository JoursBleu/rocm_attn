from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="rocm_attn_op",
    version="0.1.0",
    packages=["rocm_attn_op"],
    package_dir={"": "python"},
    ext_modules=[
        CUDAExtension(
            name="rocm_attn_op.rocm_attn_ext",
            sources=[
                "src/attn_torch.cpp",
                "src/attn_kernel_torch.cu",
            ],
            include_dirs=["src"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

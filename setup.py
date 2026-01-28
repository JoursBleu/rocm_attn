import os

from setuptools import setup
import torch.utils.cpp_extension as cpp_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, COMMON_HIP_FLAGS, IS_HIP_EXTENSION


if IS_HIP_EXTENSION:
    # 移除禁用 half 运算/转换的默认宏，确保 rocWMMA 可用
    remove_flags = {"-D__HIP_NO_HALF_OPERATORS__=1", "-D__HIP_NO_HALF_CONVERSIONS__=1"}
    COMMON_HIP_FLAGS[:] = [f for f in COMMON_HIP_FLAGS if f not in remove_flags]

    _orig_write_ninja_file = cpp_ext._write_ninja_file

    def _filter_flags(flags):
        if flags is None:
            return flags
        if isinstance(flags, str):
            return " ".join([f for f in flags.split() if f not in remove_flags])
        return [f for f in flags if f not in remove_flags]

    def _write_ninja_file(
        path,
        cflags,
        post_cflags,
        cuda_cflags,
        cuda_post_cflags,
        cuda_dlink_post_cflags,
        *rest,
        **kwargs,
    ):
        cflags = _filter_flags(cflags)
        post_cflags = _filter_flags(post_cflags)
        cuda_cflags = _filter_flags(cuda_cflags)
        cuda_post_cflags = _filter_flags(cuda_post_cflags)
        return _orig_write_ninja_file(
            path,
            cflags,
            post_cflags,
            cuda_cflags,
            cuda_post_cflags,
            cuda_dlink_post_cflags,
            *rest,
            **kwargs,
        )

    cpp_ext._write_ninja_file = _write_ninja_file

    # 限定 ROCm 架构，避免 rocWMMA 对不支持架构触发 static_assert
    rocm_arch = os.environ.get("ROCM_ARCH", "gfx1201")
    os.environ["ROCM_ARCH"] = rocm_arch
    os.environ["HCC_AMDGPU_TARGET"] = rocm_arch
    os.environ["PYTORCH_ROCM_ARCH"] = rocm_arch
else:
    rocm_arch = None


extra_nvcc = ["-O3"] + ([f"--offload-arch={rocm_arch}"] if rocm_arch else [])


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
                "nvcc": extra_nvcc,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

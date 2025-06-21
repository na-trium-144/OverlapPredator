from setuptools import setup, Extension
import numpy
from glob import glob


subsampling_module = Extension(
    name="OverlapPredator.cpp_wrappers.cpp_subsampling.grid_subsampling",
    sources=[
        "OverlapPredator/cpp_wrappers/cpp_utils/cloud/cloud.cpp",
        "OverlapPredator/cpp_wrappers/cpp_subsampling/grid_subsampling/grid_subsampling.cpp",
        "OverlapPredator/cpp_wrappers/cpp_subsampling/wrapper.cpp",
    ],
    extra_compile_args=["-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"],
    include_dirs=[numpy.get_include()],
)
neighbors_module = Extension(
    name="OverlapPredator.cpp_wrappers.cpp_neighbors.radius_neighbors",
    sources=[
        "OverlapPredator/cpp_wrappers/cpp_utils/cloud/cloud.cpp",
        "OverlapPredator/cpp_wrappers/cpp_neighbors/neighbors/neighbors.cpp",
        "OverlapPredator/cpp_wrappers/cpp_neighbors/wrapper.cpp",
    ],
    extra_compile_args=["-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"],
    include_dirs=[numpy.get_include()],
)

setup(
    name="OverlapPredator",
    version="0.1",
    packages=[
        "OverlapPredator",
        "OverlapPredator.common",
        "OverlapPredator.common.math",
        "OverlapPredator.common.math_torch",
        "OverlapPredator.configs",
        "OverlapPredator.datasets",
        "OverlapPredator.kernels",
        "OverlapPredator.lib",
        "OverlapPredator.models",
        "OverlapPredator.cpp_wrappers",
        "OverlapPredator.cpp_wrappers.cpp_neighbors",
        "OverlapPredator.cpp_wrappers.cpp_subsampling",
    ],
    include_package_data=True,
    ext_modules=[subsampling_module, neighbors_module],
    install_requires=[
        "PyYAML",
        "coloredlogs",
        "easydict",
        "h5py",
        "matplotlib",
        "nibabel",
        "open3d",
        "scipy",
        "tensorboardX",
        "torch",
        "torchvision",
        "tqdm",
    ],
)

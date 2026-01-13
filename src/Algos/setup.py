from setuptools import setup, Extension
import pybind11
import os
import glob
from setuptools.command.build_ext import build_ext as build_ext_orig

# Base directory of setup.py
base_dir = os.path.dirname(os.path.abspath(__file__))

# Collect all C++ source files
cpp_sources = [os.path.join(base_dir, "ann_algos.cpp")]
cpp_sources += glob.glob(os.path.join(base_dir, "Algorithms", "*.cpp"))

# Extension module
ext_modules = [
    Extension(
        "ann_algos",
        cpp_sources,
        include_dirs=[
            pybind11.get_include(),
            os.path.join(base_dir, "Algorithms")
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17", "-w"],
    )
]

# Custom build_ext to always build in-place
class build_ext_inplace(build_ext_orig):
    def build_extensions(self):
        # Force output directory to current folder
        self.build_lib = base_dir
        super().build_extensions()

setup(
    name="ann_algos",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext_inplace},
)

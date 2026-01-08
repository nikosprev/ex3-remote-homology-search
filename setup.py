from setuptools import setup, Extension
import pybind11
import sys
import warnings 
warnings.filterwarnings("ignore")
ext_modules = [
    Extension(
        "ann_algos",
        ["Algos/ann_algos.cpp", "Algos/Algorithms/metrics.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "src"
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17" , "-w"],
    ),
]

setup(
    name="ann_algos",
    version="0.1",
    ext_modules=ext_modules,
)
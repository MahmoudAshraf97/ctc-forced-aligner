from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages
import sys

ext_modules = [
    Pybind11Extension(
        "ctc_forced_aligner.ctc_forced_aligner",
        ["ctc_forced_aligner/forced_align_impl.cpp"],
        extra_compile_args=["/O2"] if sys.platform == "win32" else ["-O3"],
    )
]

setup(
    name="ctc_forced_aligner",
    version="0.3.1",
    packages=find_packages(),
    ext_modules=ext_modules,
    package_data={
        "ctc_forced_aligner": ["punctuations.lst", "uroman/bin/*", "uroman/data/*", "uroman/lib/*"],
        "README.md": ["README.md"],
        "LICENSE": ["LICENSE"],
    },
    include_package_data=True,
    cmdclass={"build_ext": build_ext},
)

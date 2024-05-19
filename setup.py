from setuptools import find_packages, setup
from torch.utils import cpp_extension

ext_modules = [
    cpp_extension.CppExtension(
        "ctc_forced_aligner.ctc_forced_aligner",
        ["ctc_forced_aligner/forced_align_impl.cpp"],
        extra_compile_args=['-O3'],
    )
]

setup(
    name="ctc-forced-aligner",
    py_modules=["ctc_forced_aligner"],
    version="0.1",
    description="Text to speech alignment using CTC forced alignment",
    readme="README.md",
    python_requires=">=3.8",
    author="Mahmoud Ashraf",
    author_email="hassouna97.ma@gmail.com",
    url="https://github.com/MahmoudAshraf97/ctc-forced-aligner",
    license="CC-BY-NC 4.0",
    packages=find_packages(),
    install_requires=[line.strip() for line in open("requirements.txt")],
    ext_modules=ext_modules,
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    entry_points={
        "console_scripts": ["ctc-forced-aligner=ctc_forced_aligner.align:cli"],
    },
    package_data={
        "": [
            "punctuations.lst",
            "uroman/bin/**/*.*",
            "uroman/data/**/*.*",
            "uroman/lib/**/*.*",
        ]
    },
    include_package_data=True,
)

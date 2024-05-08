from setuptools import find_packages, setup
from glob import glob

setup(
    name="ctc-forced-aligner",
    py_modules=["ctc_forced_aligner"],
    version="0.1",
    description="Text to speech alignment using CTC forced alignment",
    readme="README.md",
    python_requires=">=3.8",
    author="Mahmoud Ashraf",
    url="https://github.com/MahmoudAshraf97/ctc-forced-aligner",
    license="CC-BY-NC 4.0",
    packages=find_packages(),
    install_requires=[line.strip() for line in open("requirements.txt")],
    entry_points={
        "console_scripts": ["ctc-forced-aligner=ctc_forced_aligner.align:cli"],
    },
    package_data={"": ["punctuations.lst","uroman/*"]},
    include_package_data=True,
)

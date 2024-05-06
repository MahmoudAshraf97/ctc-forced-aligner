from setuptools import find_packages, setup

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
    install_requires=["transformers", "torchaudio", "torch", ],
    entry_points={
        "console_scripts": ["ctc-forced-aligner=ctc_forced_aligner.align:cli"],
    },
    include_package_data=True,
)
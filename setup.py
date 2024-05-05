from setuptools import find_packages, setup

setup(
    name="ctc-force-aligner",
    py_modules=["ctc_force_aligner"],
    version="0.1",
    description="Text to speech alignment using CTC force alignment",
    readme="README.md",
    python_requires=">=3.8",
    author="Mahmoud Ashraf",
    url="https://github.com/MahmoudAshraf97/ctc-forced-aligner",
    license="CC-BY-NC 4.0",
    packages=find_packages(exclude=["tests*"]),
    install_requires=["transformers", "torchaudio", "torch", ],
    entry_points={
        "console_scripts": ["ctc-forced-aligner=ctc_force_aligner.align:cli"],
    },
    include_package_data=True,
)
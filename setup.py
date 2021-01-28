"""Sets up the package."""

from pathlib import Path

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# define a function that reads a file in this directory
read = lambda p: Path(Path(__file__).resolve().parent / p).read_text()

setup(
    name="sequence-jacobian",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=read("requirements.txt").splitlines(),
    version="0.0.1",
    author="Michael Cai",
    author_email="michaelcai@u.northwestern.edu",
    description="Sequence-Space Jacobian Methods for Solving and Estimating Heterogeneous Agent Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shade-econ/sequence-jacobian",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

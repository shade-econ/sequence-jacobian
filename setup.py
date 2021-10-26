"""Sets up the package."""

from pathlib import Path

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# define a function that reads a file in this directory
read = lambda p: Path(Path(__file__).resolve().parent / p).read_text()

setup(
    name="sequence-jacobian",
    python_requires=">=3.7",
    install_requires=read("requirements.txt").splitlines(),
    version="1.0.0-alpha",
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

    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)

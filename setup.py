"""Sets up the package."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sequence-jacobian",
    version="0.0.1",
    author="Michael Cai",
    author_email="michaelcai@u.northwestern.edu",
    description="Sequence-Space Jacobian Methods for Solving and Estimating Heterogeneous Agent Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shade-econ/sequence-jacobian",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

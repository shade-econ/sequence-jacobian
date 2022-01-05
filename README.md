# Sequence-Space Jacobian (SSJ)

SSJ is a Python toolkit for analyzing dynamic macroeconomic models with (or without) rich microeconomic heterogeneity.

The conceptual framework is based on our paper Adrien Auclert, Bence Bardóczy, Matthew Rognlie, Ludwig Straub (2021), [Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models](https://doi.org/10.3982/ECTA17434), Econometrica 89(5), pp. 2375–2408 [[ungated copy]](http://mattrognlie.com/sequence_space_jacobian.pdf).

## Requirements

SSJ requires the following software to be installed:
- [Python 3](https://www.python.org/)

We recommend that you install the latest [Python 3 Anaconda](https://www.anaconda.com/distribution/) distribution. This includes all of the packages and tools that you will need to run our code.  

## Installation 

Install from [PyPI](https://pypi.org/) by running (COMING SOON)
```
pip install sequence-jacobian
```

Install from source by running 
```
pip install git+https://github.com/shade-econ/sequence-jacobian@master
```
<!-- or simply [click here](https://github.com/shade-econ/sequence-jacobian/archive/master.zip) to download all files as a zip. -->

## Resources

Learn SSJ from examples. 

- RBC notebook ([html](notebooks/rbc.html), [Jupyter](notebooks/rbc.ipynb)) 
- Krusell-Smith notebook ([html](notebooks/krusell_smith.html), [Jupyter](notebooks/krusell_smith.ipynb)) 
- One-asset HANK notebook ([html](notebooks/hank.html), [Jupyter](notebooks/hank.ipynb)) 
- Two-asset HANK notebook ([html](notebooks/two_asset.html), [Jupyter](notebooks/two_asset.ipynb)) 
- Labor search notebook ([html](notebooks/labor_search.html), [Jupyter](notebooks/labor_search.ipynb)) 

If you don't have Python yet, just start by reading the `html` version. If you do, we recommend downloading our code and running the Jupyter notebooks on your computer.

Once you install Anaconda, you will be able interact with the `Jupyter` notebooks. Just open a terminal, change directory to the folder with the notebooks, and type `jupyter notebook`. This will launch the notebook dashboard in your
default browser. Click on a notebook to get started. 

For more information on Jupyter notebooks, check out the [official quick start guide](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/). If you'd like to learn more about Python, the [QuantEcon](https://python-programming.quantecon.org/intro.html) lectures of Tom Sargent and John Stachurski are a great place to start.

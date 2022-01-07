# Sequence-Space Jacobian (SSJ)

SSJ is a toolkit for analyzing dynamic macroeconomic models with (or without) rich microeconomic heterogeneity.

The conceptual framework is based on our paper Adrien Auclert, Bence Bardóczy, Matthew Rognlie, Ludwig Straub (2021), [Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models](https://doi.org/10.3982/ECTA17434), Econometrica 89(5), pp. 2375–2408 [[ungated copy]](http://mattrognlie.com/sequence_space_jacobian.pdf).

## Requirements

SSJ runs on Python 3.7 or newer. We recommend that you install the latest [Anaconda](https://www.anaconda.com/distribution/) distribution. This includes all of the packages and tools that you will need to run our code. 

**Optional package**: SSJ provides an interface for plotting the directed acyclic graph (DAG) representation of models. This feature requires the [Graphviz](https://www.graphviz.org/) graph drawing software and the corresponding [Python package](https://pypi.org/project/graphviz/), you have to install both of these if you'd like to use it.

## Installation 

Open a terminal and type
```
pip install sequence-jacobian
```

## Resources

The `notebooks` folder contains a number of examples. We recommend working through the notebooks in this order. [Click here](https://github.com/shade-econ/sequence-jacobian/raw/master/notebooks/notebooks.zip) to download all notebooks as a zip.

- [RBC](https://github.com/shade-econ/sequence-jacobian/blob/master/notebooks/rbc.ipynb)
    - represent macro models as collections of blocks (DAG)
    - write SimpleBlocks and CombinedBlocks
    - compute linearized and non-linear (perfect-foresight) impulse responses
- [Krusell-Smith](https://github.com/shade-econ/sequence-jacobian/blob/master/notebooks/krusell_smith.ipynb)
    - write HetBlocks to represent heterogeneous agents
    - construct general-equilibrium Jacobians manually
    - compute the log-likelihood of the model given time-series data
- [One-asset HANK](https://github.com/shade-econ/sequence-jacobian/blob/master/notebooks/hank.ipynb) 
    - adapt an off-the-shelf HetBlock to any macro environment using helper functions
    - see a more advanced example of calibration
- [Two-asset HANK](https://github.com/shade-econ/sequence-jacobian/blob/master/notebooks/two_asset.ipynb)
    - write SolvedBlocks to represent implicit aggregate equilibrium conditions
    - re-use saved Jacobians
    - fine tune options of block methods 
- [Labor search](https://github.com/shade-econ/sequence-jacobian/blob/master/notebooks/labor_search.ipynb)
    - example with multiple exogenous states
    - shocks to transition matrix of exogenous states

If you need help with running Jupyter notebooks, check out the [official quick start guide](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/). If you'd like to learn more about Python, the [QuantEcon](https://python-programming.quantecon.org/intro.html) lectures of Tom Sargent and John Stachurski are a great place to start.

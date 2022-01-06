# Sequence-Space Jacobian (SSJ)

SSJ is a toolkit for analyzing dynamic macroeconomic models with (or without) rich microeconomic heterogeneity.

The conceptual framework is based on our paper Adrien Auclert, Bence Bardóczy, Matthew Rognlie, Ludwig Straub (2021), [Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models](https://doi.org/10.3982/ECTA17434), Econometrica 89(5), pp. 2375–2408 [[ungated copy]](http://mattrognlie.com/sequence_space_jacobian.pdf).

## Requirements

SSJ runs on Python 3. We recommend that you install the latest [Python 3 Anaconda](https://www.anaconda.com/distribution/) distribution. This includes all of the packages and tools that you will need to run our code. We test SSJ on Python 3.8. 

**Optional package**: SSJ provides an interface for plotting the directed acyclic graph (DAG) representation of models. This feature requires the [Graphviz](https://www.graphviz.org/) graph drawing software and the corresponding [Python package](https://pypi.org/project/graphviz/), you have to install both of these if you'd like to use it.

## Installation 

Install from [PyPI](https://pypi.org/) by running (COMING SOON)
```
pip install sequence-jacobian
```

Install from source by running 
```
pip install git+https://github.com/shade-econ/sequence-jacobian@master
```

## Resources

The `notebooks` folder contains a number of examples. We recommend working through the notebooks in this order. 

- [RBC](notebooks/rbc.ipynb)
- [Krusell-Smith](notebooks/krusell_smith.ipynb)
- [One-asset HANK](notebooks/hank.ipynb) 
- [Two-asset HANK](notebooks/two_asset.ipynb)
- [Labor search](notebooks/labor_search.ipynb) 

If you need help with running Jupyter notebooks, check out the [official quick start guide](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/). If you'd like to learn more about Python, the [QuantEcon](https://python-programming.quantecon.org/intro.html) lectures of Tom Sargent and John Stachurski are a great place to start.

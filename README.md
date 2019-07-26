# Sequence-Space Jacobian

Interactive guide to Auclert, Bardóczy, Rognlie, Straub (2019):
 "Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models". 

[Click here](https://github.com/shade-econ/sequence-jacobian/archive/master.zip) to download all files as a zip. Note: **major update** on July 26, 2019.

## 1. Resources

- [Paper](https://shade-econ.github.io/sequence-jacobian/sequence_jacobian_paper.pdf)
- [Beamer slides](https://shade-econ.github.io/sequence-jacobian/sequence_jacobian_slides.pdf)
- RBC notebook ([html](https://shade-econ.github.io/sequence-jacobian/rbc.html)) ([Jupyter](rbc.ipynb))
- Krusell-Smith notebook ([html](https://shade-econ.github.io/sequence-jacobian/krusell_smith.html)) ([Jupyter](krusell_smith.ipynb))
- One-asset HANK notebook ([html](https://shade-econ.github.io/sequence-jacobian/hank.html)) ([Jupyter](hank.ipynb))
- Two-asset HANK notebook ([html](https://shade-econ.github.io/sequence-jacobian/two_asset.html)) ([Jupyter](two_asset.ipynb))
- HA Jacobian notebook ([html](https://shade-econ.github.io/sequence-jacobian/het_jacobian.html)) ([Jupyter](het_jacobian.ipynb))

### 1.1 RBC notebook

**Warm-up.** Get familiar with solving models in sequence space using our tools. If you don't have Python,
 just start by reading the `html` version. If you do, we recommend downloading our code and running the Jupyter notebook directly on your computer.

### 1.2. Krusell-Smith notebook

**The first example.** A comprehensive tutorial in the context of a simple, well-known HA model. Shows how to compute the Jacobian both "by hand" and with our automated tools. Also shows how to calculate second moments and the likelihood function.

### 1.3. One-asset HANK notebook

**The second example.** Generalizes to a more complex model, with a focus on our automated tools to streamline the workflow. Introduces our winding number criterion for local determinacy.

### 1.4. Two-asset HANK notebook

**The third example.** Showcases the workflow for solving a state-of-the-art HANK model where households hold liquid and illiquid assets, and there are sticky prices, sticky wages, and capital adjustment costs on the production side. Introduces the concept of solved blocks.

### 1.5. HA Jacobian notebook

**Inside the black box.** A step-by-step examination of our fake news algorithm to compute Jacobians of HA blocks.    

## 2. Setting up Python

To install a full distribution of Python, with all of the packages and tools you will need to run our code,
download the latest [Python 3 Anaconda](https://www.anaconda.com/distribution/) distribution.
**Note:** make sure you choose the installer for Python version 3. 
Once you install Anaconda, you will be able to play with the notebooks we provided. Just open a terminal, change 
directory to the folder with notebooks, and type `jupyter notebook`. This will launch the notebook dashboard in your
default browser. Click on a notebook to get started. 

For more information on Jupyter notebooks, check out the
[official quick start guide](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/).
If you'd like to learn more about Python, the [QuantEcon](https://lectures.quantecon.org/py/) lectures of
Tom Sargent and John Stachurski are a great place to start.




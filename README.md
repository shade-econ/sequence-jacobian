# Sequence-Space Jacobian

Interactive guide to Auclert, Bardóczy, Rognlie, Straub (2019):
 "Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models". 

[Click here](https://github.com/shade-econ/sequence-jacobian/archive/master.zip) to download all files as a zip.

## 1. Resources

- [Beamer slides](https://shade-econ.github.io/sequence-jacobian/sequence_jacobian_slides.pdf)
- Working paper
- RBC notebook 
- Krusell-Smith notebook ([html](https://shade-econ.github.io/sequence-jacobian/krusell_smith.html)) ([Jupyter](krusell_smith.ipynb))
- One-Asset HANK notebook ([html](https://shade-econ.github.io/sequence-jacobian/hank.html)) ([Jupyter](hank.ipynb))
- Two-Asset HANK notebook
- HA Jacobian notebook ([html](https://shade-econ.github.io/sequence-jacobian/het_jacobian.html)) ([Jupyter](het_jacobian.ipynb))

### 1.1 RBC notebook

**Warm-up.** Get familiar with solving models in sequence space using our tools. If you don't have Python,
 just start by reading the `html` version. If you do, we recommend downloading our code and running the Jupyter notebook directly on your computer.

### 1.2. Krusell-Smith notebook

**The first example.** A comprehensive tutorial in the context of a simple, well-known HA model. 

### 1.3. One-Asset HANK notebook

**The second example.** Illustrates that our method generalizes well to larger models with significant
nonlinearities. Uses tools that automatically combine the Jacobians of individual model blocks. Also addresses the question of local determinacy.

### 1.4. Two-Asset HANK notebook

**The third example.** Showcases the workflow for solving a state-of-the-art HANK model with sticky prices, sticky wages, and capital adjustment costs. Introduces the concept of solved blocks.

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




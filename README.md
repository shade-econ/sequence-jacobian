# Sequence-Space Jacobian

Interactive guide to work in progress by Auclert, Bardóczy, Rognlie, Straub (2019):
 "Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models". 

[Click here](https://github.com/shade-econ/sequence-jacobian/archive/master.zip) to download all files as a zip.

## 1. Resources

- [Beamer slides](https://shade-econ.github.io/sequence-jacobian/sequence_jacobian_slides.pdf)
- Krusell-Smith notebook ([html](https://shade-econ.github.io/sequence-jacobian/krusell_smith.html)) ([Jupyter](krusell_smith.ipynb))
- HANK notebook ([html](https://shade-econ.github.io/sequence-jacobian/hank.html)) ([Jupyter](hank.ipynb))
- HA Jacobian notebook ([html](https://shade-econ.github.io/sequence-jacobian/het_jacobian.html)) ([Jupyter](het_jacobian.ipynb))

### 1.1. Beamer slides

**A good place to start.** You'll find a brief history of solution methods and where the
sequence-space Jacobians fit in; a formal description of our key concepts; and a demonstration of
their usefulness. The goal of these slides is to provide you with the big picture before you dive into the code.

### 1.2. Krusell-Smith notebook

**The first example.** A comprehensive tutorial in the context of a simple HA model. If you don't have Python,
 just start by reading the `html` version. If you do, we recommend downloading our code and running the Jupyter notebook directly on your computer.

### 1.3. HANK notebook

**The second example.** Illustrates that our method generalizes well to larger models with significant
nonlinearities. Uses tools that automatically combine the Jacobians of individual model blocks. Also addresses the question of local determinacy.

### 1.4. HA Jacobian notebook

**Inside the black box.** A step-by-step examination of our core algorithm to compute Jacobians of HA blocks.
Starts from an intuitive, brute-force construction, then shows how to exploit *time symmetries* in backward and forward iteration for a 400x speed gain.    

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




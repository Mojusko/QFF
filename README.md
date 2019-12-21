## Content

This repository includes the code used in paper:

`Mojmir Mutny & Andreas Krause, "Efficient High Dimensional Bayesian Optimization with Additivity and Quadrature Fourier Features", NIPS 2018`

For paper see [here](https://papers.nips.cc/paper/8115-efficient-high-dimensional-bayesian-optimization-with-additivity-and-quadrature-fourier-features).
Namely, we implement finite basis approximation to Gaussian processes. The main contribution of this paper 
is implementation of the method embed(x) which coincides with \Phi(x) in product approximation:

	`k(x,y) = \Phi(x)^\top \Phi(y)`

## Installation
First clone the repository:

`git clone https://github.com/Mojusko/QFF.git`

Inside the project directory, run

`pip install -e .`

The `-e` option installs the package in "editable" mode, where pip links to your local copy of the repository, instead of copying the files to the your site-packages directory. That way, a simple `git pull` will update the package.
The project requires Python 3.6, and the dependencies should be installed with the package.

## Updates
21/12/2019 - More efficient basis 


## Usage - Implements Phi(x) 

```python
from embedding import *
x = torch.random(100,1) ## 100 random points in 1D
emb = HermiteEmbedding(gamma=0.5, m=100, d=1, groups=None, approx = "hermite") # Squared exponential with lenghtscale 0.5 with 100 basis functions 
Phi = emb.embed(x)
```

## Demonstration
![alt text](https://github.com/Mojusko/QFF/blob/master/example.png "N/A")

- RFF of Rahimi & Recht (2007)
- Quasi-RFF Avron et. al. (2014) 
- Orthogonal RFF - Felix et. al. (2016)
- QFF - Mutny & Krause (2018)


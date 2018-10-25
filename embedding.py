__author__ = "Mojmir Mutny"
__copyright__ = "Copyright (c) 2018 Mojmir Mutny, ETH Zurich"
__credits__ = ["Mojmir Mutny", "Andreas Krause"]
__license__ = "MIT Licence"
__version__ = "0.2"
__email__ = "mojmir.mutny@inf.ethz.ch"
__status__ = "DEV"


"""
This file implements code used in paper:

	Mojmir Mutny & Andreas Krause, "Efficient High Dimensional Bayesian Optimization 
	with Additivity and Quadrature Fourier Features", NIPS 2018

Namely, we implement finite basis approximation to Gaussian processes. The main contribution of this paper 
is implementation of the method embed(x) which coincides with \Phi(x) in product approximation 
	k(x,y) = \Phi(x)^\top \Phi(y)
"""

"""
Copyright (c) 2018 Mojmir Mutny, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from scipy.stats import norm
import numpy as np
import stpy.helper as helper
import torch

class Embedding():
	"""
	Base class for Embeddings to approximate kernels with a higher dimensional linear product.
	"""

	def __init__(self, gamma=0.1, nu=0.5, m=100, d=1, diameter=1.0, groups=None,
				  kernel="squared_exponential", approx = "rff"):
		"""
		Called to calculate the embedding weights (either via sampling or deterministically)

		Args:
			gamma: (positional, 0.1) bandwidth of the squared exponential kernel
			nu: (positional, 0.5) the parameter of Matern family
			m: (positional, 1)
			d: dimension of the

		Returns:
			None
		"""
		self.gamma = float(gamma)
		self.n = nu
		self.m = int(m)
		self.d = int(d)
		self.nu = nu
		self.diameter = diameter
		self.groups = groups
		self.kernel = kernel
		self.approx = approx
		self.gradient_avail = 0
		if self.m % 2 == 1:
			raise AssertionError("Number of random features has to be even.")

	def sample(self):
		"""
		Called to calculate the embedding weights (either via sampling or deterministically)

		Args:
		    None

		Returns:
			None
		"""
		raise AttributeError("Only derived classes can call this method.")

	def embed(self,x):
		"""
		Called to calculate the embedding weights (either via sampling or deterministically)

		Args:
		    x: numpy array containing the points to be embedded in the format (n,d)

		Returns:
			y: numpy array containg the embedded points (n,m), where m is the embedding dimension
		"""

		raise AttributeError("Only derived classes can call this method.")


"""
===============================
	Sampling Based Methods
===============================
"""

class RFFEmbedding(Embedding):
	"""
		Random Fourier Features emebedding
	"""

	def __init__(self, biased = False,**kwargs):
		super().__init__(**kwargs)
		self.biased = biased
		self.sample()

	def sampler(self, size):
		"""
			Defines the sampler object

		Args:
		 	size:

		Return:
		"""
		if self.kernel == "squared_exponential":
			distribution = lambda size: np.random.normal(size=size) * (1. / self.gamma)
			inv_cum_dist = lambda x: norm.ppf(x) * (1. / self.gamma)

		elif self.kernel == "laplace":
			distribution = None
			inv_cum_dist = lambda x: (np.tan(np.pi * x - np.pi) / self.gamma)

		elif self.kernel == "modified_matern":
			if self.nu == 2:
				distribution = None
				inv_cum_dist = None
				pdf = lambda x: np.prod(2*(self.gamma)/(np.power((1. + self.gamma**2*x**2),2) * np.pi),axis =1)
			elif self.nu == 3:
				distribution = None
				inv_cum_dist = None
				pdf = lambda x: np.prod((8.*self.gamma)/(np.power((1. + self.gamma**2*x**2),3) *3* np.pi),axis =1)
			elif self.nu == 4:
				distribution = None
				inv_cum_dist = None
				pdf = lambda x: np.prod((16.*self.gamma)/(np.power((1. + self.gamma**2*x**2),4) *5 * np.pi),axis =1)

		# Random Fourier Features
		if self.approx == "rff":
			if distribution == None:
				if inv_cum_dist == None:
					self.W = rejection_sampling(pdf,size = size)
				else:
					self.W = sample_custom(inv_cum_dist, size=size)
			else:
				self.W = distribution(size)

		# Quasi Fourier Features
		elif self.approx == "halton":
			if inv_cum_dist != None:
				self.W = sample_qmc_halton(inv_cum_dist, size=size)
			else:
				raise AssertionError("Inverse Cumulative Distribution could not be deduced")

		return self.W

	def sample(self):
		"""
			Samples Random Fourier Features
		"""
		self.W = self.sampler(size = (self.m,self.d))
		self.W = torch.from_numpy(self.W)
		if self.biased == True:
			self.b = 2. * np.pi * np.random.uniform(size=(self.m))
			self.bs = self.b.reshape(self.m, 1)
			self.b = torch.from_numpy(self.b)
			self.bs = torch.from_numpy(self.bs)

	def embed(self,x):
		"""
		:param x: torch array
		:return: embeded vector
		"""
		(times, d) = x.shape
		if self.biased == True:
			z = np.sqrt(2. / self.m) * torch.t(torch.cos(self.W[:, 0:d].mm(torch.t(x)) +  self.b.view(self.m, 1) ))
		else:
			z = self.W[:, 0:d].mm(torch.t(x))
			z[0:int(self.m / 2), :] = np.sqrt(2. / float(self.m)) * torch.cos(z[0:int(self.m / 2), :])
			z[int(self.m / 2):self.m, :] = np.sqrt(2. / float(self.m)) * torch.sin(z[int(self.m / 2):self.m, :])
		return torch.t(z)



"""
===============================
	Quadrature Based Methods
===============================
"""


class QuadratureEmbedding(Embedding):
	"""
		General quadrature embedding
	"""
	def __init__(self,scale=1.0,**kwargs):
		Embedding.__init__(self,**kwargs)
		self.scale = scale
		self.compute()

	def compute(self):
		"""
			Computes the tensor grid for Fourier features
		:return:
		"""
		self.q = int(np.power(self.m//2, 1. / self.d))
		self.m = self.q ** self.d

		(omegas, weights) = self.nodesAndWeights(self.q)

		self.weights = helper.cartesian([weights for weight in range(self.d)])
		self.weights = np.prod(self.weights, axis=1)

		v = [omegas for omega in range(self.d)]
		self.W = helper.cartesian(v)
		self.m = self.m * 2

		self.W = torch.from_numpy(self.W)
		self.weights = torch.from_numpy(self.weights)

	def transform(self):
		"""

		:return: spectral density of a kernel
		"""
		if self.kernel == "squared_exponential":
			p = lambda omega: np.exp(-np.sum(omega ** 2, axis=1).reshape(-1, 1) / 2 * (self.gamma ** 2)) * np.power(
				(self.gamma / np.sqrt(2 * np.pi)), 1.)*np.power(np.pi / 2,1.)

		elif self.kernel == "laplace":
			p = lambda omega: np.prod(1. / ((self.gamma ** 2) * (omega ** 2) + 1.), axis=1).reshape(-1, 1) * np.power(
				self.gamma / 2., 1.)

		elif self.kernel == "modified_matern":
			if self.nu == 2:
				p = lambda omega: np.prod(1. / ((self.gamma ** 2) * (omega ** 2) + 1.) ** self.nu, axis=1).reshape(-1,																												   1) * np.power(
					self.gamma * 1, 1.)
			elif self.nu ==3:
				p = lambda omega: np.prod(1. / ((self.gamma ** 2) * (omega ** 2) + 1.) ** self.nu, axis=1).reshape(-1,																											   1) * np.power(
					self.gamma * 4/3 , 1.)
			elif self.nu ==4:
				p = lambda omega: np.prod(1. / ((self.gamma ** 2) * (omega ** 2) + 1.) ** self.nu, axis=1).reshape(-1,																												   1) * np.power(
					self.gamma *8/5, 1.)

		return p

	def nodesAndWeights(self,q):
		"""
		Compute nodes and weights of the quadrature scheme in 1D

		:param q: degree of quadrature
		:return: tuple of (nodes, weights)
		"""

		# For osciallatory integrands even this has good properties.
		#weights = np.ones(self.q) * self.scale * np.pi / (self.q + 1)
		#omegas = (np.linspace(0, self.q - 1, self.q)) + 1
		#omegas = omegas * (np.pi / (self.q + 1))

		(omegas, weights) = np.polynomial.legendre.leggauss(q)
		omegas = ((omegas + 1.) / 2.) * np.pi
		sine_scale = (1. / (np.sin(omegas) ** 2))
		omegas = self.scale / np.tan(omegas)
		prob = self.transform()
		weights = self.scale * sine_scale *  weights * prob(omegas.reshape(-1, 1)).flatten()
		return (omegas,weights)

	def embed(self,x):
		"""
		:param x: torch array
		:return: embeding of the x
		"""
		(times, d) = tuple(x.size())
		#z = torch.from_numpy(np.zeros(shape=(self.m, times),dtype=x.dtype))
		z = torch.zeros(self.m,times, dtype = x.dtype)
		q = torch.mm(self.W[:, 0:d],torch.t(x))
		z[0:int(self.m / 2), :] = torch.sqrt(self.weights.view(-1, 1)) *  torch.cos(q)
		z[int(self.m / 2):self.m, :] = torch.sqrt(self.weights.view(-1, 1)) *  torch.sin(q)
		return torch.t(z)

class HermiteEmbedding(QuadratureEmbedding):
	"""
		Hermite Quadrature Fourier Features for squared exponential kernel
	"""

	def __init__(self,**kwargs):
		QuadratureEmbedding.__init__(self,**kwargs)
		if self.kernel != "squared_exponential":
			raise AssertionError("Hermite Embedding is allowed only with Squared Exponential Kernel")
		#self.compute()

	def nodesAndWeights(self,q):
		"""
		Compute nodes and weights of the quadrature scheme in 1D

		:param q: degree of quadrature
		:return: tuple of (nodes, weights)
		"""

		(nodes, weights) = np.polynomial.hermite.hermgauss(q)
		nodes = np.sqrt(2) * nodes / self.gamma
		#weights = np.sqrt(2) * weights / self.gamma
		weights = weights/np.sqrt(np.pi)
		return (nodes, weights)


class MaternEmbedding(QuadratureEmbedding):
	"""
		Matern specific quadrature based Fourier Features
	"""
	def __init__(self,**kwargs):
		super().__init__(**kwargs)
		if self.kernel != "modified_matern" and self.kernel !="laplace":
			raise AssertionError("Matern Embedding is allowed only with Matern Kernel")

	def nodesAndWeights(self,q):
		"""
		Compute nodes and weights of the quadrature scheme in 1D

		:param q: degree of quadrature
		:return: tuple of (nodes, weights)
		"""
		(nodes, weights) = np.polynomial.hermite.hermgauss(q)
		nodes = np.sqrt(2) * nodes / self.gamma
		weights = weights/np.sqrt(np.pi)
		return (nodes, weights)


class QuadPeriodicEmbedding(QuadratureEmbedding):
	"""
		General class implementing
	"""

	def __init__(self,**kwargs):
		super().__init__(**kwargs)

	def nodesAndWeights(self,q):
		"""
		Compute nodes and weights of the quadrature scheme in 1D

		:param q: degree of quadrature
		:return: tuple of (nodes, weights)
		"""
		weights = np.ones(self.q) * self.scale * 2 / (self.q + 1)
		omegas = (np.linspace(0, self.q - 1, self.q)) + 1
		omegas = omegas * (np.pi / (self.q + 1))

		sine_scale = (1. / (np.sin(omegas) ** 2))
		omegas = self.scale / np.tan(omegas)
		prob = self.transform()
		weights = self.scale * sine_scale *  weights * prob(omegas.reshape(-1, 1)).flatten()
		return (omegas,weights)



class KLEmbedding(QuadratureEmbedding):
	"""
		General class implementing Karhunen-Loeve expansion
	"""
	def __init__(self,**kwargs):
		super().__init__(**kwargs)


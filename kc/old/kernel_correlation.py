# coding: utf-8
import math
import numpy
import scipy.linalg
import scipy.optimize


class KernelCorrelation(object):
	"""implementation of the Kernel Correlation algorithm for point set registration"""
	def __init__(self, model, scene):
		super(KernelCorrelation, self).__init__()
		self.model = model
		self.scene = scene
		self.transform_shape = (self.scene.shape[1], self.model.shape[1])
		# n(n-1)/2 - (n-m)(n-m-1)/2
		self.transform_dof = int(self.transform_shape[0]*0.5*(2*self.transform_shape[1] - self.transform_shape[0] - 1))
		# For the construction of the orthogonal matrix
		self.transform_tril_ind = [pair for pair in zip(*numpy.tril_indices(self.transform_shape[1], -1)) if pair[1] < self.transform_shape[0]]
		self.transform_tril_ind = tuple([numpy.array(tup) for tup in zip(*self.transform_tril_ind)])


	def slow_cost_fun(self, transform):
		def kernel_correlation(xi, xj):
			variance = 1
			diff = xi - xj
			# leave out the math.pow(2*math.pi*variance, -D/2) coefficient
			return math.exp(-diff.dot(diff)/(2*variance))
		# rigid case!
		cost = -sum(kernel_correlation(s, transform.dot(m)) for s in self.scene for m in self.model)
		return cost/float(self.model.shape[0])


	def get_transform(self, params):
		"""Obtain an orthogonal rectangular matrix parametrized by a number of params"""
		t = numpy.eye(*self.transform_shape).T
		t[self.transform_tril_ind] = params
		t = scipy.linalg.qr(t, overwrite_a=True, mode='economic')[0]
		return t.T


	counter = 0
	def cost_fun(self, params):
		cost = self.mat_cost_fun(self.get_transform(params))
		self.counter += 1
		if self.counter % 1000 == 0:
			print "Iteration " + str(self.counter) + ": " + str(cost)
		return cost


	def mat_cost_fun(self, transform, sig_sq_2=2):
		"""Calculate the cost for the given transformation"""		
		# construct the matrix with all combinations of points in the scene
		# and transformed points in the model, having shape (n, n, scenedim)
		m_transformed = transform.dot(self.model.T).T
		cost = 0
 		for s in self.scene:
 			# array broadcasting trick - subtracting 2D and 1D array
 			# exp works elementwise on all array elements
 			cost = cost - numpy.exp(-numpy.sum((s - m_transformed)**2,axis=-1)/sig_sq_2).sum()
 		# leave out the coefficient
		return cost/float(self.model.shape[0])


	def find_minimum(self, x0=None):
		if x0 is None:
			x0 = numpy.zeros((self.scene.shape[1], self.model.shape[1]))
		# TODO: Test method='Anneal' or 'Powell'
		result = scipy.optimize.minimize(
			self.cost_fun,
			x0,
			method='Powell',
			tol=1e-8,
			options={'maxiter': 1000000, 'maxfev':1000000, 'disp': True})
		return result.x


	def find_maximum(self, x0=None):
		if x0 is None:
			x0 = numpy.zeros((self.scene.shape[1], self.model.shape[1]))
		# TODO: Test method='Anneal' or 'Powell'
		result = scipy.optimize.minimize(
			(lambda x: -self.cost_fun(x)),
			x0,
			method='Nelder-Mead',
			tol=1e-8,
			options={'maxiter': 1000, 'disp': True})
		return result.x
# http://www.cs.indiana.edu/pub/hanson/Siggraph01QuatCourse/ggndrot.pdf
# http://www.cs.cmu.edu/~ytsin/KCReg/
# coding: utf-8
import math
import numpy
import scipy.linalg
import scipy.optimize

from registration import Registration

try:
	# try to compile and use the faster cython version
	import pyximport
	pyximport.install(setup_args={"include_dirs": numpy.get_include()})
	from kernel_correlation_inner import st_get_transform, st_mat_cost_fun
except:
	# failed... fall back to plain numpy
	FAST_VERSION = -1
	
	def st_get_transform(params, transform_shape, tril_ind):
		"""Obtain an orthogonal rectangular matrix parametrized by a number of params"""
		t = numpy.eye(*transform_shape).T
		t[tril_ind] = params
		t = scipy.linalg.qr(t, overwrite_a=True, mode='economic')[0]
		return t.T

	def st_mat_cost_fun(model, scene, transform, sig_sq_2=2):
		"""Calculate the cost for the given transformation"""		
		# construct the matrix with all combinations of points in the scene
		# and transformed points in the model, having shape (n, n, scenedim)
		m_transformed = transform.dot(model.T).T
		cost = 0
		for s in scene:
			# array broadcasting trick - subtracting 2D and 1D array
			# exp works elementwise on all array elements
			cost = cost - numpy.exp(-numpy.sum((s - m_transformed)**2,axis=-1)/sig_sq_2).sum()
		# leave out the coefficient
		return cost/float(model.shape[0])

	# ---------------------------------------------


class KernelCorrelation(Registration):
	"""Implementation of the Kernel Correlation algorithm for point set registration"""
	def __init__(self, model, scene):
		super(KernelCorrelation, self).__init__(model, scene)
		self.transform_dof = self.transform_shape[0] * self.transform_shape[1]


	def slow_cost_fun(self, transform):
		def kernel_correlation(xi, xj):
			variance = 1
			diff = xi - xj
			# leave out the math.pow(2*math.pi*variance, -D/2) coefficient
			return math.exp(-diff.dot(diff)/(2*variance))
		# rigid case!
		cost = -sum(kernel_correlation(s, transform.dot(m)) for s in self.scene for m in self.model)
		return cost/float(self.model.shape[0])


	# wrapper for the fast Cython function
	def cost_fun(self, transform, sig_sq_2=2):
		return st_mat_cost_fun(self.model, self.scene, transform, sig_sq_2)


	counter = 0
	def cost_fun(self, params, sig_sq_2=1):
		cost = st_mat_cost_fun(
			self.model,
			self.scene,
			params.reshape(self.transform_shape),
			sig_sq_2)
		if self.counter % 1000 == 0:
			print "Iteration " + str(self.counter) + ": " + str(cost)
		self.counter += 1
		return cost


	def find_minimum(self, x0=None, maxi=100000):
		if x0 is None:
			x0 = numpy.zeros((self.transform_dof,))
		result = scipy.optimize.minimize(
			self.cost_fun,
			x0,
			method='Powell',
			tol=1e-8,
			options={'maxiter': maxi, 'maxfev': maxi, 'disp': True})
		if self.transform_dof == self.transform_shape[1] * self.transform_shape[0]:
			return result.x.reshape(self.transform_shape)
		return result.x



class RigidKernelCorrelation(KernelCorrelation):
	"""implementation of the Kernel Correlation algorithm for point set registration"""
	def __init__(self, model, scene):
		super(KernelCorrelation, self).__init__(model, scene)
		# n(n-1)/2 - (n-m)(n-m-1)/2
		self.transform_dof = int(self.transform_shape[0]*0.5*(2*self.transform_shape[1] - self.transform_shape[0] - 1))
		# For the construction of the orthogonal matrix
		self.transform_tril_ind = [pair for pair in zip(*numpy.tril_indices(self.transform_shape[1], -1)) if pair[1] < self.transform_shape[0]]
		self.transform_tril_ind = tuple([numpy.array(tup) for tup in zip(*self.transform_tril_ind)])


	def get_transform(self, params):
		return st_get_transform(params, self.transform_shape, self.transform_tril_ind)

	def cost_fun(self, params, sig_sq_2=1):
		cost = st_mat_cost_fun(
			self.model,
			self.scene,
			st_get_transform(params, self.transform_shape, self.transform_tril_ind),
			sig_sq_2)
		if self.counter % 1000 == 0:
			print "Iteration " + str(self.counter) + ": " + str(cost)
		self.counter += 1
		return cost

	def find_minimum(self, x0=None, maxi=100000):
		x = super(RigidKernelCorrelation, self).find_minimum(x0=x0, maxi=maxi)
		if self.transform_dof == 1:
			x = numpy.array([x])
		return self.get_transform(x)
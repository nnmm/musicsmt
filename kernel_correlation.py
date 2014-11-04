# coding: utf-8
import math
import numpy
import scipy.linalg
import scipy.optimize as opt


class KernelCorrelation(object):
	"""implementation of the Kernel Correlation algorithm for point set registration"""
	def __init__(self, model, scene):
		super(KernelCorrelation, self).__init__()
		self.model = model
		self.scene = scene


	def find_minimum(self):
		x0 = numpy.zeros((self.scene.shape[0], self.model.shape[0]))
		opt.minimize(self.cost_fun, x0)
		return result.x


	# calculate once and save
	def p_S(self, x):
		N = len(self.scene)
		# this iterates through the rows of scene
		sum(kernel(x, point) for point in self.scene)/N


	def p_M(self, x, transform):
		N = len(self.scene)
		# this iterates through the rows of scene
		sum(kernel(x, transform.dot(point)) for point in self.model)/N
	

	def kernel_correlation(self, xi, xj):
		# D = float(x.shape[0])
		D = float(xi.shape[0])
		variance = 1
		diff = xi - xj
		return math.pow(2*math.pi*variance, -D/2)*math.exp(-diff.dot(diff)/(2*variance))


	def kernel(self, x, xi):
		# D = float(x.shape[0])
		D = float(x.shape[0])
		variance = 1
		diff = x - xi
		return math.pow(math.pi*variance, -D/2)*math.exp(-diff.dot(diff)/variance)


	def slow_cost_fun(self, transform):
		# rigid case!
		cost = -sum(self.kernel_correlation(s, transform.dot(m)) for s in self.scene for m in self.model)
		print cost
		return cost


	def fast_cost_fun(self, transform, sigma = 1):
		D = float(self.scene.shape[1])
		sig_sq_2 = 2*sigma**2
		coeff = (math.pi*sig_sq_2)**(-D/2)
		# construct the matrix with all combinations of points in the scene
		# and transformed points in the model, having shape (n, n, scenedim)
		m_transformed = transform.dot(self.model.T).T
		cost = 0
 		for s in self.scene:
 			# array broadcasting trick - subtracting 2D and 1D array
 			# exp works elementwise on all array elements
 			cost = cost - numpy.exp(-numpy.sum((s - m_transformed)**2,axis=-1)/sig_sq_2).sum()
		print coeff*cost
		return coeff*cost


# TODO: Investigate in 2D why/when results from fast_cost_fun and cost_fun differ
# TODO: Find good value for sigma
# TODO: What are the limits in terms of array size on my computer?
import unittest
import kernel_correlation
import numpy
import scipy

class KernelCorrelationTest(unittest.TestCase):
	"""Testing suite for Kernel Correlation"""

	def setUp(self):
		def rand_rot(n):
		    """Return a random rotation

		    Return a random orthogonal matrix with determinant 1"""
		    q, _ = scipy.linalg.qr(numpy.random.randn(n, n))
		    if scipy.linalg.det(q) < 0:
		        q[:, 0] = -q[:, 0]
		    return q

		self.model = numpy.random.rand(1000, 200)
		self.transf = rand_rot(200)
		self.scene = self.transf.dot(self.model.T).T
		self.cost_test_mat = rand_rot(200)
		self.reg = kernel_correlation.KernelCorrelation(self.model, self.scene)


	def test_vec_minimization(self):
		def toy_cost_fun(x):
			return numpy.linalg.norm(numpy.array([[1, 2], [3, 4]]).dot(x))
		x0 = numpy.ones(2)
		found_minimum = scipy.optimize.minimize(toy_cost_fun, x0, method='Nelder-Mead', tol=1e-8, options={'disp': True}).x
		true_minimum = numpy.array([[0], [0]])
		print "The found minimum is " + str(found_minimum)
		self.assertTrue(numpy.allclose(found_minimum, true_minimum), 'Did not find minimizing vector.')


	def test_both_cost_functions(self):
		self.assertAlmostEqual(
			self.reg.slow_cost_fun(self.cost_test_mat),
			self.reg.fast_cost_fun(self.cost_test_mat))


	def test_cost_function(self):
		pass

if __name__ == '__main__':
	unittest.main()
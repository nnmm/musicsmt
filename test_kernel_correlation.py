import unittest
import kernel_correlation
import numpy
import scipy
import cProfile
import pstats

class KernelCorrelationTest(unittest.TestCase):
	"""Testing suite for Kernel Correlation"""

	def setUp(self):
		print "=== New Test ==="
		self.pr = cProfile.Profile()
		self.pr.enable()

		self.mdim = 8
		self.sdim = 4
		self.num_vectors = 100

		self.correct_transform = numpy.random.rand(self.sdim, self.mdim)
		model = numpy.random.rand(self.num_vectors, self.mdim)
		scene = self.correct_transform.dot(model.T).T
		self.reg = kernel_correlation.KernelCorrelation(model, scene)


	def print_profile(self):
		with open('kcstats', 'w') as stream:
			p = pstats.Stats(self.pr, stream=stream)
			p.strip_dirs()
			p.sort_stats('cumtime')
			p.print_stats()


	@staticmethod
	def rand_rot(n):
		"""Return a random rotation

		Return a random orthogonal matrix with determinant 1"""
		q, _ = scipy.linalg.qr(numpy.random.randn(n, n))
		if scipy.linalg.det(q) < 0:
			q[:, 0] = -q[:, 0]
		return q


	# ---------------------------------------------------------------------------------


	def test_vec_minimization(self):
		"""Test if the correct minimum for ||A*x|| is found, where x is variable"""
		def toy_cost_fun(x):
			return numpy.linalg.norm(numpy.array([[1, 2], [3, 4]]).dot(x))
		x0 = numpy.ones(2)
		found_minimum = scipy.optimize.minimize(toy_cost_fun, x0, method='Nelder-Mead', tol=1e-8).x
		true_minimum = numpy.array([[0], [0]])
		self.assertTrue(numpy.allclose(found_minimum, true_minimum), 'Did not find minimizing vector.')


	def test_ident_cost(self):
		"""Test if the identity matrix returns a reasonable result for equal matrices"""
		print "test_ident_cost"
		self.reg.model = numpy.random.rand(self.num_vectors, self.mdim)
		self.reg.scene = self.reg.model.copy()
		self.assertTrue(-self.num_vectors < self.reg.mat_cost_fun(numpy.identity(self.mdim)) < -1, 'Cost not in the expected range.')


	def test_parametrization(self):
		"""Test if the matrix we create by our parametrization is orthogonal"""
		print "test_parametrization"
		params = numpy.random.rand(self.reg.transform_dof)
		t = self.reg.get_transform(params)
		self.assertTrue(numpy.allclose(t.dot(t.T), numpy.identity(t.shape[0])))


	def test_different_minimizers(self):
		print "test_different_minimizers"
		numpy.random.seed(0)
		# 35 for Nelder-Mead
		scalefactor = 10
		model = numpy.random.rand(self.num_vectors, self.mdim*scalefactor)
		scene = numpy.random.rand(self.num_vectors, self.sdim*scalefactor)
		print "model: "
		print model
		print "scene: "
		print scene
		self.reg = kernel_correlation.KernelCorrelation(model, scene)
		found_minimum = self.reg.find_minimum(x0=numpy.zeros((self.reg.transform_dof,)))
		print found_minimum
		self.assertTrue(numpy.all(found_minimum == numpy.zeros((self.reg.transform_dof,))))
		self.print_profile()


	@unittest.skip("later alligator")
	def test_min_idemp_ident(self):
		"""Test if the correct transformation is a minimum of the cost function"""
		model = numpy.random.rand(self.num_vectors, self.mdim)
		self.reg = kernel_correlation.KernelCorrelation(
			# the model
			model,
			# the scene
			model.copy())
		found_minimum = self.reg.find_minimum(x0=numpy.zeros((self.reg.transform_dof,)))
		print "test_min_idemp_ident found"
		print found_minimum
		self.assertTrue(numpy.all(found_minimum == numpy.zeros((self.reg.transform_dof,))))
		self.print_profile()


	@unittest.skip("Doesn't work in the general case")
	def test_ident_recovery(self):
		model = numpy.random.rand(self.num_vectors, self.mdim)
		self.reg = kernel_correlation.KernelCorrelation(
			# the model
			model,
			# the scene
			model.copy)
		self.reg.scene = self.reg.model.copy()
		# only a bit of noise
		x0 = 0.001*numpy.random.rand(self.mdim, self.mdim)
		found_minimum = self.reg.find_minimum(x0=x0)
		self.assertTrue(found_minimum == numpy.identity(self.mdim))


	@unittest.skip("demonstrating skipping")
	def test_min_idemp_rigid(self):
		self.reg.model = numpy.random.rand(self.num_vectors, self.mdim)
		self.correct_transform = self.rand_rot(self.mdim)
		self.reg.scene = self.correct_transform.dot(self.model.T).T
		found_minimum = self.reg.find_minimum(x0=numpy.identity(self.mdim)).reshape(self.mdim, -1)
		print "test_min_idemp_rigid found"
		print found_minimum
		self.assertTrue(numpy.all(found_minimum == self.correct_transform))
		self.print_profile()


	@unittest.skip("save time")
	def test_cost_minimization(self):
		print self.reg.find_minimum()
		self.assertTrue(True)
		# self.assertTrue(self.reg.mat_cost_fun(self.reg.find_minimum()) < self.reg.mat_cost_fun(self.cost_test_mat))

if __name__ == '__main__':
	unittest.main()
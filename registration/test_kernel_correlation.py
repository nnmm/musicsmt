import unittest
from kernel_correlation import KernelCorrelation,  RigidKernelCorrelation
import numpy
import scipy
import cProfile
import pstats

class KernelCorrelationTest(unittest.TestCase):
	"""Testing suite for Kernel Correlation"""

	def setUp(self):
		print "==== " + self.id().split('.')[-1] + " ===="
		self.pr = cProfile.Profile()
		self.pr.enable()

		self.mdim = 4
		self.sdim = 4
		self.num_vectors = 100

		self.correct_transform = numpy.random.rand(self.sdim, self.mdim)
		model = self.normed_rand(self.num_vectors, self.mdim)
		scene = self.correct_transform.dot(model.T).T
		self.reg = KernelCorrelation(model, scene)


	def tearDown(self):
		print "\n"


	def print_profile(self):
		with open('kcstats', 'w') as stream:
			p = pstats.Stats(self.pr, stream=stream)
			p.strip_dirs()
			p.sort_stats('cumtime')
			p.print_stats()


	@staticmethod
	def rand_rot(n):
		"""Return a random orthogonal matrix with determinant 1"""
		q, _ = scipy.linalg.qr(numpy.random.randn(n, n))
		if scipy.linalg.det(q) < 0:
			q[:, 0] = -q[:, 0]
		return q


	def normed_rand(self, m, n):
		"""Return a random matrix of unit vectors"""
		mat = numpy.random.rand(m, n)
		norms = numpy.apply_along_axis(numpy.linalg.norm, 0, mat)
		return mat/norms

	def print_minim_info(self, x0, found_minimum):
		print "x0: "
		print x0
		print "cost x0: "
		print self.reg.cost_fun(x0)
		print "found_minimum: "
		print found_minimum
		print "cost found_minimum: "
		print KernelCorrelation.cost_fun(self.reg, found_minimum)

	# ---------------------------------------------


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
		self.reg.model = self.normed_rand(m=self.num_vectors, n=self.mdim)
		self.reg.scene = self.reg.model.copy()
		self.assertTrue(-self.num_vectors < self.reg.cost_fun(numpy.identity(self.mdim)) < -1, 'Cost not in the expected range.')


	def test_parametrization(self):
		"""Test if the matrix we create by our parametrization is orthogonal"""
		self.reg = RigidKernelCorrelation(self.reg.model, self.reg.model.copy())
		params = numpy.random.rand(self.reg.transform_dof)
		t = self.reg.get_transform(params)
		print t.dot(t.T)
		self.assertTrue(numpy.allclose(t.dot(t.T), numpy.identity(t.shape[0])))

	
	def test_min_idemp_ident(self):
		"""Test if the correct transformation is a minimum of the cost function"""
		model = self.normed_rand(m=self.num_vectors, n=self.mdim)
		self.reg = RigidKernelCorrelation(model, model.copy())
		x0=numpy.zeros((self.reg.transform_dof,))
		found_minimum = self.reg.find_minimum(x0=x0)
		self.print_minim_info(x0, found_minimum)
		self.assertTrue(numpy.allclose(found_minimum, numpy.identity(found_minimum.shape[0])))


	def test_miniature_corpus(self):
		"""Output a rigid transform between models of two miniature corpora."""
		m996 = numpy.array([[-0.02298838, -0.99973571], [ 0.26926005,  0.96306747], [-0.63116926, -0.77564514], [-0.71778536,  0.69626445]])
		m6759 = numpy.array([[ 0.79119831,  0.61155969], [-0.99854398,  0.05394306], [ 0.72569704, -0.68801433], [-0.71835101, -0.6956808 ]])
		self.reg = RigidKernelCorrelation(m996, m6759)
		x0=numpy.zeros((self.reg.transform_dof,))
		found_minimum = self.reg.find_minimum(x0=x0)
		print "found_minimum: "
		print found_minimum
		self.print_minim_info(x0, found_minimum)


	def test_miniature_corpus_nonrigid(self):
		"""Output a nonrigid transform between models of two miniature corpora."""
		m996 = numpy.array([[-0.02298838, -0.99973571], [ 0.26926005,  0.96306747], [-0.63116926, -0.77564514], [-0.71778536,  0.69626445]])
		m6759 = numpy.array([[ 0.79119831,  0.61155969], [-0.99854398,  0.05394306], [ 0.72569704, -0.68801433], [-0.71835101, -0.6956808 ]])
		self.reg = KernelCorrelation(m996, m6759)
		x0=numpy.zeros((self.reg.scene.shape[1], self.reg.model.shape[1]))
		found_minimum = self.reg.find_minimum(x0=x0)
		print "found_minimum: "
		print found_minimum
		print "cost found_minimum: "
		print self.reg.cost_fun(found_minimum)



	def test_general_matrix_minimization(self):
		"""Test whether the result returned by unconstrained optimization is sensible"""
		x0 = numpy.zeros((self.reg.scene.shape[1], self.reg.model.shape[1]))
		found_minimum = self.reg.find_minimum(x0=x0)
		print "correct_transform: "
		print self.correct_transform
		print "cost correct_transform: "
		print self.reg.cost_fun(self.correct_transform)
		print "found_minimum: "
		print found_minimum
		print "cost found_minimum: "
		print self.reg.cost_fun(found_minimum)


	def test_minimum_better_than_random(self):
		"""Test if for a random transform, the cost that is found after a few iterations is lower than the initial cost"""
		numpy.random.seed(0)
		scalefactor = 2
		model = self.normed_rand(self.num_vectors, self.mdim*scalefactor)
		scene = self.normed_rand(self.num_vectors, self.mdim*scalefactor)
		self.reg = RigidKernelCorrelation(model, scene)
		x0=numpy.random.rand(self.reg.transform_dof)
		found_minimum = self.reg.find_minimum(x0=x0)
		self.print_minim_info(x0, found_minimum)
		self.assertTrue(KernelCorrelation.cost_fun(self.reg, found_minimum) < self.reg.cost_fun(x0))


	# ---------------------------------------------

	@unittest.skip("Nope")
	def test_ident_recovery(self):
		"""Test if the correct (identity) transform is found, starting from a very close x0"""
		model = self.normed_rand(m=self.num_vectors, n=self.mdim)
		self.reg = RigidKernelCorrelation(
			# the model
			model,
			# the scene
			model.copy())
		# only a bit of noise
		x0 = 0.001*numpy.random.rand(self.reg.transform_dof)
		found_minimum = self.reg.find_minimum(x0=x0)
		self.print_minim_info(x0, found_minimum)
		self.assertTrue(numpy.allclose(found_minimum, numpy.zeros(self.reg.transform_shape)))


if __name__ == '__main__':
	unittest.main()
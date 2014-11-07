import cython
import numpy
cimport numpy
import scipy

from cpython cimport PyCObject_AsVoidPtr
from scipy.linalg.blas import fblas

REAL = numpy.float64
ctypedef numpy.float64_t REAL_t

ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)


def st_get_transform(numpy.ndarray[REAL_t, ndim=1] params, transform_shape, tril_ind):
	"""Obtain an orthogonal rectangular matrix parametrized by a number of params"""
	cdef numpy.ndarray[REAL_t, ndim=2] t = numpy.eye(*transform_shape).T
	t[tril_ind] = params
	t = scipy.linalg.qr(t, overwrite_a=True, mode='economic')[0]
	return t.T

def st_mat_cost_fun(
	numpy.ndarray[REAL_t, ndim=2] model,
	numpy.ndarray[REAL_t, ndim=2] scene,
	numpy.ndarray[REAL_t, ndim=2] transform,
	sig_sq_2=2):
	"""Calculate the cost for the given transformation"""
	# construct the matrix with all combinations of points in the scene
	# and transformed points in the model, having shape (n, n, scenedim)
	cdef numpy.ndarray[REAL_t, ndim=2] m_transformed = transform.dot(model.T).T
	cdef int i_s, i_m
	cdef double cost = 0.0
	for i_s in range(scene.shape[1]):
		for i_m in range(model.shape[1]):
			pass
	for s in scene:
		# array broadcasting trick - subtracting 2D and 1D array
		# exp works elementwise on all array elements
		cost = cost - numpy.exp(-numpy.sum((s - m_transformed)**2,axis=-1)/sig_sq_2).sum()
	# leave out the coefficient
	return cost/float(model.shape[0])

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0
cdef float *x = [<float>10.0]
cdef float *y = [<float>0.01]
cdef float expected = <float>0.1
cdef int size = 1
cdef double d_res
d_res = dsdot(&size, x, &ONE, y, &ONE)
cdef float *p_res
p_res = <float *>&d_res
if (abs(d_res - expected) < 0.0001):
	print "double"
elif (abs(p_res[0] - expected) < 0.0001):
	print "float"
else:
	print "neither"
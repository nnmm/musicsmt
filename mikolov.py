# coding: utf-8
import logging
import config
from music2vec import *
from numpy import amax, empty_like
from numpy.random import rand
from scipy.linalg import block_diag, lstsq
from scipy.sparse.linalg import cg

logging.basicConfig(forX='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = Music2Vec.load(config.model_dir + "enwiki300M")

# the cols of X are the word vectors
X = model.syn0norm.T

# the transformation we want to recover
ldim = 100
A_true = X[:ldim,:200]
A_true = A_true/amax(A_true)

numcols = 20000
# X.shape = (200, numcols)
X = X[:,:numcols]

# Y.shape = (ldim, numcols)
Y = A_true.dot(X)

# now we're looking at the transposes: ||X^T A^T - Y^T||^2
# the dimensions are (numcols, 200) (200, ldim) - (numcols, ldim)
# since this is in the Frobenius norm, it's equivalent to the following 2-Norm:
# we write the columns of A^T in a vector a so that we have ||C a - b||^2
# the dimensions are (ldim*numcols, ldim*200) (ldim*200) - (ldim*numcols)

# -----------------------------------------------------------------------
# # we need to call Y.reshape to obtain a vector that strings together the columns of Y.T
# b = Y.reshape(reduce(lambda x, y: x*y, Y.shape))
# actual_a = A_true.reshape(reduce(lambda x, y: x*y, A_true.shape))
# C = block_diag(*X.T.reshape((1,) + X.T.shape).repeat(ldim, axis=0))
# # C.dot(actual_a) - b == 0 

# # The solution is given as C^T C a = C^T b, where a is the unrolled version of A_true.T
# # C^T C, left hand side
# CTC = C.T.dot(C)
# # C^T b, right hand side
# CTb = C.T.dot(b)

# #del model, Y, b, C, X
# a = cg(CTC, CTb)[0]
# initial_guess = actual_a + 0.01*rand(*actual_a.shape)
# a_ig = cg(CTC, CTb, x0=initial_guess)[0]

# a_l = lstsq(C, b)[0]

# A_whole = a.reshape(A_true.T.shape).T

#
A_decomp = empty_like(A_true)
for ii in xrange(0,ldim):
	A_decomp[ii, :] = lstsq(X.T, Y.T[:,ii])[0]
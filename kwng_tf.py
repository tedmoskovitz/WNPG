import tensorflow as tf
import numpy as np
import pdb


class KWNG(object):

	def __init__(self, kernel, num_basis=5, eps=1e-5, with_diag_mat=True):
		
		self.kernel = kernel	
		self.eps = tf.Variable(eps, trainable=False, dtype=tf.float32)
		self.thresh = 0.
		self.num_basis = num_basis
		self.with_diag_mat = with_diag_mat
		self.K = None
		self.T = None

	def compute_cond_matrix(self, net, outputs):
		
		L,d = outputs.get_shape().as_list()
		idx = tf.random.shuffle(tf.constant(np.arange(L)))
		
		outputs = tf.reshape(outputs, [L, -1])
		basis = tf.gather(outputs, idx[:self.num_basis])
		basis = tf.stop_gradient(tf.identity(basis))
		
		mask_int = tf.random.uniform([self.num_basis], minval=0, maxval=d, dtype=tf.int64)
		
		mask = tf.one_hot(mask_int, d) 
		mask = tf.cast(mask, outputs.dtype)

		###
		sigma = tf.log(tf.reduce_mean(self.kernel.square_dist(basis,outputs)))#.clone().detach()
		sigma = tf.stop_gradient(tf.identity(sigma))
		#print(" sigma:   " + str(tr.exp(sigma).item()))
		sigma /= np.log(10.)

		if hasattr(self.kernel, 'params_0'):
			self.kernel.params = self.kernel.params_0 + sigma#0.5 * (self.kernel.params_0 + sigma) #self.kernel.params_0 + sigma

		dkdxdy, dkdx, _= self.kernel.dkdxdy(basis, outputs, mask=mask)
		self.K = (1./L) * tf.einsum('mni,kni->mk', dkdxdy, dkdxdy)
		aux_loss = tf.reduce_mean(dkdx, axis=1)
		self.T = self.compute_jacobian(aux_loss, net)

	def compute_natural_gradient(self, g):
		### VERIFIED ###
		
		ss, uu, vv = tf.linalg.svd(tf.cast(self.K, g.dtype)) #.double()

		ss_inv, mask = self.pseudo_inverse(ss)
		ss_inv = tf.sqrt(ss_inv)
		vv = tf.einsum('i,ji->ij', ss_inv, vv)
		self.T = tf.einsum('ij,jk->ik', tf.cast(vv, g.dtype), tf.cast(self.T, g.dtype))
		cond_g, G, D = self.make_system(g, mask)
		cond_g_copy1 = tf.identity(cond_g)
		cond_g_copy2 = tf.identity(cond_g)

		try:
			U = tf.linalg.cholesky(G)
			# some differences between torch and tf chol solve, keep in mind
			cond_g = tf.squeeze(tf.linalg.cholesky_solve(U, tf.expand_dims(cond_g, axis=-1)), axis=-1)
			#
		except:
			try:
				# also flipped ordering for tf solve compared to torch
				cond_g = tf.squeeze(tf.linalg.solve(G, tf.expand_dims(cond_g_copy1, axis=-1)), axis=-1)
			except:
				pinv = pinverse(G)
				cond_g = tf.einsum('mk,k', pinv, cond_g_copy2)
		cond_g = tf.einsum('md,m->d', self.T, cond_g)
		cond_g = (g - cond_g) / tf.cast(self.eps, g.dtype)
		cond_g = D * cond_g
		return tf.cast(cond_g, tf.float32)

	def make_system(self, g, mask):
		### VERIFIED ###
		T = self.T #tf.cast(self.T, tf.float64)
		if self.with_diag_mat == 1:
			#T = tf.cast(self.T, tf.float64)
			D = tf.sqrt(tf.reduce_sum(T * T, axis=0))
			D = 1. / (D + 1e-8)
		elif self.with_diag_mat == 0:
			D = tf.ones(self.T.get_shape().as_list()[1], dtype=T.dtype) 

		cond_g = D * g
		cond_g = tf.einsum('md,d->m', T, cond_g)
		
		P = tf.cast(mask, T.dtype)
		G =  tf.einsum('md,d,kd->mk', T, D, T) + tf.cast(self.eps, T.dtype) * tf.diag(P)
		return cond_g, G, D

	def pseudo_inverse(self, S):
		### VERIFIED ###
		SS = 1. / S
		#mask = tf.cast(S <= self.thresh, tf.float32)
		mask = tf.cast(S > self.thresh, SS.dtype)
		SS *= mask # set all elments <= thresh to 0 
		#mask = (S > self.thresh)
		return SS, tf.cast(mask, tf.bool) 

	def compute_jacobian(self, loss, net, scope="model"):
		### VERIFIED ###
		# loss is [n_basis]
		J = []
		b_size = loss.get_shape().as_list()[0]
		for i in range(b_size):
			grads = tf.gradients(loss[i], tf.trainable_variables(scope=scope))
			grads = [tf.reshape(x, [-1]) for x in grads]
			grads = tf.concat(grads, axis=0)
			J.append(grads)

		return tf.stack(J, axis=0)


def pinverse(matrix):

	"""Returns the Moore-Penrose pseudo-inverse"""

	s, u, v = tf.svd(matrix)

	threshold = tf.reduce_max(s) * 1e-5
	s_mask = tf.boolean_mask(s, s > threshold)
	s_inv = tf.diag(tf.concat([1. / s_mask, tf.zeros([tf.size(s) - tf.size(s_mask)])], 0))

	return tf.matmul(v, tf.matmul(s_inv, tf.transpose(u)))


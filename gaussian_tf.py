import tensorflow as tf


class Gaussian(object):
	def __init__(self, D, log_sigma, dtype=tf.float32, device='cpu'):
		self.D = D
		self.params = log_sigma
		self.dtype = dtype
		self.device = device
		self.adaptive=  False
		self.params_0 = log_sigma  

	def get_exp_params(self):
		### verified ###s
		return pow_10(self.params, dtype= self.dtype, device = self.device)

	def update_params(self,log_sigma):
		### verified ###
		self.params = log_sigma


	def square_dist(self, X, Y):
		### verified ###
		# Squared distance matrix of pariwise elements in X and basis
		# Inputs:
		# X 	: N by d matrix of data points
		# basis : M by d matrix of basis points
		# output: N by M matrix

		return self._square_dist(X, Y)

	def kernel(self, X,Y):
		### verified ###
		# Gramm matrix between vectors X and basis
		# Inputs:
		# X 	: N by d matrix of data points
		# Y     : M by d matrix of basis points
		# output: N by M matrix

		return self._kernel(self.params, X, Y)

	def dkdxdy(self, X, Y, mask=None):
		return self._dkdxdy(self.params, X, Y, mask=mask)
# Private functions 

	def _square_dist(self,X, Y):
		### verified ###
		n_x, d = X.get_shape().as_list()
		n_y, d = Y.get_shape().as_list()
		dist = -2*tf.einsum('mr,nr->mn', X, Y) + tf.tile(tf.expand_dims(tf.reduce_sum(X**2, axis=1), -1), (1,n_y)) +\
			tf.tile(tf.expand_dims(tf.reduce_sum(Y**2, axis=1), 0), (n_x,1)) #  tr.einsum('m,n->mn', tr.ones([ n_x],dtype=self.dtype, device = self.device),tr.sum(Y**2,1)) 

		return dist 

	def _kernel(self, log_sigma, X, Y):
		### verified ###
		N, d = X.get_shape().as_list()
		sigma = pow_10(log_sigma, dtype=self.dtype, device=self.device)
		tmp = self._square_dist(X, Y)
		dist = tf.cast(tf.maximum(tmp, tf.zeros_like(tmp)), self.dtype)
		if self.adaptive:
			ss = tf.stop_gradient(tf.identity(tf.reduce_mean(dist))) #.clone().detach() # might not want stop gradient here
			dist = dist / (ss+1e-5)
		return  tf.exp(-0.5 * dist / sigma)


	def _dkdxdy(self,log_sigma,X,Y,mask=None):
		# X : [M,D]
		# Y : [N,D]

		# dkdxdy ,   dkdxdy2  = [M,N,D, D]  
		# dkdx = [M,N,D]

		# mask: [M, D]
		N, d = X.get_shape().as_list()
		print (X.get_shape().as_list(), Y.get_shape().as_list(), mask.get_shape().as_list())
		sigma = pow_10(log_sigma, dtype=self.dtype, device=self.device)
		gram = self._kernel(log_sigma, X, Y)

		D = tf.cast((tf.expand_dims(X, 1) - tf.expand_dims(Y, 0)), self.dtype) / sigma
		 
		I  = tf.ones(D.get_shape().as_list()[-1], dtype=self.dtype) / sigma #, device=self.device

		dkdy = tf.einsum('mn,mnr->mnr', gram,D)
		dkdx = -dkdy

		if mask is None:
			D2 = tf.einsum('mnt,mnr->mntr', D, D)
			I  = tf.eye(D.get_shape().as_list()[-1], dtype=self.dtype) / sigma #, device=self.device
			dkdxdy = I - D2
			dkdxdy = tf.einsum('mn, mntr->mntr', gram, dkdxdy)
		else:
			D_masked = tf.einsum('mnt,mt->mn', D, mask)
			D2 = tf.einsum('mn,mnr->mnr', D_masked, D)

			dkdxdy =  tf.einsum('mn,mr->mnr', gram, mask) / sigma - tf.einsum('mn, mnr->mnr', gram, D2)
			dkdx = tf.einsum('mnt,mt->mn', dkdx, mask)

		return dkdxdy, dkdx, gram




def pow_10(x, dtype=tf.float32, device='cpu'): 
	### verified ###
	return tf.pow(tf.constant(10.), x) #tf.pow(tf.Tensor([10.], dtype=dtype), x) #, device=device



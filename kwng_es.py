import torch as tr
import torch.nn as nn
import numpy as np
import pdb



def get_flat_grad(net):
	grads = []
	for param in net.parameters():
		grads.append(param.grad.view(-1))
	flat_grad = tr.cat(grads)
	return flat_grad

def set_flat_grad(model, flat_grad):
	prev_ind = 0
	for param in model.parameters():
		flat_size = int(np.prod(list(param.size())))
		param.grad = flat_grad[prev_ind:prev_ind + flat_size].view(param.size())
		prev_ind += flat_size

class OptimizerWrapper(object):
	def __init__(self, criterion, net, clip_grad, embedder=None):
		self.criterion = criterion
		self.net = net
		self.embedder = embedder
		self.clip_value = 1.
		self.clip_grad = clip_grad 

	def step(self, g, perturbed_rewards, wdists, outputs, sample_params, param_dist, sigma, lr):

		set_flat_grad(self.net, g)

		if self.clip_grad:
			nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_value)

		return get_flat_grad(self.net)

	
	def eval(self,inputs,targets):
		self.optimizer.zero_grad()
		outputs = self.net(inputs)
		run_loss = self.criterion(outputs, targets).item()
		_, predicted =  outputs.max(1)
		return run_loss,predicted



class KWNGWrapper(OptimizerWrapper):
	def __init__(self, criterion, net, clip_grad, KWNG_estimator, dumping_freq, reduction_coeff, min_red, max_red, basis_schedule={}, beta=-1.0):
		OptimizerWrapper.__init__(self, criterion, net, clip_grad)
		self.net = net
		self.criterion = criterion
		self.KWNG_estimator = KWNG_estimator
		#self.KWNG_estimator.set_beta(beta)
		self.clip_grad = clip_grad
		self.clip_value = 1.
		self.old_loss = -1.
		self.dot_prod = 0.
		self.reduction_coeff = reduction_coeff
		self.dumping_freq = dumping_freq
		self.min_red = min_red
		self.max_red = max_red
		self.eps_min = 1e-10
		self.eps_max = 1e5
		self.dumping_counter = 0
		self.reduction_factor = 0.
		self.step_count = 0
		self.basis_schedule = basis_schedule

	def step(self, g, perturbed_rewards, wdists, outputs, sample_params, param_dist, sigma, lr):
		"""
		take a step.
		args:
			g: gradients
			perturbed_rewards: the rewards obtained from rollouts with the noisy params
			wdists: the associated wasserstein distances
			outputs: the trajectory embeddings of the rollouts
			sample_params: the sampled parameters
			param_dist: the distribution from which the parameters were sampled
			lr: the learning rate
		returns: 
			the pre-conditioned gradients
		"""
		if len(self.basis_schedule) > 0 and self.step_count in self.basis_schedule:
			self.update_basis_points(self.basis_schedule[self.step_count])

		# compute the loss 
		loss = self.criterion(perturbed_rewards, wdists, sample_params, param_dist)
		# Adjust epsilon
		self.dumping(loss, lr)
		
		self.KWNG_estimator.compute_cond_matrix(self.net, outputs, sigma, sample_params)

		#g = get_flat_grad(self.net)
		cond_g = self.KWNG_estimator.compute_natural_gradient(g)

		# If the dot product is negative, just use the euclidean gradient
		self.dot_prod = tr.sum(g*cond_g)
		if self.dot_prod<=0:
			cond_g = g
		# Gradient clipping by norm
		if self.clip_grad:
			cond_g = self.clip_gradient(cond_g)

		# Saving the current value of the loss
		self.old_loss = loss.item()

		set_flat_grad(self.net, cond_g)
		#self.optimizer.step()
		#_, predicted = outputs.max(1)

		self.step_count += 1

		return cond_g #loss.item(),predicted


	def update_basis_points(self, num_basis):
		print (f"setting KWNG basis points to {num_basis}")
		self.KWNG_estimator.num_basis = num_basis


	def clip_gradient(self,cond_g):
		
		norm_grad = tr.norm(cond_g)
		clip_coef = self.clip_value / (norm_grad + 1e-6)
		if clip_coef<1.:
			self.dot_prod = self.dot_prod/norm_grad
			return cond_g/norm_grad
		else:
			return cond_g
	
	def dumping(self,loss,lr):
		if self.old_loss>-1:
			# Compute the reduction ratio
			red = 2.*(self.old_loss-loss)/(lr*self.dot_prod)
			if red > self.reduction_factor:
				self.reduction_factor = red.item()
		self.dumping_counter +=1
		if self.old_loss>-1 and np.mod(self.dumping_counter,self.dumping_freq)==0:
			if self.reduction_factor< self.min_red and self.KWNG_estimator.eps<self.eps_max:
				self.KWNG_estimator.eps /=  self.reduction_coeff
			if self.reduction_factor>self.max_red and self.KWNG_estimator.eps>self.eps_min:
				self.KWNG_estimator.eps =  self.KWNG_estimator.eps*self.reduction_coeff
			print("New epsilon: "+ str(self.KWNG_estimator.eps) + ", Reduction_factor: " + str(self.reduction_factor))
			self.reduction_factor = 0.

class KWNG(nn.Module):

	def __init__(self, kernel, num_basis=5, eps=1e-5, with_diag_mat=True, beta=-1.0):
		super(KWNG, self).__init__()
		self.kernel = kernel	
		self.eps = eps
		self.thresh = 0.
		self.num_basis = num_basis
		self.with_diag_mat= with_diag_mat
		self.K = None
		self.T = None
		self.beta = beta
		print (f"KWNG beta = {beta}")

	def set_beta(self, beta_new):
		print (f"setting KWNG beta = {beta_new}")
		self.beta = beta_new

	def compute_cond_matrix(self, net, outputs, noise_std, noisy_params):
		"""
		compute the conditioning matrix
		specifically, compute K and T from eq. 18 and 19 in the paper
		"""
		L, d = outputs.shape # 100 x 6
		idx = tr.randperm(outputs.shape[0])
		outputs = outputs.view(outputs.size(0), -1)
		basis = outputs[idx[0: self.num_basis]].clone().detach()
		mask_int = tr.LongTensor(self.num_basis).random_(0,d)
		mask = tr.nn.functional.one_hot(mask_int,d).to(outputs.device)
		mask = mask.type(outputs.dtype)

		sigma = tr.log(tr.mean(self.kernel.square_dist(basis,outputs))).clone().detach()
		print("sigma:   " + str(tr.exp(sigma).item()))
		sigma /=  np.log(10.)

		if hasattr(self.kernel, 'params_0'):
			self.kernel.params = self.kernel.params_0 + sigma

		dkdxdy, dkdx, _= self.kernel.dkdxdy(basis, outputs, mask=mask)
		self.K = (1./L)*tr.einsum('mni,kni->mk',dkdxdy,dkdxdy)
		aux_loss = tr.mean(dkdx,dim = 1)

		# compute T
		# from the note: T = (1 / N\sigma) * \sum dkdx * (\psi - \theta)
		# here, there are really two sets of noise, as the algorithm tests rollouts for 
		# theta + noise and theta - noise 
		# therefore, psi - theta aka the noise is the concatenation [psi - theta; theta - psi]
		# in the default settings, that produces a matrix of size 100 x 101 (2 x 50 rollouts 
		# for the +/- perturbations) and 101 parameters in the policy network  
		noise_up = noisy_params - net.params.view(1, -1)
		noise_down = -noise_up
		noise = tr.cat([noise_up, noise_down], dim=0)
		self.T = tr.matmul(dkdx, noise) / (L * noise_std)
		#self.T = 0.1 * tr.ones([self.K.shape[0], net.N]) # 5 x n_params = 5 x 101
		#self.T = self.compute_jacobian(aux_loss, net)

	def compute_natural_gradient(self,g):
		
		uu,ss,vv = tr.svd(self.K.double())
		ss_inv,mask = self.pseudo_inverse(ss)
		ss_inv = tr.sqrt(ss_inv)
		vv = tr.einsum('i,ji->ij',ss_inv,vv)
		self.T = tr.einsum('ij,jk->ik', vv.float(), self.T)
		cond_g, G,D = self.make_system(g,mask)

		try:
			U  = tr.cholesky(G)
			cond_g = tr.cholesky_solve(cond_g.unsqueeze(-1), U).squeeze(-1)
		except:
			try:
				cond_g = tr.solve(cond_g.unsqueeze(-1), G)[0].squeeze(-1)
			except:
				pinv = tr.pinverse(G)
				cond_g = tr.einsum('mk,k',pinv,cond_g)
		cond_g = tr.einsum('md,m->d',self.T,cond_g)
		#cond_g = (g + self.beta * cond_g) / self.eps # grad(L) - M (sort of) # (1. - self.beta)
		cond_g = (g - self.beta * cond_g) / self.eps # original
		cond_g = D*cond_g
		return cond_g

	def make_system(self,g,mask):
		if self.with_diag_mat==1:
			D = tr.sqrt(tr.sum(self.T * self.T, dim=0))
			D = 1./(D+1e-8)
		elif self.with_diag_mat==0:
			D = tr.ones(self.T.shape[1], dtype=self.T.dtype,device=self.T.device)

		cond_g = D * g
		cond_g = tr.einsum('md,d->m', self.T,cond_g)
		P = tr.zeros_like(cond_g)
		P[mask] = 1.
		G =  tr.einsum('md,d,kd->mk',self.T, D, self.T) + self.eps * tr.diag(P) #self.beta * 
		return cond_g, G, D

	def pseudo_inverse(self,S):
		SS = 1./S
		mask = (S<=self.thresh)
		SS[mask]=0.
		mask = (S>self.thresh)
		return SS, mask

	def compute_jacobian(self,loss,net):
		J = []
		b_size = loss.shape[0]
		for i in range(b_size):
			grads =  tr.autograd.grad(loss[i], net.parameters(), retain_graph=True)
			grads = [x.view(-1) for x in grads]
			grads = tr.cat(grads)
			J.append(grads)

		return tr.stack(J,dim=0)
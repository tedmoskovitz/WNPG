import torch as tr
from torch import optim, nn
import numpy as np
from gaussian import * 
from kwng_es import * 

def get_wrapped_optimizer(args, criterion, net, device='cpu'):
    if args['estimator'] == 'EuclideanGradient':
        return OptimizerWrapper(criterion, net, args['clip_grad'])
    elif args['estimator'] == 'KWNG':
        kernel  = Gaussian(1, args['log_bandwidth'], dtype=tr.float32, device=device)
        estimator = KWNG(kernel,
                         eps=args['epsilon'],
                         num_basis=args['num_basis'],
                         with_diag_mat=args['with_diag_mat'],
                         beta=args['kwng_beta'])
        
        return KWNGWrapper(criterion,
                           net,
                           args['clip_grad'],
                           estimator,
                           args['dumping_freq'],
                           args['reduction_coeff'],
                           args['min_red'],
                           args['max_red'],
                           basis_schedule={} if 'basis_schedule' not in args else args['basis_schedule'])
    
def get_optimizer(args, net):
    if args['base_optimizer'] == 'sgd':
        return optim.SGD(net.parameters(),
                         lr=args['learning_rate'],
                         momentum=args['momentum'],
                         weight_decay=args['weight_decay'])


def toeplitz(c, r):
    """
    (slow) pytorch implementation of scipy's toeplitz function
    args:
        c: the first column of the toeplitz matrix
        r: the first row; note that differences in the top left entry 
            default to the first entry of the first column
    outputs:
        the result toeplitz matrix
    """
    c = c.view(-1); r = r.view(-1); 
    num_rows = len(c); num_cols = len(r); 
    m = tr.zeros([num_rows, num_cols], dtype=tr.float32)
    m[0, :] = r
    m[:, 0] = c
    
    # the final num_cols - 1 entries in row r are the first 
    # num_cols - 1 entries of the row above
    # this results in a matrix with constant diagonals
    for r in range(1, num_rows):
        m[r, 1:] = m[r-1, :-1]
    return m


def fixed_embed(data):
    # default form for data is a list of [state, reward] lists
    # new form for data is [state, reward, action]
    # the default embedding is then the final state of the trajectory
    embedding = tr.tensor(data[-1][0], dtype=tr.float32)
    # should try with action sequence maybe--this is a list of outputs that can be 
    # passed to KWNG
    # add alternative PPEs
    
    return (embedding)


class Embed(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super(Embed, self).__init__()
        self.l1 = nn.Linear(in_dim, out_dim)
        
    def __call__(self, x):
        """for now, just make the input the final [state, reward, action]"""
        x = tr.cat(
            [tr.tensor(x[-1][0], dtype=tr.float32),
             tr.tensor([x[-1][1]], dtype=tr.float32),
             x[-1][-1].view(-1,)]
        )
        return self.l1(x)

def calcdists(embeddings, dists, i, master, m_embedding, args):
    
    n_master = min(args['n_iter'], args['n_prev'])
    dists0 = dists.clone()

    if n_master== 1:
        dists[i, 0] = master.wass.wd(m_embedding[0], embeddings[i])
        dists[i, 1] = master.wass.wd(m_embedding[0], embeddings[i+args['num_sensings']])
    else:
        # if we are comparing vs multiple previous policies
        dists[i, 0] = tr.mean(tr.tensor([master.wass.wd(x, embeddings[i]) for x in m_embedding]))
        dists[i, 1] = tr.mean(tr.tensor([master.wass.wd(x, embeddings[i+args['num_sensings']]) for x in m_embedding]))
    if any(tr.isnan(dists.flatten())):
        pdb.set_trace()
    return(dists)



def Adam(dx, m, v, learning_rate, t, eps = 1e-8, beta1 = 0.9, beta2 = 0.999):
    m = beta1 * m + (1 - beta1) * dx
    mt = m / (1 - beta1 ** t)
    v = beta2 * v + (1-beta2) * (dx.pow(2))
    vt = v / (1 - beta2 **t)
    update = learning_rate * mt / (tr.sqrt(vt) + eps)
    return(update, m, v)

def SGA(dx, m, v, learning_rate, t):
    return (learning_rate * dx, None, None)


def env_dim(env):

	env_dims = {
	'Swimmer-v2' : 16,
    'Swimmer-v1' : 16,
	'Hopper-v2': 16,
	'Reacher-v2': 16,
	'Pusher-v2': 16,
	'HalfCheetah-v2': 32,
	'Walker2d-v2': 32,
	'Ant-v2': 64,
	'Humanoid-v2': 128,
    'DMBall-v0': 16,
    'DMCartPoleSwingupSparse-v0': 16,
    'DMHopperHop-v0': 32,
    'point-v0': 16,
    'antwall-v0': 64,
    'swimmerfinishline-v0': 16}

	return(env_dims[env])

def get_rf_dim(params):
    
    if params['embedding'] == 'state_final':
        dim = params['ob_dim']
    elif params['embedding'] == 'r_to_go':
        dim = params['steps']
    elif params['embedding'].split('-')[0] == 'state_cluster':
        dim = int(params['embedding'].split('-')[1])
    elif params['embedding'] == 'state_vector':
        dim = dim = params['ob_dim'] * params['steps']
    return(dim)

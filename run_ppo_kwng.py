#!/usr/bin/env python
import argparse
from baselines import bench, logger

import os
import numpy as np
import pickle
try:
    import multigoal
except:
    pass
import time


class CALLBACK(object):

    def __init__(self, arg):
        self.arg = arg
        if arg['gae'] is True:
            self.directory = 'ppo2_{}/seed_{}lr_{}entcoef_{}D_{}M_{}gammaW_{}wassersteincoeff_{}T_{}lrbeta_{}sigma_{}_opt_{}_{}'.format(arg['env'], arg['seed'], arg['lr'], arg['entcoef'],
                arg['D'], arg['M'], arg['gamma_W'], arg['wasserstein_coeff'], arg['T'], arg['lr_beta'], arg['sigma'], arg['base_opt'], arg['kwng_method'])
        else:
            self.directory = 'ppo2_{}/seed_{}lr_{}entcoef_{}D_{}M_{}gammaW_{}wassersteincoeff_{}T_{}lrbeta_{}sigma_{}gae_{}_opt_{}_kwng'.format(arg['env'], arg['seed'], arg['lr'], arg['entcoef'],
                arg['D'], arg['M'], arg['gamma_W'], arg['wasserstein_coeff'], arg['T'], arg['lr_beta'], arg['sigma'], arg['gae'], arg['base_opt'])            
        self.timesteps_origin = 0
        if arg['continue-train'] == 1:
            logger.info('loading model')
            assert os.path.exists(self.directory)
            self.epoch_episode_rewards = list(np.load(self.directory + '/epoch_episode_rewards.npy'))
            self.epoch_episode_steps = list(np.load(self.directory + '/epoch_episode_steps.npy'))
            self.epoch_xpos = list(np.load(self.directory + '/epoch_xpos.npy'))
            self.total_timesteps = list(np.load(self.directory + '/total_timesteps.npy'))
            self.loss_dict = {}  # do not reload loss
            assert os.path.exists(self.directory + '/model')
            self.timesteps_origin = self.total_timesteps[-1]
        elif arg['continue-train'] == 0:
            logger.info('training from scratch')
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            self.epoch_episode_rewards = []
            self.epoch_episode_steps = []
            self.epoch_xpos = []
            self.total_timesteps = []
            self.loss_dict = {}
            self.entropys = []
            self.approxkls = []
            self.clocktime = []
        else:
            raise NotImplementedError
        self.clocktime.append(time.time())

    def __call__(self, lcl, glb):
        self.epoch_episode_rewards.append(lcl['eprewmean'])
        self.epoch_episode_steps.append(lcl['eplenmean'])
        #self.epoch_xpos.append(np.copy(lcl['epxposlist']))
        self.total_timesteps.append(lcl['total_timesteps_sofar'] + self.timesteps_origin)
        self.entropys.append(lcl['policy_entropy'])
        self.approxkls.append(lcl['approxkl'])
        self.clocktime.append(time.time())
        np.save(self.directory + '/epoch_episode_rewards', self.epoch_episode_rewards)   
        np.save(self.directory + '/epoch_episode_steps', self.epoch_episode_steps)  
        np.save(self.directory + '/total_timesteps', self.total_timesteps)
        np.save(self.directory + '/policy_entropy', self.entropys)
        np.save(self.directory + '/approxkl', self.approxkls)
        np.save(self.directory + '/clocktime', self.clocktime)

        return False 

def train(env_id, num_timesteps, seed, lr, entcoef, continue_train, nsteps, D, M, gamma_W, wasserstein_coeff, T, lr_beta, sigma, gae, base_opt, n_kwng_basis, kwng_method, lr_decay, decay_factor):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    import ppo2_kwng2 as ppo2
    from policies import MlpPolicy
    import gym
    import pybulletgym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    
    if gae == 1:
        gae = True
    elif gae == 0:
        gae = False
    else:
        raise NotImplementedError

    # build call back
    arg = {}
    arg['seed'] = seed
    arg['env'] = env_id
    arg['lr'] = lr
    arg['entcoef'] = entcoef
    arg['continue-train'] = continue_train
    arg['D'] = D
    arg['M'] = M
    arg['gamma_W'] = gamma_W
    arg['wasserstein_coeff'] = wasserstein_coeff
    arg['T'] = T
    arg['lr_beta'] = lr_beta
    arg['sigma'] = sigma
    arg['gae'] = gae
    arg['base_opt'] = base_opt
    arg['kwng_method'] = kwng_method
    #arg[]
    callback = CALLBACK(arg)

    def make_policy(*args, **kwargs):
        return policy(*args, **kwargs, M=M)

    ppo2.learn(policy=make_policy, env=env, nsteps=nsteps, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=entcoef,
        lr=lr,
        cliprange=10.0,
        total_timesteps=num_timesteps, callback=callback, D=D, M=M, gamma_W=gamma_W, wasserstein_coeff=wasserstein_coeff,
            T=T, lr_beta=lr_beta, sigma=sigma, gae=gae, base_opt=base_opt, num_kwng_basis=n_kwng_basis, kwng_method=kwng_method, lr_decay=lr_decay, decay_factor=decay_factor)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--entcoef', type=float, default=0.0)
    parser.add_argument('--continue-train', type=int, default=0) # 1 for continued training
    parser.add_argument('--nsteps', type=int, default=2048)
    parser.add_argument('--D', type=int, default=100)
    parser.add_argument('--M', type=int, default=10)
    parser.add_argument('--gamma_W', type=float, default=1.)
    parser.add_argument('--wasserstein_coeff', type=float, default=0.1)
    parser.add_argument('--T', type=int, default=25)
    parser.add_argument('--lr_beta', type=float, default=5e-2)
    parser.add_argument('--sigma', type=float, default=.1)
    parser.add_argument('--gae', type=int, default=1)
    parser.add_argument('--base_opt', type=str, default='adam')
    parser.add_argument('--n_kwng_basis', type=int, default=5)
    parser.add_argument('--kwng_method', type=str, default='kwng')
    parser.add_argument('--lr_decay', type=int, default=-1)
    parser.add_argument('--decay_factor', type=float, default=4.0)
    args = parser.parse_args()
    logger.configure()
    train(args.env, nsteps=args.nsteps, entcoef=args.entcoef, num_timesteps=args.num_timesteps, seed=args.seed, lr=args.lr, continue_train=args.continue_train,
        D=args.D, M=args.M, gamma_W=args.gamma_W, wasserstein_coeff=args.wasserstein_coeff, T=args.T, lr_beta=args.lr_beta,
        sigma=args.sigma, gae=args.gae, base_opt=args.base_opt, n_kwng_basis=args.n_kwng_basis, kwng_method=args.kwng_method, lr_decay=args.lr_decay, decay_factor=args.decay_factor)


if __name__ == '__main__':
    main()


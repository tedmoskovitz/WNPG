import torch as tr
from torch import nn
import numpy as np
import time
import pdb
import gym
import simpleenvs
from evograd import expectation
from embeddings import * 
#from evograd.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from agent import *


class ES_criterion(nn.Module):

    def __init__(self, beta=0.):
        """
        the ES loss function
        """
        super(ES_criterion, self).__init__()
        self.beta = beta
        
    def forward(self, perturbed_rewards, wdists, sample_params, param_dist):
        # compute \bar{\ell}
        F = (1 - self.beta) * perturbed_rewards + self.beta * wdists
        #avg_F = expectation(F, sample_params, p=param_dist)
        return F.mean() # minimize -F

def ES_grad(perturbed_rewards, noise, wdists, sigma, pop_size=50, beta=0.):
    """
    compute the flattened ES gradients -- should definitely be sped up
    """
    n_params = noise.shape[1]
    g = tr.zeros(n_params)
    for i in range(pop_size):
        eps = noise[i, :]
        # combine reward and WD... beta could be adaptive as in the NSRA algorithm 
        g += eps * ((1 - beta) * perturbed_rewards[i] + beta * wdists[i])
    
    g /= (2 * sigma)

    return g 


class worker(object):
    
    def __init__(self, args, master, noise, i, train=True): # viz_only=False, env=None
        self.env = gym.make(args['env_name'])# if env is None else env
        self.env_name = args['env_name']
        self.v = noise[i, :] # the perturbation we will use
         
        args['zeros'] = True # initialize policy with zeros so we can set it to the current policy
        self.policy = ToeplitzPolicy(args)
        self.policy.update(master.params)
        #if viz_only: self.policy.update(noise[0, :]); 
        self.timesteps = 0
        self.rollout_length = args['steps']

        
    def do_rollouts(self, seed=0, train=True):
        
        self.policy.update(self.v) # Add the perturbation, to calculate F(theta + sigma * epsilon)
        up, up_data = self.rollout(seed, train)
        
        self.policy.update(-2 * self.v) # Subtract the perturbation, to calculate F(theta - sigma * epsilon)
        down, down_data = self.rollout(seed, train)
        #down = np.zeros_like(up)
        
        self.rewards = tr.tensor([up, down]).view(2)
        self.up_data = up_data
        self.down_data = down_data
    
    def rollout(self, seed=0, train=True, render=False):
        self.env.seed(seed)
        #if render: self.env.render(); 
        state = self.env.reset()
        self.env._max_episode_stesp = self.rollout_length # if you want to use another env, then change this!
        total_reward = 0
        done = False
        data = []
        while not done:
            action = self.policy(tr.tensor(state, dtype=tr.float32))
            action_dist = MultivariateNormal(action, 0.01 * tr.eye(action.numel()))
            action = action_dist.sample((1,))
            try:
                state, reward, done, _ = self.env.step(action.detach().numpy())
            except:
                pdb.set_trace()
                #state, reward, done, _ = self.env.step(action.detach().numpy())
                self.env.seed(seed+1)
                state = self.env.reset()
                state, reward, done, _ = self.env.step(action.detach().numpy())
            total_reward += reward
            data.append([state, reward])
            self.timesteps += 1
            #if render:
            #    self.env.render()
            #    time.sleep(0.01) 
            #    #print (self.timesteps, np.mean(state), np.mean(action.detach().numpy()))
        #if render: self.env.close(); 
        return (total_reward, data)



def aggregate_rollouts(master, noise, args):
       
    all_rollouts = tr.zeros([args['num_sensings'], 2])
    up, down = [], []
    up_actions, down_actions = [], []
    timesteps = 0 # counter for total number of steps
    #embed_fn = args['embedding']
    
    # want outputs to be [pop_size, 2]
    
    for i in range(args['num_sensings']):
        w = worker(args, master, noise, i)
        w.do_rollouts()
        all_rollouts[i] = w.rewards
        # up, down are lists of embeddings for \theta + noise, \theta - noise, respectively
        up.append(embed(args, w.up_data)); down.append(embed(args, w.down_data)); 
        #up_actions.append(tr.cat([d[-1].view(-1, 2) for d in w.up_data]))
        #down_actions.append(tr.cat([d[-1].view(-1, 2) for d in w.down_data]))
        timesteps += w.timesteps

    
    embeddings = up + down
    
    if args['optimizer'] == 'ES':
        wdists = tr.zeros(args['num_sensings'])
    else:
        # Update behavioral test funcs and use them to calculate WDs for each perturbed policy
        if args['n_iter'] == 1:
            wdists = tr.zeros(args['num_sensings'])
        else:
            wdists = tr.zeros([args['num_sensings'], 2])
            master.wass.update(master.buffer, embeddings, args)
            for i in range(args['num_sensings']):
                wdists = calcdists(embeddings, wdists, i, master, master.embedding, args)

            if any(tr.isnan(wdists.flatten())):
                pdb.set_trace()
            # normalize wdists
            wdists = (wdists - tr.mean(wdists)) / (tr.std(wdists)  + 1e-8)    
            wdists = wdists[:, 0] - wdists[:, 1]
            
        master.buffer = embeddings
    
    # normalize rewards    
    all_rollouts = (all_rollouts - tr.mean(all_rollouts)) / (tr.std(all_rollouts)  + 1e-8)  
    # compute R_k - R_t -> F(\theta + noise) - F(\theta)
    perturbed_rewards = all_rollouts[:,0] - all_rollouts[:, 1]

    if args['output_type'] == "embeddings": outputs = tr.tensor(np.stack(embeddings)); 
    elif args['output_type'] == "actions": outputs = tr.cat(up_actions + down_actions, dim=0); # no good atm
    elif args['output_type']== "rewards": outputs = tr.cat([all_rollouts[:,0], all_rollouts[:,1]]).view(-1, 1); 
    else: outputs = tr.stack(embeddings); 

    return perturbed_rewards, wdists, outputs, timesteps
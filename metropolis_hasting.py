# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 22:42:16 2017

@author: AJ
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
from six.moves import range
import time


class configuration():
    def __init__(self, config):
        if isinstance(config, np.ndarray) and len(config.shape)==1:
            self.config = configz
            self.length = len(config)
        elif isinstance(config, configuration) and len(config.config.shape)==1:
            self.config = config.config
            self.length = len(config.config)
        else:
            return NotImplemented
        
    def __add__(self, obj):
        if isinstance(obj.config, np.ndarray) and len(obj.config.shape)==1 and len(obj.config)==self.length:
            return configuration(config = self.config + obj.config)
        elif isinstance(obj, np.ndarray) and len(obj.shape)==1 and len(obj)==self.length:
            return configuration(config = self.config + obj)
        else:
            return NotImplemented

# function you want to minimize
def target_prob(configObj):
    #E is the function that is minimized and the minimium point is optimal point and the minimum value is 0
    #users is welcomed to modify their favorite function
    n = configObj.length
    x = configObj.config
    energy = np.sum([x[i]*x[n-i-1] for i in range(n)])+ np.sum([i*x[i] for i in range(n)])
    pi = np.exp(-energy) #if discrete
    #pi = np.exp(-energy**2) #if continuous # to ensure converge over R^n.
    return pi

def proposed_prob_And_perturb_config(configObj):
    n = configObj.length
    proposed_config = None
    g = None
    g_symmetric = None
    
    #Note if g is symmetric probability density function, then g(x|y)=g(y|x) and the computation of g is not needed actually.
    # Pick one of the following ways to generate proposed probability density function and new config
    
    #####The discrete probability distribution is as######################################
    #####First pick the number of positions will be flipped, labeled by num###############
    #####Pick all num locations in a config for flipping a sign.##########################
    proposed_config = configObj.config
    num = np.random.randint(low=0, high = n)+1
    positions =list(range(n))
    np.random.shuffle(positions)
    [proposed_config[pos]=-1*proposed_config[pos] for pos in positions[0:num]]
    gCurrentToProposed = 1/n*(1/cnr(n = n,r = num));
    gProposedToCurrent = 1/n*(1/cnr(n = n,r = num));                  
    g_symmetric = True
    """
    ######discrete uniform probability distribution (symmetric)############
    proposed_config = configObj.config
    pos = np.random.randint(low=0, high = n)
    proposed_config[pos] = np.random.randint(low = 0, high = 2)*2-1
    gCurrentToProposed = 1/2*1/n;
    gProposedToCurrent = 1/2*1/n;
    g_symmetric = True
    """
    
    """
    ######continuous uniform probability distribution (symmetric)############
    proposed_config = configObj.config
    pos = np.random.randint(low=0, high = n)
    proposed_config[pos] = np.random.randint(low = 0, high = 2)*2-1
    gCurrentToProposed = 1;
    gProposedToCurrent = 1;
    g_symmetric = True
    """
    
    """
    ######discrete random walk distribution (symmetric)############
    perturb = np.random.randint(low = -1, high = 2, size = n)
    proposed_config = proposed_config + np.random.randint(low = 0, high = 2, size = n)*2-1
    gCurrentToProposed = 1/2**n
    gProposedToCurrent = 1/2**n;
    g_symmetric = True
    """
    
    """
    ######continuous Gaussian probability distribution (symmetric)############
    std = 1
    perturb = np.random.normal(loc = 0, scale = std, size = n)
    proposed_config = configObj.config + np.random.normal(loc = 0, scale = 1, size = n)
    gCurrentToProposed = np.exp(-perturb**2/2/(std**n))
    gProposedToCurrent = np.exp(-perturb**2/2/(std**n))
    g_symmetric = True
    """
    
    return proposed_config, g_symmetric, gCurrentToProposed, gProposedToCurrent;

    
def ratio_probability(current_pi, proposed_pi, g_symmetric=True, current_g=None, proposed_g=None):
    if g_symmetric:
        ratio_prob = min(1, proposed_pi/current_pi)
    else:
        ratio_prob = min(1, proposed_pi/current_pi*proposed_g/current_g)
    
    return ratio_prob

def run(initial_config, iteration, eqm_itr, seed, denom):
    np.random.seed(seed)
    configObj = configuration(initial_config)
    stable_const = 0.000000000001
    seqOfConfigs = np.matrix(shape = (iteration - eqm_itr, initial_config.length))
    #xValue=[i for i in range(iteration)];
    #yValue=[]
    for itr in range(iteration):
        
        proposed_config, g_symmetric, _, _ =proposed_prob_And_perturb_config(configObj)
        current_pi = target_prob(configObj.config)
        proposed_pi = target_prob(proposed_config)
        accepted_prob = ratio_probability(current_pi=current_pi, proposed_pi=proposed_pi, g_symmetric=True, current_g=None, proposed_g=None)
        if accepted_prob > np.random.rand():
            configObj = configuration(proposed_config)
        if itr >= eqm_itr:
            seqOfConfigs[itr-eqm_itr] = configObj.config
                    
    for i in range(100):
        print("Markov chain following the target probability density func is")
        print(seqOfConfigs[i])
    #Calculate expectation, std to verify
    #Generate a Gaussian distribution?
        
if __name__ == "__main__":
    initial_config = np.array([10000,-5000,0.00001,-20]) #user can change the initial configuration state
    iteration = 60000 #user can change iterations
    initial_temperature = 100 #user can change initial_temperature
    seed = 1314 #user can change the seed number
    denom = 100 #user can change the decay rate of the temperature
    run(initial_config, iteration, initial_temperature, seed, denom)
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
import math

def cnr(n,r):
    numerator = 1;
    denominator= 1;
    for i in range(n, r+1, -1):
        numerator = numerator*i
    for j in range(r,1,-1):
        denominator = denominator*j
    
    return numerator/denominator
class configuration():
    def __init__(self, config):
        if isinstance(config, np.ndarray) and len(config.shape)==1:
            self.config = config
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

# The probability that configs in the Markov chain follow.
def target_prob(configObj):
    n = configObj.length
    x = configObj.config
    
    """
    #TYPE A target probability
    #If the states of the configObj.config take values +1 and -1 only, we can use the following target probabilitiy.
    energy = np.sum([x[i]*x[n-i-1] for i in range(n)])+ np.sum([i*x[i] for i in range(n)])
    pi = np.exp(-energy) #if discrete
    """
    
    #TYPE B target probability
    #If the states of the configObj.config take any integer or real values, we can use the following target probabilitiy.
    #The following target probability is Gaussian whose mean is zero and convariance matrix is the identity matrix.
    pi = 1/np.sqrt(2*np.pi)*np.exp(-np.sum((x)**2)/2)
    
    return pi

# The probability that is used to perturb configs in the Markov chain
def proposed_prob_And_perturb_config(configObj):
    n = configObj.length
    proposed_config = None
    gCurrentToProposed = None
    gProposedToCurrent = None
    g_symmetric = None
    
    #Note if g is symmetric probability density function, then g(x|y)=g(y|x) and the computation of g is not needed actually.
    # Pick one of the following ways to generate proposed probability density function and new config
    
    
    """
    #################TYPE A Prosposed_Prob And Perturbation##########################
    #Used for configObj.config which has two states only: +1 and -1, e.g. [1,-1,1,-1]#
    proposed_config = configObj.config
    num = np.random.randint(low=0, high = n)+1
    positions =list(range(n))
    np.random.shuffle(positions)
    for pos in positions[0:num]:
        proposed_config[pos]=-1*proposed_config[pos]
    gCurrentToProposed = 1/n*(1/cnr(n = n,r = num));
    gProposedToCurrent = 1/n*(1/cnr(n = n,r = num));                  
    g_symmetric = True
    """
    
    """
    ####################TYPE B Prosposed_Prob And Perturbation##########################
    #Used for configObj.config whose states take integer values only, e.g. [2,10,0,-10]#
    ##############discrete uniform probability distribution (symmetric)#################
    rand = np.zeros(n)
    pos = np.random.randint(low=0, high = n)
    rand[pos] = np.random.randint(low = 0, high = 2)*2-1
    proposed_config =configObj.config + rand
    gCurrentToProposed = 1/2*1/n;
    gProposedToCurrent = 1/2*1/n;
    g_symmetric = True
    """
    
    ########################TYPE C Prosposed_Prob And Perturbation##########################
    #Used for configObj.config whose states take real values only, e.g. [2.2,-1.3,0.0,0.09]#
    ############continuous uniform probability distribution (symmetric)#####################
    pos = np.random.randint(low=0, high = n)
    rand = np.zeros(n)
    rand[pos] = np.random.uniform(low = -1, high = 1)
    proposed_config =  configObj.config + rand
    gCurrentToProposed = 1;
    gProposedToCurrent = 1;
    g_symmetric = True
    
    
    """
    ####################TYPE D Prosposed_Prob And Perturbation##########################
    #Used for configObj.config whose states take integer values only, e.g. [2,10,0,-10]#
    ####################discrete random walk distribution (symmetric)###################
    perturb = np.random.randint(low = -1, high = 2, size = n)
    proposed_config = configObj.config + np.random.randint(low = 0, high = 2, size = n)*2-1
    gCurrentToProposed = 1/2**n
    gProposedToCurrent = 1/2**n;
    g_symmetric = True
    """
    
    """
    ########################TYPE E Prosposed_Prob And Perturbation##########################
    #Used for configObj.config whose states take real values only, e.g. [2.2,-1.3,0.0,0.09]#
    #################continuous Gaussian probability distribution (symmetric)###############
    std = 1
    perturb = np.random.normal(loc = 0, scale = std, size = n)
    proposed_config = configObj.config + np.random.normal(loc = 0, scale = 1, size = n)
    gCurrentToProposed = np.exp(-perturb**2/2/(std**n))
    gProposedToCurrent = np.exp(-perturb**2/2/(std**n))
    g_symmetric = True
    """
    
    proposed_configObj = configuration(proposed_config)
    return proposed_configObj, g_symmetric, gCurrentToProposed, gProposedToCurrent;


# The compute the accepted probability to accept the proposed config or just keep the current config.  
def ratio_probability(current_pi, proposed_pi, g_symmetric=True, gCurrentToProposed=None, gProposedToCurrent=None):
    if g_symmetric:
        ratio_prob = min(1, proposed_pi/(current_pi))
    else:
        ratio_prob = min(1, proposed_pi*gProposedToCurrent/(gCurrentToProposed*current_pi))
    
    return ratio_prob

#Running Metropolis-Hasting Algorithm
def run(initial_config, iteration, eqm_itr, display, seed):
    np.random.seed(seed)
    configObj = configuration(initial_config)
    seqOfConfigs = np.ndarray(shape = (iteration - eqm_itr, configObj.length))
    for itr in range(iteration):
        proposed_configObj, g_symmetric, _, _ =proposed_prob_And_perturb_config(configObj)
        current_pi = target_prob(configObj)
        proposed_pi = target_prob(proposed_configObj)
        accepted_prob = ratio_probability(current_pi=current_pi, proposed_pi=proposed_pi, g_symmetric=True, gCurrentToProposed=None, gProposedToCurrent=None)  
        if accepted_prob > np.random.rand():
            configObj = proposed_configObj
        if itr >= eqm_itr:
            seqOfConfigs[itr-eqm_itr] = configObj.config
    
    print("The last %d Markov chain following the target probability density func is" %display)        
    for i in range(-display,0):   
        print(seqOfConfigs[i])
        
    #Calculate expectation and convariance matrix to verify the Metropolis-Hasting Algorithm
    #If type B target probabillity is used, empirical_mean and empirical_convariance matrix should be around 0 and the identity matrix respectively.
    #Generate a Gaussian distribution?
    empirical_mean = np.mean(seqOfConfigs, axis = 0)
    empirical_convariance = np.diag(np.mean((seqOfConfigs-empirical_mean)**2, axis = 0))
    print("empirical_mean", empirical_mean)
    print(" empirical_convariance matrix",  empirical_convariance)
    
    """
    #The following is used if TYPE A target probability and TYPE A proposed probability and perturbation are used.
    empirical_energy = np.mean([np.log(target_prob(configuration(seqOfConfig))) for seqOfConfig in seqOfConfigs] )
    print("empirical_energy", empirical_energy)
    """
    
if __name__ == "__main__":
    initial_config = np.array([10]*5) #user can change the initial configuration state, 
                                      #if TYPE A target probability  and TYPE A proposed probability and perturbation are used, each value in the initial_config is either 1 or -1
                                      #if TYPE B target probability  and TYPE B or D proposed probability and perturbation are used, each value in the initial_config is integer.
                                      #if TYPE B target probability  and TYPE C or E proposed probability and perturbation are used, each value in the initial_config is any real number.
    iteration = 100000 #user can change iterations
    eqm_itr = 90000    #user can change its value to start to collect states in the Markov Chain for simulation, 
                      #like computation of the mean of states and convariance matrix.
    display = 10      #user can change to show the last "display" number of states in the Markov Chain
    seed = 1314       #user can change the seed number
    run(initial_config, iteration, eqm_itr, display, seed)
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 22:57:43 2017

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

# function you want to minimize
def energy(configObj, optimal_point):
    #E is the function that is minimized and the minimium point is optimal point and the minimum value is 0
    #users is welcomed to modify their favorite function
    E = np.sum((configObj.config - optimal_point)**2)+(configObj.config[0] - optimal_point[0])**2
    return E

def perturb_config(configObj):
    mu = 0
    sigma = 0.5 #this value can modify. 
                #Small sigma helps config to converge to the optimal point but the convergence takes longer tim
    length = configObj.length
    """the symmetric probabiltiy density function is used here for 2 reasons.
    (1) it matches with MCMC setting in which the proposed probability density function is symmetric.
    (2) in general, we have no ideas about the landscape and smoothness of the energy function 
    which is required to be minimized. Here, it makes sense for a config state to explore in 
    all directions equally likely.
    """
    perturb = configuration(np.random.normal(loc = mu, scale = sigma, size = length))
    return configObj + perturb
    
def ratio_probability(current_energy, temp_energy, temperature):
    """probability density function is a function that the probability is 
    almost 1 at the optimal point and the probabiltiy is almost zero at anywhere else.
    Since the simulated annealing requires the ratio of probability, the so-called "probability 
    density function" does not need to be sum to 1.
    """
    #probability_density_function is prob(energy, temperature) = exp(-energy/temperature)
    #ratio_prob = prob(temp_energy, temperature)/prob(current_energy, temperature)
    if temp_energy <= current_energy:
        ratio_prob = 1
    else:
        ratio_prob = np.exp(-(temp_energy - current_energy)/(temperature+1));
    #probability_density_function is prob(energy, temperature) = 1/(energy+0.000001)**2   
    #user can try this probability density function too.
    #if temp_energy <= current_energy:
    #   ratio_prob = 1
    #else:
    #   ratio_prob = ((current_energy+0.000001)/(temp_energy+0.000001))**2
    return ratio_prob

#Running Simulated-Annealing Algorithm
def run(initial_config, iteration, initial_temperature, seed, denom):
    np.random.seed(seed)
    configObj = configuration(initial_config)
    temperature =initial_temperature;
    optimal_point = np.random.uniform(low = -200, high = 201, size = configObj.length)
    #stable_const = 0.000000000001
    xValue=[i for i in range(iteration)];
    yValue=[]
    for itr in range(iteration):
        current_energy = energy(configObj, optimal_point)
        yValue.append(current_energy)
        
        temp_configObj = perturb_config(configObj)
        temp_energy = energy(temp_configObj, optimal_point)
        
 
        if ratio_probability(current_energy, temp_energy, temperature) > np.random.rand():
            configObj = temp_configObj
        
        
        #the temperature drops to its half at every denom itr
        temperature = initial_temperature * 0.5 **(itr/denom);
    
    output_energy, output_point = current_energy, configObj.config
    minimum_energy = energy(configuration(optimal_point), optimal_point)
    print("Actual minimum energy is",minimum_energy," at the configuration state ",optimal_point)
    print("From simulated annealing")
    print("The approxmiated minimum energy is ",output_energy," at the configuration stat", output_point)
    
    plt.plot(xValue, yValue, 'ro')
    plt.xlabel('iteration')
    plt.ylabel('energy')
    plt.title('Energy approximated by simulated annealing along iteration')
    plt.show
if __name__ == "__main__":
    initial_config = np.array([10000,-5000,0.00001,-20]) #user can change the initial configuration state
    iteration = 60000 #user can change iterations
    initial_temperature = 100 #user can change initial_temperature
    seed = 1314 #user can change the seed number
    denom = 100 #user can change the decay rate of the temperature
    run(initial_config, iteration, initial_temperature, seed, denom)
        
    
    
        
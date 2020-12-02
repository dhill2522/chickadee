import chickadee
import numpy as np
import time
import matplotlib.pyplot as plt

n = 110 # number of time points
time_horizon = np.linspace(0, 10 , n)

steam = chickadee.Resource('steam')
electricity = chickadee.Resource('electricity')

load = chickadee.TimeSeries()

def smr_cost(dispatch):
    return sum(-0.001 * dispatch[steam])

def smr_transfer(data, meta):
    return data, meta


smr_capacity = np.ones(n)*1200
smr_ramp = 600*np.ones(n)
smr_guess = 100*np.sin(time_horizon) + 300
smr = chickadee.PyOptSparseComponent('smr', smr_capacity, smr_ramp, steam,
                                smr_transfer, smr_cost, produces=steam, guess=smr_guess)
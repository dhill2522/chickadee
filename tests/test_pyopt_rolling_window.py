'''
Test that rolling-window dispatch is working properly in Chickadee
using the pyOpySparse dispatcher
'''
import chickadee
import numpy as np
import time

n = 110 # number of time points
time_horizon = np.linspace(0, 10 , n)

steam = chickadee.Resource('steam')
electricity = chickadee.Resource('electricity')

load = chickadee.TimeSeries()

def smr_cost(dispatch):
    return sum(-0.001 * dispatch[steam])

def smr_transfer(inputs):
    return {}


smr_capacity = np.ones(n)*1200
smr_ramp = 600*np.ones(n)
smr_guess = 100*np.sin(time_horizon) + 300
smr = chickadee.PyOptSparseComponent('smr', smr_capacity, smr_ramp, smr_ramp, steam,
                                smr_transfer, smr_cost, produces=steam, guess=smr_guess)

def turbine_transfer(inputs):
    effciency = 0.7  # Just a random guess at a turbine efficiency
    return {steam: -1/effciency * inputs}

def turbine_cost(dispatch):
    return sum(-1 * dispatch[steam])

turbine_capacity = np.ones(n)*1000
turbine_guess = 100*np.sin(time_horizon) + 500
turbine_ramp = 20*np.ones(n)
turbine = chickadee.PyOptSparseComponent('turbine', turbine_capacity,
                                turbine_ramp, turbine_ramp, electricity,
                                turbine_transfer, turbine_cost,
                                produces=electricity, consumes=steam, guess=turbine_guess)


def el_market_transfer(inputs):
    return {} 

def elm_cost(dispatch):
    return sum(5.0 * dispatch[electricity])

# elm_capacity = np.ones(n)*-800
elm_capacity = -(100*np.sin(time_horizon) + 500)
elm_ramp = 1e10*np.ones(n)
elm = chickadee.PyOptSparseComponent('el_market', elm_capacity, elm_ramp, elm_ramp,
                                electricity, el_market_transfer, elm_cost,
                                consumes=electricity, dispatch_type='fixed')

dispatcher = chickadee.PyOptSparse(window_length=10)

comps = [smr, turbine, elm]
# comps = [smr, tes, turbine, elm]

start_time = time.time()
sol = dispatcher.dispatch(comps, time_horizon, [load], verbose=False)
end_time = time.time()
# print('Full optimal dispatch:', optimal_dispatch)
print('Dispatch time:', end_time - start_time)

# Check to make sure that the ramp rate is never too high
ramp = np.diff(sol.dispatch['turbine'][electricity])
assert max(ramp) <= turbine.ramp_rate_up[0], 'Max ramp rate exceeded!'


balance = sol.dispatch['turbine'][electricity] + \
    sol.dispatch['el_market'][electricity]

import matplotlib.pyplot as plt
plt.plot(sol.time, sol.dispatch['turbine'][electricity], label='El gen')
plt.plot(sol.time, sol.dispatch['el_market'][electricity], label='El cons')
plt.plot(sol.time, balance, label='Electricity balance')
plt.plot(sol.time[:-1], ramp, label='Ramp rate')
plt.legend()
plt.show()

print(sol.dispatch['smr'][steam])

import chickadee
import numpy as np
import time
import pandas as pd
from sqlalchemy import create_engine

n = 110 # number of time points

time_horizon = np.linspace(0, n-1 , n)
db_load = '../../data/Ercot/Ercot Load/ERCOT_load_2019.csv'
Loaddata = pd.read_csv(db_load)
Load = .5*Loaddata['ERCOT'].values[0:n]

steam = chickadee.Resource('steam')
electricity = chickadee.Resource('electricity')

load = chickadee.TimeSeries()

def smr_cost(dispatch: dict) -> float:
    # Impose a high ramp cost
    ramp_cost = 50*sum(abs(np.diff(dispatch[steam])))
    return sum(-1.0 * dispatch[steam] - ramp_cost)

def smr_transfer(data: dict, meta: dict) -> list:
    return data, meta


smr_capacity = np.ones(n)*1280*35
smr_ramp = np.ones(n)*1000  # FIXME This is the ramprates that need to change
smr_guess = np.ones(n)*.9*1280*35
smr = chickadee.PyOptSparseComponent('smr', smr_capacity, smr_ramp, smr_ramp, steam,
                                smr_transfer, smr_cost, produces=steam, guess=smr_guess)

def tes_transfer(data, meta):
    return data, meta

def tes_cost(dispatch):
    # Simulating high-capital and low-operating costs
    return -1000 - 0.000001*np.sum(dispatch[steam])

tes_capacity = np.ones(n)*1e5
tes_ramp = np.ones(n)*5e5
tes_guess = np.zeros(n)
tes = chickadee.PyOptSparseComponent('tes', tes_capacity, tes_ramp, tes_ramp, steam, tes_transfer,
                                    tes_cost, stores=steam, guess=tes_guess)

def turbine_transfer(data, meta):
    effciency = 0.7  # Just a random guess at a turbine efficiency

    if steam in data:
        # Determine the electricity output for a given steam input
        data[electricity] = -1 * effciency * data[steam]
    elif electricity in data:
        # Determine the steam input for a given electricity output
        data[steam] = -1/effciency * data[electricity]
    else:
        raise Exception(
            "Generator Transfer Function: Neither 'electricity' nor 'steam' given")

    return data, meta

def turbine_cost(dispatch):
    return sum(-1 * dispatch[steam])

turbine_capacity = np.ones(n)*1280*35
turbine_guess = np.ones(n)*.8*1280*35
turbine_ramp = np.ones(n)*.5*1280*35
turbine = chickadee.PyOptSparseComponent('turbine', turbine_capacity,
                                turbine_ramp, turbine_ramp, electricity,
                                turbine_transfer, turbine_cost,
                                produces=electricity, consumes=steam, guess=turbine_guess)


def el_market_transfer(data, meta):
    return data, meta

def elm_cost(dispatch):
    return sum(5.0 * dispatch[electricity])

# elm_capacity = np.ones(n)*-800
elm_capacity = -Load
elm_ramp = 1e10*np.ones(n)
elm = chickadee.PyOptSparseComponent('el_market', elm_capacity, elm_ramp, elm_ramp,
                                electricity, el_market_transfer, elm_cost,
                                consumes=electricity, dispatch_type='fixed')

dispatcher = chickadee.PyOptSparse(window_length=20)

# comps = [smr, turbine, elm]
comps = [smr, tes, turbine, elm]

start_time = time.time()
sol = dispatcher.dispatch(comps, time_horizon, [load], verbose=False)
end_time = time.time()
# print('Full optimal dispatch:', optimal_dispatch)
print('Dispatch time:', end_time - start_time)

# Check to make sure that the ramp rate is never too high
turbine_ramp = np.diff(sol.dispatch['turbine'][electricity])
tes_ramp = np.diff(sol.dispatch['tes'][steam])

balance = sol.dispatch['turbine'][electricity] + \
    sol.dispatch['el_market'][electricity]

import matplotlib.pyplot as plt
plt.subplot(2,1,1)
plt.plot(sol.time, sol.dispatch['tes'][steam], label='TES activity')
plt.plot(sol.time, sol.storage['tes'], label='TES storage level')
plt.plot(sol.time[:-1], tes_ramp, label='TES ramp')
ymax = max(sol.storage['tes'])
plt.vlines([w[0] for w in sol.time_windows], 0, ymax, colors='green', linestyles='--')
plt.vlines([w[1] for w in sol.time_windows], 0, ymax, colors='blue', linestyles='--')
plt.legend()

plt.subplot(2,1,2)
plt.plot(sol.time, sol.dispatch['smr'][steam], label='SMR generation')
plt.plot(sol.time, sol.dispatch['turbine'][electricity], label='turbine generation')
plt.plot(sol.time, sol.dispatch['el_market'][electricity], label='El market')
plt.plot(sol.time, balance, label='Electricity balance')
plt.plot(sol.time[:-1], turbine_ramp, label='turbine ramp')
ymax = max(sol.dispatch['smr'][steam])
plt.vlines([w[0] for w in sol.time_windows], 0,
           ymax, colors='green', linestyles='--')
plt.vlines([w[1] for w in sol.time_windows], 0,
           ymax, colors='blue', linestyles='--')
plt.legend()
plt.show()

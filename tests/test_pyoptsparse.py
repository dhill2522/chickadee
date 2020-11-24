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

# def tes_transfer(data, meta):
#     return data, meta

# def tes_cost(dispatch):
#     return sum(-0.01*np.ones(len(dispatch[steam])))

# tes_capacity = 200*np.ones(n)
# tes_ramp = 200*np.ones(n)
# tes_guess = np.zeros(n)
# tes = chickadee.PyOptSparseComponent('tes', tes_capacity, tes_ramp, steam,
#                                     tes_transfer, tes_cost, stores=steam,
#                                     guess=tes_guess)


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

turbine_capacity = np.ones(n)*1000
turbine_guess = 100*np.sin(time_horizon) + 500
turbine_ramp = 100*np.ones(n)
turbine = chickadee.PyOptSparseComponent('turbine', turbine_capacity,
                                turbine_ramp, electricity,
                                turbine_transfer, turbine_cost,
                                produces=electricity, consumes=steam, guess=turbine_guess)


def el_market_transfer(data, meta):
    return data, meta

def elm_cost(dispatch):
    return sum(5.0 * dispatch[electricity])

# elm_capacity = np.ones(n)*-800
elm_capacity = -(100*np.sin(time_horizon) + 500)
elm_ramp = 1e10*np.ones(n)
elm = chickadee.PyOptSparseComponent('el_market', elm_capacity, elm_ramp,
                                electricity, el_market_transfer, elm_cost,
                                consumes=electricity, dispatch_type='fixed')

dispatcher = chickadee.PyOptSparse(window_length=10)

comps = [smr, turbine, elm]
# comps = [smr, tes, turbine, elm]

start_time = time.time()
optimal_dispatch = dispatcher.dispatch(comps, time_horizon, [load], verbose=False)
end_time = time.time()
# print('Full optimal dispatch:', optimal_dispatch)
print('Dispatch time:', end_time - start_time)

ramp = np.diff(optimal_dispatch.state['turbine'][electricity])

balance = optimal_dispatch.state['turbine'][electricity] + \
    optimal_dispatch.state['el_market'][electricity]

plt.plot(time_horizon,
         optimal_dispatch.state['turbine'][electricity], label='El gen')
plt.plot(time_horizon,
         optimal_dispatch.state['el_market'][electricity], label='El cons')
plt.plot(time_horizon, balance, label='Electricity balance')
plt.plot(time_horizon[:-1], ramp, label='Ramp rate')
plt.legend()
plt.show()

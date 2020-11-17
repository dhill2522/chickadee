import chickadee
import numpy as np
import matplotlib.pyplot as plt

n = 10 # number of time points

steam = chickadee.Resource('steam')
electricity = chickadee.Resource('electricity')

load = chickadee.TimeSeries()

def smr_cost(dispatch):
    return sum(-1 * dispatch[steam])

def smr_transfer(data, meta):
    return data, meta


smr_capacity = np.ones(n)*600
smr_ramp = 600
smr = chickadee.PyOptSparseComponent('smr', smr_capacity, smr_ramp, steam,
                                smr_transfer, smr_cost, produces=steam)


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
turbine_ramp = 100
turbine = chickadee.PyOptSparseComponent('turbine', turbine_capacity,
                                turbine_ramp, electricity,
                                turbine_transfer, turbine_cost,
                                produces=electricity, consumes=steam)


def el_market_transfer(data, meta):
    return data, meta

def elm_cost(dispatch):
    return sum(5.0 * dispatch[electricity])

elm_capacity = np.ones(n)*800
elm_ramp = np.ones(n)*1e10
elm = chickadee.PyOptSparseComponent('el_market', elm_capacity, elm_ramp,
                                electricity, el_market_transfer, elm_cost,
                                consumes=electricity, dispatch_type='fixed')

dispatcher = chickadee.PyOptSparse(window_length=10)

comps = [smr, turbine, elm]
time_horizon = np.linspace(0, 1, n)
optimal_dispatch = dispatcher.dispatch(comps, time_horizon, [load], verbose=True)
print('Full optimal dispatch:', optimal_dispatch)

ramp = np.diff(optimal_dispatch.state['turbine'][electricity])

plt.plot(optimal_dispatch.state['time'], optimal_dispatch.state['turbine'][electricity], label='El gen')
plt.plot(optimal_dispatch.state['time'][:-1], ramp, label='Ramp rate')
plt.show()

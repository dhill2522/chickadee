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

if __name__ == "__main__":
    n_runs = 5
    # Dispatch using no scaling and no "slack" storage
    print('Testing with slack storage and no obj scaling')
    ttl = 0.0
    for i in range(n_runs):
        start_time = time.time()
        optimal_dispatch = dispatcher.dispatch(
            comps, time_horizon, [load], slack_storage=True, scale_objective=False)
        end_time = time.time()
        ttl += end_time - start_time
    print(f'Avg of {n_runs} dispatches: {ttl/n_runs:.4f}')

    print('Testing with slack storage and obj scaling')
    ttl = 0.0
    for i in range(n_runs):
        start_time = time.time()
        optimal_dispatch = dispatcher.dispatch(
            comps, time_horizon, [load], slack_storage=True, scale_objective=True)
        end_time = time.time()
        ttl += end_time - start_time
    print(f'Avg of {n_runs} dispatches: {ttl/n_runs:.4f}')

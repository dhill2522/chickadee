import chickadee
import numpy as np
import matplotlib.pyplot as plt

steam = chickadee.Resource('steam')
electricity = chickadee.Resource('electricity')

load = chickadee.TimeSeries()

smr = chickadee.PyOptSparseComponent('smr', 600, 10000, steam, None, -1.0,
                                produces=steam, dispatch_type='fixed')


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


turbine = chickadee.PyOptSparseComponent('turbine', 1000, 100, electricity,
                                turbine_transfer, -1.0, produces=electricity,
                                consumes=steam)


def el_market_transfer(data, meta):
    return data, meta


el_market = chickadee.PyOptSparseComponent('el_market', 1e10, 1e10, electricity,
                                el_market_transfer, 5.0, consumes=electricity,
                                dispatch_type='fixed')

dispatcher = chickadee.PyOptSparse(window_length=5)

comps = [smr, turbine, el_market]
time_horizon = np.linspace(0, 1, 10)
optimal_dispatch = dispatcher.dispatch(comps, time_horizon, [load])
print(optimal_dispatch)

ramp = np.diff(optimal_dispatch.state['turbine'][electricity])

plt.plot(optimal_dispatch.state['time'], optimal_dispatch.state['turbine'][electricity], label='El gen')
plt.plot(optimal_dispatch.state['time'][:-1], ramp, label='Ramp rate')
plt.show()

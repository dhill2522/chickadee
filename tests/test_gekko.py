import chickadee
import gekko
import numpy as np
import matplotlib.pyplot as plt

steam = chickadee.Resource('steam')
electricity = chickadee.Resource('electricity')

load = chickadee.TimeSeries()

def smr_transfer(m: gekko.GEKKO):
    npp_cap = 1280 * 20     # MW, capacity of South Texas Nuclear Generatingstation
    npp_ramprate = 0.01     # Max ramping as a percent of total capacity
    npp_gen = m.MV(value=.8*npp_cap, lb=0.2*npp_cap, ub=npp_cap)
    npp_gen.STATUS = 1
    npp_gen.DMAX = npp_ramprate*npp_cap
    npp_gen.DCOST = 10
    m.npp_gen = npp_gen
    return m

smr = chickadee.Component('smr', 600, steam, smr_transfer, -1.0,
                          produces=steam, dispatch_type='independent')

tes = chickadee.Component('tes', )

def turbine_transfer(model: gekko.GEKKO):
    return model


turbine = chickadee.Component('turbine', 1000, electricity, turbine_transfer, -1.0,
                              produces=electricity, consumes=steam)


def el_market_transfer(model):
    return model


el_market = chickadee.Component('el_market', 1e10, electricity, el_market_transfer, 5.0,
                                consumes=electricity, dispatch_type='fixed')

dispatcher = chickadee.PyOptSparse(window_length=5)

comps = [smr, turbine, el_market]
time_horizon = np.linspace(0, 1, 10)
optimal_dispatch = dispatcher.dispatch(comps, time_horizon, [load])
print(optimal_dispatch)

plt.plot(optimal_dispatch.state['time'],
         optimal_dispatch.state['turbine'][electricity])
plt.show()

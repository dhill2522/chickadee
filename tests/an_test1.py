'''
Test that storage is properly handled in Chickadee
using the pyOpySparse dispatcher
'''

import matplotlib.pyplot as plt
import chickadee
import numpy as np
import time

n = 40  # number of time points
time_horizon = np.linspace(0, n-1, n)

Electricity = 'electricity'
Natural_Gas = 'natural_gas'  # lb/hr



def GT_cost(dispatch: dict) -> float:
    '''Ecomonic cost function
    :param dispatch: dict, component dispatch
    :returns: float, the cost of running that dispatch

    This function will receive a dict representing a dispatch for a particular
    component and is required to return a float representing the economic cost
    of running that dispatch. A negative cost indicates an overall economic cost
    while a postive value indicates an economic benefit.

    This cost function is not required if the `external_obj_func` option is used
    in the dispatcher.
    '''
    # Impose a high ramp cost
    LHV = 46000*0.429923  # BTU/lb
    C_NG = 4.1155*LHV/999761  # $/lb
    return sum(-C_NG*np.array(dispatch[Natural_Gas])/1e6)  # $/hr


def GT_transfer(inputs: list) -> list:

    return {}

def gp_transfer(inputs: list):
    return {}

def gp_cost(dispatch: dict):
    return 0


gas_producer = chickadee.PyOptSparseComponent('gp', np.ones(
    n)*1e6, np.ones(n)*1e6, np.ones(n)*1e6, Natural_Gas, gp_transfer, gp_cost, produces=Natural_Gas)

GT_capacity = np.ones(n)*1200
GT_ramp = 600*np.ones(n)
GT_guess = np.ones(n) * 500
GT = chickadee.PyOptSparseComponent('Gas Turbine', GT_capacity,
                                    GT_ramp, GT_ramp,
                                    Natural_Gas, GT_transfer, GT_cost,
                                    consumes=Natural_Gas,
                                    produces=Electricity, guess=GT_guess)


def tes_transfer(inputs: list, init_store):
    tmp = np.insert(inputs, 0, init_store)
    return np.cumsum(tmp)[1:]

def tes_cost(dispatch):
    # Simulating high-capital and low-operating costs
    return -1000 - 0.01*np.sum(dispatch[Electricity])

tes_capacity = np.ones(n)*800
tes_ramp = np.ones(n)*50
tes_guess = np.zeros(n)
tes = chickadee.PyOptSparseComponent('tes', tes_capacity, tes_ramp,
                                     tes_ramp, Electricity, tes_transfer,
                                     tes_cost, stores=Electricity,
                                     guess=tes_guess, min_capacity=np.zeros(n))

def load_transfer(inputs: list):
    return {}
def load_cost(dispatch):
    return sum(5.0 * dispatch[Electricity])

load_capacity = -(20*np.sin(time_horizon) + 500)
load_ramp = 1e10*np.ones(n)
load = chickadee.PyOptSparseComponent('load', load_capacity,
                                      load_ramp, load_ramp,
                                      Electricity, load_transfer, load_cost,
                                      consumes=Electricity, dispatch_type='fixed')

dispatcher = chickadee.PyOptSparse(window_length=10)

comps = [gas_producer, GT, tes, load]

start_time = time.time()
sol = dispatcher.dispatch(comps, time_horizon, [], verbose=True)
end_time = time.time()
# print('Full optimal dispatch:', optimal_dispatch)
print('Dispatch time:', end_time - start_time)
print('Obj Value: ', sol.objval)


plt.plot(sol.time, sol.dispatch['Gas Turbine'][Electricity],
         label='Gas Turbine')
plt.plot(sol.time, sol.dispatch['load'][Electricity], label='Load')
plt.plot(sol.time, sol.storage['tes'], label='Battery')
# plt.plot(sol.time[:-1], turbine_ramp, label='turbine ramp')
ymax = max(sol.dispatch['Gas Turbine'][Electricity])
# plt.vlines([w[0] for w in sol.time_windows], 0,
#            ymax, colors='green', linestyles='--')
# plt.vlines([w[1] for w in sol.time_windows], 0,
#            ymax, colors='blue', linestyles='--')
plt.legend()
plt.savefig("Turbine test.png")
plt.show()

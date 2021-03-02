'''
Test that storage is properly handled in Chickadee
using the pyOpySparse dispatcher
'''
import chickadee
import numpy as np
import time

n = 20 # number of time points
time_horizon = np.linspace(0, n-1 , n)

steam = chickadee.Resource('steam')
electricity = chickadee.Resource('electricity')

load = chickadee.TimeSeries()

def smr_cost(dispatch: dict) -> float:
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
    ramp_cost = 5*sum(abs(np.diff(dispatch[steam])))

    return sum(-0.1 * dispatch[steam] - ramp_cost)

def smr_transfer(input: list) -> dict:
    '''A component transfer function
    Uses a different format than is used in other HERON dispatchers. Takes in a
    list of time-resolved inputs of the capacity resource and returns a dict of
    arrays. Each dict entry represents the time-resolved response of the other
    involved resources. If no other resources are involved than an empty dict is
    returned.

    Note that "meta" is neither passed in nor returned as is done in
    other HERON dispatchers. This is because it has not been used to this point,
    but severely limits possible methods of determining the function
    derivatives.
    '''
    return {}


smr_capacity = np.ones(n)*1200
smr_ramp = 600*np.ones(n)
smr_guess = 100*np.sin(time_horizon) + 600
# smr_guess = np.ones(n) * 700
smr = chickadee.PyOptSparseComponent('smr', smr_capacity, smr_ramp, smr_ramp,
                    steam, smr_transfer, smr_cost, produces=steam,
                    guess=smr_guess)

def tes_transfer(input, init_level):
    '''This is a storage component transfer function. This transfer function
    works just the same as the others with two significant differences.
    1) It requires a second argument that is the intial storage level of the
    component.
    2) Instead of returning the activities of the other involved
    resources (there are none for storage components) it returns a time-
    resolved array of storage levels for the component.'''
    tmp = np.insert(input, 0, init_level)
    return np.cumsum(tmp)[1:]

def tes_cost(dispatch):
    # Simulating high-capital and low-operating costs
    return -1000 - 0.01*np.sum(dispatch[steam])

tes_capacity = np.ones(n)*800
tes_ramp = np.ones(n)*50
tes_guess = np.zeros(n)
tes = chickadee.PyOptSparseComponent('tes', tes_capacity, tes_ramp,
                                    tes_ramp, steam, tes_transfer,
                                    tes_cost, stores=steam, guess=tes_guess)

def turbine_transfer(inputs):
    '''Just using a guesstimated efficiency'''
    return {steam: -1/0.7 * inputs}

def turbine_cost(dispatch):
    return sum(-1 * dispatch[steam])

turbine_capacity = np.ones(n)*1000
turbine_guess = 20*np.sin(time_horizon) + 500
turbine_ramp = 20*np.ones(n)
turbine = chickadee.PyOptSparseComponent('turbine', turbine_capacity,
                                turbine_ramp, turbine_ramp, electricity,
                                turbine_transfer, turbine_cost,
                                produces=electricity, consumes=steam, guess=turbine_guess)


def el_market_transfer(inputs: list):
    return {}

def elm_cost(dispatch):
    return sum(5.0 * dispatch[electricity])

# elm_capacity = np.ones(n)*-800
elm_capacity = -(20*np.sin(time_horizon) + 500)
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
print('Obj Value: ', sol.objval)

# Check to make sure that the ramp rate is never too high
turbine_ramp = np.diff(sol.dispatch['turbine'][electricity])
tes_ramp = np.diff(sol.dispatch['tes'][steam])

el_balance = sol.dispatch['turbine'][electricity] + \
    sol.dispatch['el_market'][electricity]
steam_balance = sol.dispatch['smr'][steam] + sol.dispatch['turbine'][steam]

print(len(sol.dispatch['smr'][steam]))
print(len(sol.time))

import matplotlib.pyplot as plt
plt.subplot(2,1,1)
plt.plot(sol.time, sol.dispatch['tes'][steam], label='TES activity')
plt.plot(sol.time, sol.storage['tes'], label='TES storage level')
plt.plot(sol.time[:-1], tes_ramp, label='TES ramp')
ymax = max(sol.storage['tes'])
plt.vlines([w[0] for w in sol.time_windows], 0,
           ymax, colors='green', linestyles='--')
plt.vlines([w[1] for w in sol.time_windows], 0,
           ymax, colors='blue', linestyles='--')
plt.legend()

plt.subplot(2,1,2)
plt.plot(sol.time, sol.dispatch['smr'][steam], label='SMR generation')
plt.plot(sol.time, sol.dispatch['turbine'][electricity], label='turbine el generation')
plt.plot(sol.time, sol.dispatch['turbine'][steam], label='turbine steam consumption')
plt.plot(sol.time, sol.dispatch['el_market'][electricity], label='El market')
plt.plot(sol.time, el_balance, 'r.', label='Electricity balance')
plt.plot(sol.time, steam_balance, 'b.', label='Steam balance')
plt.plot(sol.time[:-1], turbine_ramp, label='turbine ramp')
ymax = max(sol.dispatch['smr'][steam])
plt.vlines([w[0] for w in sol.time_windows], 0,
           ymax, colors='green', linestyles='--')
plt.vlines([w[1] for w in sol.time_windows], 0,
           ymax, colors='blue', linestyles='--')
plt.legend()
plt.show()

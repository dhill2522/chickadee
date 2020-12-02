'''A unit test file for the pyoptsparse dispatcher'''

import chickadee
import numpy as np

n = 10
time_horizon = np.linspace(0, 1, n)

dispatcher = chickadee.PyOptSparse()
res1 = chickadee.Resource('res1')

# ---------------- Component Tests ----------------
comp1_cap = np.ones(n)
comp1_ramp = np.ones(n)
def comp1_trans(data, meta):
    return data, meta
def comp1_cost(dispatch):
    return -1*sum(dispatcher[res1])
comp1 = chickadee.PyOptSparseComponent('comp1', comp1_cap, comp1_ramp, res1,
                                        comp1_trans, comp1_cost, produces=res1)
# Should auto-generate the right guess value
assert np.array_equal(comp1.guess, comp1_cap)


comp2_cap = np.ones(n)
comp2_ramp = np.ones(n)
comp2_guess = 0.9*comp2_cap
def comp2_trans(data, meta):
    return data, meta
def comp2_cost(dispatch):
    return 1*sum(dispatcher[res1])
comp2 = chickadee.PyOptSparseComponent('comp2', comp2_cap, comp2_ramp, res1,
                                       comp2_trans, comp2_cost, produces=res1,
                                       guess=comp2_guess)
# Should use given guess values
assert np.array_equal(comp2.guess, comp2_guess)


# ---------------- Pool Constraint Tests ----------------
comps = [comp1, comp2]
dispatcher.components = comps
dispatchState = chickadee.DispatchState(comps, time_horizon)

# Should return zero for when balanced
pool_cons = dispatcher._gen_pool_cons(res1)
assert pool_cons(dispatchState) == 0.0
# Should square the errors
dispatchState.state[comp1.name][res1][0] = 100.0
assert pool_cons(dispatchState) == 10000.0
# Should not allow errors at different time point to cancel
dispatchState.state[comp2.name][res1][1] = -100.0
assert pool_cons(dispatchState) == 20000.0

# ---------------- determine_dispatch tests ----------------
opt_vars = {
    comp1.name: 0.5*np.ones(n),
    comp2.name: np.ones(n)
}

# FIXME: Add test of external objective function
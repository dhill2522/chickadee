import chickadee
import gekko
import numpy as np
import matplotlib.pyplot as plt

smr_cost_cap = 1.0
smr_cost_fixed_om = 1.0
smr_cost_var_om = 1.0
def smr_transfer(m):
    return m
smr = chickadee.GekkoComponent('smr', smr_cost_cap, smr_cost_fixed_om, smr_cost_var_om, smr_transfer)
    
turb_cost_cap = 1.0
turb_cost_fixed_om = 1.0
turb_cost_var_om = 1.0
def turb_transfer(m):
    return m
turbine = chickadee.GekkoComponent('smr', turb_cost_cap, turb_cost_fixed_om, turb_cost_var_om, turb_transfer)
    
el_cost_cap = 1.0
el_cost_fixed_om = 1.0
el_cost_var_om = 1.0
def el_transfer(m):
    return m
el_market= chickadee.GekkoComponent('el_market', el_cost_cap, el_cost_fixed_om, el_cost_var_om, el_transfer)

dispatcher = chickadee.Gekko(window_length=5)

comps = [smr, turbine, el_market]
time_horizon = np.linspace(0, 1, 10)
optimal_dispatch = dispatcher.dispatch(comps, time_horizon) 

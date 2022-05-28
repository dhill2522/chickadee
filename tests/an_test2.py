import matplotlib.pyplot as plt
import chickadee
import numpy as np
import time

n = 40  # number of time points
time_horizon = np.linspace(0, n-1, n)

el = 'electricity'
ng = 'natural_gas'  # lb/hr
cw = 'cold_water'
hw = 'hot_water'

def gp_transfer(inputs):
    return {}
def gp_cost(dispatch):
    return 0
gas_producer = chickadee.PyOptSparseComponent('gas_producer', np.ones(n)*1e6, np.ones(n)*1e6, np.ones(n)*1e6,
                    ng, gp_transfer, gp_cost, produces=ng)

def turbine_cost(dispatch: dict) -> float:
    LHV = 46000*0.429923  # BTU/lb
    C_NG = 4.1155*LHV/999761  # $/lb
    return sum(-C_NG*np.array(dispatch[ng])/1e6)  # $/hr

def turbine_transfer(inputs: list) -> list:
    return {}

turbine = chickadee.PyOptSparseComponent('turbine', np.ones(
    n)*1e6, np.ones(n)*1e6, np.ones(n)*1e6, ng, turbine_transfer, turbine_cost, consumes=ng, produces=el)

el_load = 

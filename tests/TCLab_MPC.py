import tclab
import numpy as np
import matplotlib.pyplot as plt
import chickadee
import time

heat = chickadee.Resource('heat')
# Questions:
# 1. Why does this not need sol.time[:9] in line 96
# 2. Should it be taking [0] or [1] on 20
# 3. Why isn't it solving


def get_T():  # in K
    print(a.T1)
    return a.T1 + 273.15

def set_Q(sol):
    print(f"T: {a.T1}, Q: {sol.dispatch['heater'][heat][0]}")
    a.Q1(sol.dispatch['heater'][heat][0])


pl = 10  # Prediction Length
n = 600
t = np.arange(n)*3
T_sp = np.zeros(n)
T_sp[10:50] = 50
T_sp[50:150] = 30
T_sp[150:300] = 80
T_sp[300:500] = 60
T_sp = T_sp + 273.15

ns = 30 # this is to try a shorter time window
T_sp = T_sp[:ns]
t = t[:ns]

def heater_cost(dispatch):
    return 0.001 * sum(dispatch[heat])

def heater_transfer(data):
    return {}

heater_capacity = np.zeros(pl) + 100
heater_ramp = heater_capacity.copy()
heater_guess = np.zeros(pl)
heater = chickadee.PyOptSparseComponent('heater', heater_capacity, heater_ramp, heater_ramp,
                                        heat, heater_transfer, heater_cost, produces=heat,
                                        guess=heater_guess)

def tes_cost(dispatch):
    return sum(dispatch[heat]) 

def tes_transfer(Q, T0):
    Ta = 23.0+273.15     # K
    mass = 4.0/1000.0    # kg
    Cp = 0.5*1000.0      # J/kg-K
    U = 10 
    A = 10.0/100.0**2    # Area not between heaters in m^2
    As = 2.0/100.0**2    # Area between heaters in m^2
    eps = 0.9            # Emissivity
    sigma = 5.67e-8            # Stefan-Boltzmann
    alpha = .01
    # Tnew = T0 + 1/(mass*Cp)*(U*A(Ta - T) + sigma*eps*A*(Ta**4 - T**4) + alpha*Q)
    T_stores = []
    for q in Q:
        Tnew = T0 + 1/(mass*Cp)*(U*A*(Ta - T0) + sigma*eps*A*(Ta**4 - T0**4) + alpha*q)
        T_stores.append(Tnew)
        T0 = Tnew
    return T_stores

tes_capacity = np.zeros(pl) + 373
tes_ramp = np.zeros(pl) + 100
tes_guess = np.zeros(pl) + 320
tes = chickadee.PyOptSparseComponent('tes', tes_capacity, tes_ramp, tes_ramp, heat,
                                     tes_transfer, tes_cost, stores=heat
                                     , guess=tes_guess)
comps = [heater, tes]
measure_funcs = [get_T]
control_funcs = [set_Q]

# a = tclab.TCLab()
# MPC = chickadee.MPC(comps, measure_funcs, control_funcs, "tes", heat, pred_len = pl)
# sol = MPC.control(t, T_sp)
# 
# a.Q1(0)
# a.Q2(0)
# a.close()
# print('Disconecting TClab')

dispatcher = chickadee.PyOptSparse()
sol = dispatcher.dispatch(comps, t[:pl], t[:pl])

plt.subplot(2,1,1)
plt.plot(sol.time, sol.dispatch['tes'][heat], label='TES activity')
plt.plot(sol.time, sol.storage['tes'], label='TES storage level')
plt.plot(sol.time, tes_ramp, label='TES ramp')
ymax = max(sol.storage['tes'])
plt.vlines([w[0] for w in sol.time_windows], 0, ymax, colors='green', linestyles='--')
plt.vlines([w[1] for w in sol.time_windows], 0, ymax, colors='blue', linestyles='--')
plt.legend()

plt.subplot(2,1,2)
plt.plot(sol.time, sol.dispatch['heater'][heat], label='heater')
plt.legend()
plt.show()

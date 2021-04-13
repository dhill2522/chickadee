import tclab
import numpy as np
import matplotlib.pyplot as plt
import chickadee
import time
import pandas as pd

heat = chickadee.Resource('heat')
# Questions:
# 1. 
# 2. 
# 3. 

# prep data file
f = open('TClab_MPC_data.csv', 'w')
f.write('Time,Tsp,T1,Q1\n')

# define functions to interact with system
def get_T():  # in K
    return np.round(a.T1 + 273.15, 2)

def set_Q(Qval):
    a.Q1(Qval)

# create set point and time history
sols = []
pl = 10  # Prediction Length
n = 600
t = np.arange(n)*3
T_sp = np.zeros(n) + 15
T_sp[10:50] = 50
T_sp[50:150] = 30
T_sp[150:300] = 80
T_sp[300:500] = 60
T_sp = T_sp + 273.15

ns = 301 # this is to try a shorter time window
T_sp = T_sp[:ns]
time_horizon = t[:ns]

# Create chickadee model
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
    td = 3               # time delay
    Ta = 23.0+273.15     # K
    mass = 4.0/1000.0    # kg
    Cp = 0.5*1000.0      # J/kg-K
    U = 10 
    A = 10.0/100.0**2    # Area not between heaters in m^2
    As = 2.0/100.0**2    # Area between heaters in m^2
    eps = 0.9            # Emissivity
    sigma = 5.67e-8            # Stefan-Boltzmann
    alpha = .01
    T_stores = []
    for q in Q:
        Tnew = T0 + td*1/(mass*Cp)*(U*A*(Ta - T0) + sigma*eps*A*(Ta**4 - T0**4) + alpha*q)
        T_stores.append(Tnew)
        T0 = Tnew
    return T_stores

tes_capacity = np.zeros(pl) + 373
tes_ramp = np.zeros(pl) + 100
tes_guess = np.zeros(pl) + 320
T0 = 273+25
tes = chickadee.PyOptSparseComponent('tes', tes_capacity, tes_ramp, tes_ramp, heat,
                                     tes_transfer, tes_cost, stores=heat
                                     , guess=tes_guess, storage_init_level=T0)
comps = [heater, tes]

# Creat MPC

# connect to hardware and create dispatcher
a = tclab.TCLab()
dispatcher = chickadee.PyOptSparse(window_length=pl)

# MPC Loop
for i in range(len(time_horizon) - pl):
    ie = pl+i
    start_time = time.time() 

    T0 = get_T() # get current temperature
    comps[1].storage_init_level=T0

    # create custom objective with set point
    def objective(dispatch):
        tes_store = tes_transfer(dispatch.state['heater'][heat], T0)
        return np.sum((tes_store - T_sp[i:ie])**2)
    
    # run optimization on 1 time window
    sol = dispatcher.dispatch(comps, t[:pl], external_obj_func=objective)
    sols.append(sol)
    # output first solution heater value to hardware
    Qval = np.round(sol.dispatch['heater'][heat][0],2) 
    set_Q(Qval)   
    
    # print and output the values
    print(f'Time: {time_horizon[i]}s, Tsp: {T_sp[i]} K, T: {T0} K, Q: {Qval}')
    f.write(f'{time_horizon[i]},{T_sp[i]},{T0},{Qval}\n')

    # wait the rest of the time between time points
    end_time = time.time()
    dt = end_time - start_time
    dt_time_horizon = time_horizon[1] - time_horizon[0]
    time.sleep(dt_time_horizon - dt) 

a.Q1(0)
a.Q2(0)
a.close()
f.close()   
print('Disconecting TClab')

plt.figure(1)
data = pd.read_csv('TClab_MPC_data.csv')
plt.plot(time_horizon[:-10]/3, data['T1'].values, '.', label='actual')
plt.plot(time_horizon/3, T_sp, label='setpoint')
plt.legend()

# plt.figure(2)
# si = 0
# plt.subplot(2,1,1)
# plt.plot(sol[si].time/3, sol[si].dispatch['tes'][heat], label='TES activity')
# plt.plot(sol[si].time/3, sol[si].storage['tes'], label='TES storage level')
# plt.plot(sol[si].time/3, T_sp[:pl], label='TES setpoint')
# ymax = max(sol[si].storage['tes'])
# plt.vlines([w[0] for w in sol[si].time_windows], 0, ymax, colors='green', linestyles='--')
# plt.vlines([w[1] for w in sol[si].time_windows], 0, ymax, colors='blue', linestyles='--')
# plt.legend()
# 
# plt.subplot(2,1,2)
# plt.plot(sol[si].time, sol[si].dispatch['heater'][heat], label='heater')
# plt.legend()
plt.show()



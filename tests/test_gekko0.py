from gekko import GEKKO
from collections.abc import Iterable
import copy
import numpy as np
import matplotlib.pyplot as plt

m = GEKKO(remote=False)

m.time = np.linspace(0, 5, 501)

load = m.Param(np.cos(2*np.pi*m.time)+3)
gen = m.Var(load[0])

err = m.CV(0)
err.STATUS = 1
err.SPHI = err.SPLO = 0
err.WSPHI = 1000
err.WSPLO = 1

dgen = m.MV(0, lb=-1, ub=1)
dgen.STATUS = 1

m.Equations([gen.dt() == dgen,  err == load-gen])
m.options.IMODE = 6

original_model = copy.deepcopy(m)

# Run a "first window"
# Need to be able to adjust the time horizons covered
# Try shrinking the time horizon to the first lnew points in a deepcopy problem
lnew = 87

def adjust_window(original_model, start, end, prev_win=None):
    '''Note that this requires the prev_win to have already been solved'''

    # Create the new model
    model = copy.deepcopy(original_model)
    model._model_name = original_model._model_name + f'_win_{start}_{end}'
    model.time = original_model.time[start:end]

    # Adjust the time series variables and parameters
    for v in model._variables + model._parameters:
        # Robustly determine the "length" of the GK_Value
        try:
            val = v.value.value.value
        except:
            try:
                val = v.value.value
            except:
                val = v.value
        l = len(val) if isinstance(val, Iterable) else 1
        if l > 1:
            v.value = v.value[start:end]

    if prev_win:
        # Set the new window to start where the previous window ended
        for i, v in enumerate(model._variables):
            try:
                val = v.value.value.value
            except:
                try:
                    val = v.value.value
                except:
                    val = v.value
            l = len(val) if isinstance(val, Iterable) else 1
            try:
                v.value[0] = prev_win._variables[i].value[-1]
            except:
                v.value = prev_win._variables[i].value[-1]
    return model


m.solve(disp=False)
print('Original problem solved:')

win0 = adjust_window(original_model, 0, lnew)
win0.solve(disp=False)
print('Window 0 solved:')

win1 = adjust_window(original_model, lnew-1, 2*lnew, win0)
win1.solve(disp=False)
print('Window 1 solved:')

double = adjust_window(original_model, 0, 2*lnew)
double.solve(disp=False)
print('Double window problem solved:')

plt.subplot(2, 1, 1)
plt.plot(m.time, load, label='load(m)')
plt.plot(win0.time, win0._parameters[0], label='win 0')
plt.plot(win1.time, win1._parameters[0], label='win 1')
plt.plot(double.time, double._parameters[0], label='double')
plt.legend()

plt.subplot (2, 1, 2)
plt.plot(m.time, gen, label='gen(m)')
plt.plot(win0.time, win0._variables[0], label='win 0')
plt.plot(win1.time, win1._variables[0], label='win 1')
plt.plot(double.time, double._variables[0], label='double')
plt.legend()
plt.show()


# Compare results to the first two windows run together
# double = copy.deepcopy(original_model)
# double._model_name = original_model._model_name + '_double'
# double.time = original_model.time[:2*lnew]

# Will need to be able to run "rolling-window" dispatch with GEKKO

# Need to find which vars are capacities to be optimized

# Need to be able to adjust the capacities dynamically

# Key lesson: Any edited models must be based on a deepcopy of the model before it is run.

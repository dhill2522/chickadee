import chickadee
from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

m = GEKKO(remote=False)

m.time = np.linspace(0, 1, 101)

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

# cap_comps = {
#     gen: {'initial': 1, 'min': 0, 'max': 100}
# }

dispatcher = chickadee.GekkoDispatcher(m, 10)

windows = dispatcher.dispatch()

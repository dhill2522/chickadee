from .Dispatcher import Dispatcher
from .Component import PyOptSparseComponent
from .TimeSeries import TimeSeries
from .Solution import Solution

import pyoptsparse
import numpy as np
import sys
import os
import time as time_lib
import traceback
from typing import List
from itertools import chain
from pprint import pformat
from gekko import GEKKO

class DispatchState(object):
    '''Modeled after idaholab/HERON NumpyState object'''
    def __init__(self, components: List[PyOptSparseComponent], time: List[float]):
        s = {
        }

        for c in components:
            s[c.name] = {}
            for resource in c.get_resources():
                s[c.name][resource] = np.zeros(len(time))

        self.state = s
        self.time = time

    def set_activity(self, component: PyOptSparseComponent, resource, activity, i=None):
        if i is None:
            self.state[component.name][resource] = activity
        else:
            self.state[component.name][resource][i] = activity

    def get_activity(self, component: PyOptSparseComponent, resource, i=None):
        try:
            if i is None:
                return self.state[component.name][resource]
            else:
                return self.state[component.name][resource][i]
        except Exception as err:
            print(i)
            raise err

    def set_activity_vector(self, component: PyOptSparseComponent,
                            resource, start, end, activity):
        self.state[component.name][resource][start:end] = activity

    def __repr__(self):
        return pformat(self.state)


class Gekko(Dispatcher):
    def __init__(self, window_length=10):
        self.name = 'GekkoDispatcher'
        self._window_length = window_length
        

    def dispatch(self, components, time):
        # Note: Using normal PyOptSparseComponents for right now...
        # Will need to define a component that has economic parameters attached for calculating the LCOE
        self.components = components
        self.time = time

        m = GEKKO()
        # Will run the transfer functions once in order to determine the model
        for comp in self.components:
            # Note: we are only expecting one of the functions to actually do anything. That allows 
            # the model to be defined as cohesively and flexibly as possible. We are expecting that 
            # single component to effectively define the dynamics of the entire system
            comp.transfer(m)

        # Will assemble an objective function that calculates the system LCOE
        m.Obj()
        m.options.IMODE = 5
        # FIXME: Will want to make sure this solves local.... :)

        # Will optimize the system dispatch to minimize the system LCOE
        m.solve()
        # FIXME: Handle failed runs by raising the right error

        # Return the final dispatch
        # FIXME: Assemble the final dispatch
        pass

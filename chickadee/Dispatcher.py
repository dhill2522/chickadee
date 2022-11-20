from .Component import Component

from gekko import GEKKO
import numpy as np
from pprint import pformat
from typing import List

class Dispatcher(object):
    def __init__(self):
        super(Dispatcher, self).__init__()

    def dispatch(self, components, time, time_series):
        raise NotImplementedError('Use a subclass of Dispatcher instead')


class DispatchState(object):
    '''Modeled after idaholab/HERON NumpyState object'''

    def __init__(self, components: List[Component], time: List[float]):
        s = {}

        for c in components:
            s[c.name] = {}
            for resource in c.get_resources():
                s[c.name][resource] = np.zeros(len(time))

        self.state = s
        self.time = time

    def set_activity(self, component: Component, resource, activity, i=None):
        if i is None:
            self.state[component.name][resource] = activity
        else:
            self.state[component.name][resource][i] = activity

    def get_activity(self, component: Component, resource, i=None):
        try:
            if i is None:
                return self.state[component.name][resource]
            else:
                return self.state[component.name][resource][i]
        except Exception as err:
            print(i)
            raise err

    def set_activity_vector(self, component: Component,
                            resource, start, end, activity):
        self.state[component.name][resource][start:end] = activity

    def __repr__(self):
        return pformat(self.state)

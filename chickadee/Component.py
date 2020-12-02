from .Resource import Resource

from typing import List, Callable
import numpy as np


class Component(object):
    def __init__(self, name, capacity, capacity_resource, transfer, econ_param, produces=None, consumes=None, stores=None,
                dispatch_type='independent'):
        """A generic Component
        :param name: Name of the component. Used in representing dispatches
        :param capacity: Maximum capacity of the component in terms of `capacity_resource`
        :param transfer: a method for calculating the component transfer at a time point
        :param econ_param: an economic parameter describing the economics of running the unit
        :param produces: a list of resources produced by the component
        :param consumes: a list of resources consumed by the component
        :param stores: a list of resources stored by the component
        :param dispatch_type: the dispatch type of the component
        """
        super(Component, self).__init__()
        self.name = name
        self.capacity = capacity
        self.capacity_resource = capacity_resource
        self.transfer = transfer
        if type(produces) == list:
            self.produces = produces
        else:
            self.produces = [produces]
        if type(stores) == list:
            self.stores = stores
        else:
            self.stores = [stores]
        if type(consumes) == list:
            self.consumes = consumes
        else:
            self.consumes = [consumes]
        self.dispatch_type = dispatch_type

        # Check to make sure that it does interact with at least one resource
        if produces is None and consumes is None and stores is None:
            raise RuntimeWarning('This component does not interact with any resource!')


    def get_resources(self):
        return [t for t in set([*self.produces, *self.stores, *self.consumes]) if t]


class PyOptSparseComponent(Component):
    def __init__(self, name: str, capacity: np.ndarray, ramp_rate: np.ndarray,
                capacity_resource: Resource, transfer: Callable, cost_function: Callable,
                produces=None, consumes=None, stores=None, min_capacity=None, dispatch_type: str='independent',
                guess: np.ndarray=None):
        """A Component compatible with the PyOptSparse dispatcher
        :param name: Name of the component. Used in representing dispatches
        :param capacity: Maximum capacity of the component in terms of `capacity_resource`
        :param ramp_rate: the maximum ramp rate of the component in terms of capacity resource units per time
        :param transfer: a method for calculating the component transfer at a time point
        :param cost_function: an function describing the economic cost of running the unit over a given dispatch
        :param produces: a list of resources produced by the component
        :param consumes: a list of resources consumed by the component
        :param stores: a list of resources stored by the component
        :param min_capacity: Minimum capacity of the unit at each time point. Defaults to 0.
        :param dispatch_type: the dispatch type of the component
        :param guess: a guess at the optimal dispatch of the unit in terms of its `capacity_resource`. Defaults to the capacity.

        Not that a component cannot store resources in addition to producing or consuming
        resources. Storage components must be separate from producers and consumers.

        Components can store multiple resources simultaneously.
        """
        super(PyOptSparseComponent, self).__init__(name, capacity, capacity_resource,
                                                transfer, 0.0, produces, consumes,
                                                stores, dispatch_type)

        if type(capacity) is not np.ndarray:
            raise TypeError(
                'PyOptSparseComponent capacity must be a numpy array')
        if type(ramp_rate) is not np.ndarray:
            raise TypeError(
                'PyOptSparseComponent ramp_rate must be a numpy array')

        if guess is None:
            self.guess = self.capacity
        else:
            if type(guess) is not np.ndarray:
                raise TypeError(
                    'PyOptSparseComponent guess must be a numpy array')
            self.guess = guess

        if min_capacity is None:
            self.min_capacity = np.zeros(len(self.capacity))
        else:
            # Check the datatype
            if type(min_capacity) is not np.ndarray:
                raise TypeError('min_capacity must be a numpy array')
            self.min_capacity = min_capacity

        self.ramp_rate = ramp_rate
        self.cost_function = cost_function

        # Set up the storage tracking dict
        # This is different than the activities. Activities are the change at each point
        # in time. Storage represents the amount of a resoure that is currently being
        # stored.
        self.storage_level = {res: np.zeros(len(self.capacity)) for res in self.stores}

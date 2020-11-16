from .Resource import Resource

from typing import List, Callable

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
    def __init__(self, name: str, capacity: float, ramp_rate: float,
                capacity_resource: Resource, transfer: Callable, econ_param: float,
                produces=None, consumes=None, stores=None, dispatch_type: str='independent'):
        """A Component compatible with the PyOptSparse dispatcher
        :param name: Name of the component. Used in representing dispatches
        :param capacity: Maximum capacity of the component in terms of `capacity_resource`
        :param ramp_rate: the maximum ramp rate of the component in terms of capacity resource units per time
        :param transfer: a method for calculating the component transfer at a time point
        :param econ_param: an economic parameter describing the economics of running the unit
        :param produces: a list of resources produced by the component
        :param consumes: a list of resources consumed by the component
        :param stores: a list of resources stored by the component
        :param dispatch_type: the dispatch type of the component
        """
        super(PyOptSparseComponent, self).__init__(name, capacity, capacity_resource,
                                                transfer, econ_param, produces, consumes,
                                                stores, dispatch_type)
        self.ramp_rate = ramp_rate

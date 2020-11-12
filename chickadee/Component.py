class Component(object):
    def __init__(self, name, capacity, capacity_resource, transfer, econ_param, produces=None, consumes=None, stores=None,
                dispatch_type='independent'):
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

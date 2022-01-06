from gekko import GEKKO

class Dispatcher(object):
    def __init__(self):
        super(Dispatcher, self).__init__()

    def dispatch(self, components, time, time_series):
        raise NotImplementedError('Use a subclass of Dispatcher instead')

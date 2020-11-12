from gekko import GEKKO

class Dispatcher(object):
    def __init__(self):
        super(Dispatcher, self).__init__()

    def dispatch(self, components, time, time_series):
        raise NotImplementedError('Use a subclass of Dispatcher instead')


class GekkoDispatcher(Dispatcher):
    def __init__(self):
        super(GekkoDispatcher, self).__init__()
        self.m = GEKKO(remote=False)

    def dispatch(self, components, time, time_series):
        self.m.time = time
        return {}
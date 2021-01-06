class Solution(object):
    time = None
    dispatch = None
    storage = None
    error = False

    def __init__(self, time, dispatch, storage, error, objval, time_windows=None):
        '''Representation of the problem solution.
        :param time: time horizon used in the problem
        :param dispatch: an object/dict describing the optimal dispatch of the system
        :param storage: an object/dict describing the usage of storage over the time horizon
        :param error: the total constraint error of the final solution
        :param objval: the final value of the objective function
        :param time_windows: description of where the involved windows start and end
        '''
        self.time = time
        self.dispatch = dispatch
        self.storage = storage
        self.error = error
        self.objval = objval
        self.time_windows = time_windows

        # FIXME: Could add things like solvetime, constraint violations, timing breakdown...
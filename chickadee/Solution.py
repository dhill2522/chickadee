class Solution(object):
    time = None
    dispatch = None
    storage = None
    error = False

    def __init__(self, time, dispatch, storage, error, time_windows=None):
        self.time = time
        self.dispatch = dispatch
        self.storage = storage
        self.error = error
        self.time_windows = time_windows

        # FIXME: Could add things like solvetime, constraint violations, timing breakdown...
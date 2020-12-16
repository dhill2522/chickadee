class Solution(object):
    time = None
    dispatch = None
    storage = None
    error = False

    def __init__(self, time, dispatch, storage, error):
        self.time = time
        self.dispatch = dispatch
        self.storage = storage
        self.error = error

        # FIXME: Could add things like solvetime, constraint violations, timing breakdown...
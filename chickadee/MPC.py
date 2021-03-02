''' Implementation of MPC using Chickadee models'''


class ModelPredictiveController(object):

    def __init__(self, components, dispatcher, pred_len=10):
        self.time_horizon = time_horizon
        self.components = components
        self.dispatcher = dispatcher
        self.pred_len = pred_len

    def _opt_pred(self):
        '''Optimize the model response '''
        pass


    def control(self, time_horizon, set_point):
        '''Control a realtime system using the given model
        Will consist of a loop of the following executed for each step in the time horizon:
            - wait for the chosen timestep (a configureable variable)
            - measure the real system response (this will need to come from a user-supplied function)
            - optimize the system response as predicted by the model
            - apply the first set of control moves as suggested by the optimal system input
        '''

        pass

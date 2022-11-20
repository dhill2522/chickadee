''' Implementation of MPC using Chickadee models'''
from .PyOptSparseDispatch import PyOptSparse, DispatchState
import time
import numpy as np

class MPC(object):

    def __init__(self, components, measurement_funcs,
		 control_funcs, component_name, cv_resource, pred_len=10):
        self.components = components
        self.measure_funcs = measurement_funcs
        self.control_funcs = control_funcs
        self.component_name = component_name
        self.cv_resource = cv_resource
        self.pred_len = pred_len
        self.solution = []


    def control(self, time_horizon, set_point):
        '''Not Implemented yet.

        Control a realtime system using the given model
        Will consist of a loop of the following executed for each step in the time horizon:
            - wait for the chosen timestep (a configureable variable)
            - measure the real system response (this will need to come from a user-supplied function)
            - optimize the system response as predicted by the model
            - apply the first set of control moves as suggested by the optimal system input
        '''
        raise Exception("MPC control is not implemented yet in Chickadee.")
        # Make data arrays

        # Make obj function
        def objective(dispatch: DispatchState) -> float:
            cv_disp = dispatch.state[self.component_name][self.cv_resource]
            return np.sum((cv_disp - set_point)**2)

        # Iterate over time horizon
        # FIXME: This currently stops 1 prediction length before end of time horizon
        for i in range(len(time_horizon) - self.pred_len):
            start_time = time.time()
            ie = i + self.pred_len  # This is the end index for prediction window

            # Run the optimization
            dispatcher = PyOptSparse(window_length=self.pred_len)
            sol = dispatcher.dispatch(self.components, time_horizon[i:ie], objective)

            # Apply the control step from the optimized dispatch
            for f in self.control_funcs:
                f(sol)

            # Update components with current state
            for c in self.components:
                if len(c.capacity) != self.pred_len:
                    raise BaseException('Inputs for constraints need to have same length as pred_length')
            self.solution.append(sol)

            # Wait the dt time in time history (assumes uniform dt)
            # dt changes based on the time steps in the time horizon
            end_time = time.time()
            dt = end_time - start_time
            dt_time_horizon = time_horizon[1] - time_horizon[0]
            sleep_time = dt_time_horizon - dt
            if sleep_time < 0:
               raise BaseException("Warning you have tried to bend the rules of time you need to make your time steps bigger or shorten prediction window")
            time.sleep(dt_time_horizon - dt)
        return self.solution

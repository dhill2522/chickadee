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
        '''Control a realtime system using the given model
        Will consist of a loop of the following executed for each step in the time horizon:
            - wait for the chosen timestep (a configureable variable)
            - measure the real system response (this will need to come from a user-supplied function)
            - optimize the system response as predicted by the model
            - apply the first set of control moves as suggested by the optimal system input
        '''
        # Make data arrays

        # Iterate over time horizon
        # FIXME: This currently stops 1 prediction length before end of time horizon
        if len(time_horizon) - self.pred_len <= 0:
            raise Exception('Time horizon needs to be bigger than prediction length')
        for i in range(len(time_horizon) - self.pred_len):      
            start_time = time.time()
            ie = i + self.pred_len  # This is the end index for prediction window

            # Make obj function
            def objective(dispatch: DispatchState) -> float:
                cv_disp = dispatch.state[self.component_name][self.cv_resource]
                return np.sum((cv_disp - set_point[i:ie])**2)


            # Run the optimization
            dispatcher = PyOptSparse(window_length=self.pred_len)
            sol = dispatcher.dispatch(self.components, time_horizon[i:ie], external_obj_func=objective)
            # Adjust new storage temperature -> FIXME: This will need to be fixed for different MPC applications
            for f in self.measure_funcs:
                self.components[1].storage_init_level = f() 

            # Apply the control step from the optimized dispatch
            for f in self.control_funcs:
                f(sol)    

            # Update components with current state
            for c in self.components:
                if len(c.capacity) != self.pred_len:
                    raise Exception('Inputs for constraints need to have same length as pred_length')
            self.solution.append(sol)
            # Wait the dt time in time history (assumes uniform dt) 
            # dt changes based on the time steps in the time horizon
            end_time = time.time()
            dt = end_time - start_time
            dt_time_horizon = time_horizon[1] - time_horizon[0]
            sleep_time = dt_time_horizon - dt
            if sleep_time < 0:
               raise Exception("Warning you have tried to bend the rules of time you need to make your time steps bigger or shorten prediction window")
            time.sleep(dt_time_horizon - dt)
        return self.solution

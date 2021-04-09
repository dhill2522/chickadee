''' Implementation of MPC using Chickadee models'''
from .PyOptSparseDispatch import PyOptSparse, DispatchState
import time
import numpy as np

class MPC(object):

    def __init__(self, components, dispatcher, measurement_funcs, 
		 control_funcs, component_name, cv_resource, pred_len=10):
        self.components = components
        self.dispatcher = dispatcher
        self.measure_funcs = measurement_funcs
        self.control_funcs = control_funcs
        self.component_name = component_name
        self.cv_resource = cv_resource
        self.pred_len = pred_len
        print('capacity init:', self.components[0].capacity)


    def control(self, time_horizon, set_point):
        '''Control a realtime system using the given model
        Will consist of a loop of the following executed for each step in the time horizon:
            - wait for the chosen timestep (a configureable variable)
            - measure the real system response (this will need to come from a user-supplied function)
            - optimize the system response as predicted by the model
            - apply the first set of control moves as suggested by the optimal system input
        '''
        # Make data arrays

        # Make obj function
        def objective(dispatch: DispatchState) -> float:
            cv_disp = dispatch.state[self.component_name][self.cv_resource]
            return np.sum((cv_disp - set_point)**2)

        for _ in time_horizon:
            # Read in current state from measurement functions
            start_time = time.time()

            # Run the optimization
            dispatcher = PyOptSparse(window_length=self.pred_len)
            print('capacity:', self.components[0].capacity)
            #FIXME: some how the constraint values are getting deleted
            sol = dispatcher.dispatch(self.components, time_horizon, objective)
    
                # Apply the control step from the optimized dispatch
            for f in self.control_funcs:
                f(sol) # FIXME need to pull heater value out 
    
            # Update components with current state
            for c in self.components:
                c.capacity = np.delete(c.capacity, 0)
                # Repeat for all arrays that need to be shortened
            np.append(c.capacity, self.measure_funcs[0])

            print(f'time: {_}, T: {a.T1}')
            # Wait the dt time in time history (assumes uniform dt) 
            end_time = time.time()
            dt = end_time - start_time
            dt_time_horizon = time_horizon[0] - time_horizon[1]
            time.sleep(dt_time_horizon - dt)

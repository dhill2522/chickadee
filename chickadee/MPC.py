''' Implementation of MPC using Chickadee models'''
from PyOptSparseDispatch import PyOptSparse, DispatchState


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


    def control(self, time_horizon, set_point):
        '''Control a realtime system using the given model
        Will consist of a loop of the following executed for each step in the time horizon:
            - wait for the chosen timestep (a configureable variable)
            - measure the real system response (this will need to come from a user-supplied function)
            - optimize the system response as predicted by the model
            - apply the first set of control moves as suggested by the optimal system input
        '''
	
	import time

        # Make data arrays

        # Make obj function
        def objective(dispatch: DispatchState) -> float:
            cv_disp = dispatch.state[self.component_name][self.cv_resource]
    	    return np.sum((cv_disp - set_point)**2)

        for i, t in enumerate(time_horizon, set_point):
            # Read in current state from measurement functions
	    start_time = time.time()
            # Update components with current state
            for j, c in enumerate(self.components):
                c.ramp_rate = np.delete(c.ramp_rate, 0)
                # Repeat for all arrays that need to be shortened
		c.capacity.append(self.measure_funcs[j]) # This assumes measure funcs were added in same order as components

            # Run the optimization
            sol = PyOptSparse.dispatch(self.components, self.time_horizon, objective)

            # Apply the control step from the optimized dispatch
	    for f in self.control_funcs:
	        f(sol) 

            # Wait the dt time in time history (assumes uniform dt) 
	    end_time = time.time()
	    dt = end_time - start_time
	    dt_time_horizon = time_horizon[0] - time_horizon[1]
	    time.sleep(dt_time_horizon - dt)

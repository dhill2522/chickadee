''' Implementation of MPC using Chickadee models'''
from PyOptSparseDispatch import PyOptSparse, DispatchState


class MPController(object):

    def __init__(self, components, dispatcher, measurement_funcs, 
                 component_name, cv_resource, pred_len=10):
        self.components = components
        self.dispatcher = dispatcher
        self.measure_funcs = measurement_funcs
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
        # Make data arrays

        # Make obj function
        def objective(dispatch: DispatchState) -> float:
            cv_disp = dispatch.state[self.component_name][self.cv_resource]
    		return np.sum((cv_disp - set_point)**2)

        for i, t in enumerate(time_horizon, set_point):
            # Read in current state from measurement functions

            # Update components with current state
            for c in self.components:
                c.ramp_rate = np.delete(c.ramp_rate, 0)
                # Repeat for all arrays that need to be shortened

            # Run the optimization
            sol = PyOptSparse.dispatch(self.components, self.time_horizon, objective)
            # Apply the control step from the optimized dispatch

            # Wait the 

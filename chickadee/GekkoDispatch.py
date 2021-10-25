from .Dispatcher import Dispatcher
from .Solution import Solution

from collections.abc import Iterable
import copy
import numpy as np
from gekko import GEKKO

def get_var_len(v):
    '''Gets the length of a GK_Value'''
    try:
        val = v.value.value.value
    except:
        try:
            val = v.value.value
        except:
            val = v.value
    return len(val) if isinstance(val, Iterable) else 1

class GekkoDispatcher(Dispatcher):
    def __init__(self, model, window_length, capacity_components=[]):
        '''Make a GEKKO dispatcher
        model: GEKKO object describing the model
        window_length: float, length of the internal dispatch window
        capacity_components: dict, a dict of dicts describing the initial, min and max values
        '''
        #m.get_names() # This will get the names used to define the variables in
                      # the script. It allows the mapping between the script
                      # name and the variable object itself.
        super(GekkoDispatcher, self).__init__()
        self.original_model = model # Only modify deepcopies of this
        self.window_length = window_length
        self.cc = capacity_components
        # FIXME: Need to check and make sure that all the time series lengths are the same

    # def comb_opt(self, time_series, capacity_components):
    #     '''performs combined dispatch and capacity optimization
    #     Returns the optimal objective function value and capacity sizes
    #     Should it return the dispatch or ?'''
    #
    #     # Update the capacity limits
    #     # can use the .name property of m._variables to match back to the user input
    #
    #     # Update the time series
    #     try:
    #         self.m.solve()
    #     except Exception as err:
    #         print(err)
    #         raise Exception('Combined Optimization failed:')
    #
    #     # Return the right results
    #
    #
    # def capacity_opt(self):
    #     '''Estimate the optimal unit capacities
    #     Optimizes the capacity and dispatch for all of the time windows.'''
    #
    #     # Split the entire time horizon up into windows
    #     win_start_i = 0
    #     win_i = 0
    #     prev_win_end_i = 0
    #     time_windows = []
    #     while win_start_i < self.tfinal:
    #         win_end_i = win_start_i + self.wl
    #         time_windows.append({'i': win_i, 'start': win_start_i, 'end': win_end_i})
    #
    #
    #     # Run the combined optimization problem on each windows, preferably in parrallel
    #     opt_capacities = []
    #     for window in time_windows:
    #         # Put together the time series to use
    #         local_ts = {k, v[window['start']:window['end']] for key, value in dict.items()}
    #         comb_opt(local_ts, self.cc)
    #         # FIXME: apend to the opt_capacities array
    #
    #     # Take the max (for now) from each of the output capacity distributions
    #     return

    def adjust_window(self, start, end, prev_win=None):
        '''Generate a new copy of the original model with an adjusted time horizon
        start: start index for the new window
        end: end index for the new window
        prev_win: the preceding window. See note below.

        Use of prev_win
        Used to fix the initial values of the
        new window to the final values of the preceding one
        Note that this requires the prev_win to have already been solved'''

        # Create the new model
        model = copy.deepcopy(self.original_model)
        model._model_name = self.original_model._model_name + f'_win_{start}_{end}'
        model.time = self.original_model.time[start:end]

        # Adjust the time series variables and parameters
        for v in model._variables + model._parameters:
            # Robustly determine the "length" of the GK_Value
            try:
                val = v.value.value.value
            except:
                try:
                    val = v.value.value
                except:
                    val = v.value
            l = len(val) if isinstance(val, Iterable) else 1
            if l > 1:
                v.value = v.value[start:end]

        if prev_win:
            # Set the new window to start where the previous window ended
            # FIXME: Check if prev_win has been solved
            for i, v in enumerate(model._variables):
                try:
                    val = v.value.value.value
                except:
                    try:
                        val = v.value.value
                    except:
                        val = v.value
                l = len(val) if isinstance(val, Iterable) else 1
                try:
                    v.value[0] = prev_win._variables[i].value[-1]
                except:
                    v.value = prev_win._variables[i].value[-1]
        return model

    def dispatch(self):
        # Break up the time horizon into windows of the right size by index
        windows = []
        max_i = len(self.original_model.time)
        i = 0
        while i < max_i:
            start = i
            end = start + self.window_length + 1
            if end > max_i:
                end = max_i
            # Don't try and dispatch a window with only two time points!
            if (end - start) < 2:
                windows[-1]['end'] = end
            else:
                windows.append({
                    'start': start,
                    'end': end
                })
            i += self.window_length

        # Solve and generate each window sequentially
        # Note that the preceding window MUST be solved before the next window
        # can be generated to allow them to connect
        for i, window in enumerate(windows):
            if i == 0:
                model = self.adjust_window(window['start'], window['end'])
            else:
                model = self.adjust_window(window['start'], window['end'], windows[i-1]['model'])

            print(f'Dispatching window from {window["start"]} to {window["end"]}...', end='')
            model.solve(disp=False)
            print('\tComplete')
            window['model'] = model

        # Assemble a final combined solution
        # We need to mutate and return the original model as that is the one the
        # user script variable names refer to.
        m = self.original_model

        # Now we add the right data as if we solved the original model
        for i, w in enumerate(windows):
            for j, v in enumerate(w['model']._variables):
                l = get_var_len(v)
                if l > 1:
                    # Becuase the original model has not yet been solved some array vars are
                    # still just floats. Need to make them arrays to hold the full solution
                    if i == 0:
                        m._variables[j].value = np.zeros(len(m.time))
                    m._variables[j].value[w['start']:w['end']] = v
                else:
                    m._variables[j].value = v

            for j, p in enumerate(w['model']._parameters):
                l = get_var_len(p)
                if l > 1:
                    if i == 0:
                        m._parameters[j].value = np.zeros(len(m.time))
                    m._variables[j].value[w['start']:w['end']] = p
                else:
                    m._variables[j].value = p
        return m


# Need a way to handle the time series involved
# Determine approximate optimal sizing first

# Do independent dispatch for each of the windows
# Determine the max(?) part of the resulting distribution of optimal sizes
# Run rolling-window dispatch over the entire horizon

# All the time series will need to be defined in __init__. If any are left in the
# model they will cause an array length mismatch

# Assume that the entire time horizon is an even multiple of the window_length?
# It could cause strange optimal capacities for the final window if it is not

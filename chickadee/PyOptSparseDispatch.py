from .Dispatcher import Dispatcher
from .Component import PyOptSparseComponent
from .TimeSeries import TimeSeries
from .Solution import Solution

import pyoptsparse
import numpy as np
import sys
import os
import time as time_lib
import traceback
from typing import List
from itertools import chain
from pprint import pformat

class DispatchState(object):
    '''Modeled after idaholab/HERON NumpyState object'''
    def __init__(self, components: List[PyOptSparseComponent], time: List[float]):
        s = {}

        for c in components:
            s[c.name] = {}
            for resource in c.get_resources():
                s[c.name][resource] = np.zeros(len(time))

        self.state = s
        self.time = time

    def set_activity(self, component: PyOptSparseComponent, resource, activity, i=None):
        if i is None:
            self.state[component.name][resource] = activity
        else:
            self.state[component.name][resource][i] = activity

    def get_activity(self, component: PyOptSparseComponent, resource, i=None):
        try:
            if i is None:
                return self.state[component.name][resource]
            else:
                return self.state[component.name][resource][i]
        except Exception as err:
            print(i)
            raise err

    def set_activity_vector(self, component: PyOptSparseComponent,
                            resource, start, end, activity):
        self.state[component.name][resource][start:end] = activity

    def __repr__(self):
        return pformat(self.state)



class PyOptSparse(Dispatcher):
    '''
    Dispatch using pyOptSparse optimization package and a pool-based method.
    '''

    slack_storage_added = False # In case slack storage is added in a loop

    def __init__(self, window_length=10):
        self.name = 'PyOptSparseDispatcher'
        self._window_length = window_length

        # Defined on call to self.dispatch
        self.components = None
        self.case = None

    def _gen_pool_cons(self, resource) -> callable:
        '''A closure for generating a pool constraint for a resource
        :param resource: the resource to evaluate
        :returns: a function representing the pool constraint
        '''

        def pool_cons(dispatch_window: DispatchState) -> float:
            '''A resource pool constraint
            Checks that the net amount of a resource being consumed, produced and
            stored is zero.
            :param dispatch_window: the dispatch to evaluate
            :returns: SSE of resource constraint violations
            '''
            time = dispatch_window.time
            n = len(time)
            err = np.zeros(n)

            # FIXME: This is an inefficient way of doing this. Find a better way
            cs = [c for c in self.components if resource in c.get_resources()]
            for i, _ in enumerate(time):
                for c in cs:
                    if c.stores:
                        err[i] += -dispatch_window.get_activity(c, resource, i)
                    else:
                        err[i] += dispatch_window.get_activity(c, resource, i)

            # FIXME: This simply returns the sum of the errors over time. There
            # are likely much better ways of handling this.
            # Maybe have a constraint for the resource at each point in time?
            # Maybe use the storage components as slack variables?
            return sum(err**2)

        return pool_cons

    def _build_pool_cons(self) -> List[callable]:
        '''Build the pool constraints
        :returns: List[callable] a list of pool constraints, one for each resource
        '''

        cons = []
        for res in self.resources:
            # Generate the pool constraint here
            pool_cons = self._gen_pool_cons(res)
            cons.append(pool_cons)
        return cons

    def determine_dispatch(self, opt_vars: dict, time: List[float],
                            start_i: int, end_i: int) -> DispatchState:
        '''Determine the dispatch from a given set of optimization
        vars by running the transfer functions. Returns a Numpy dispatch
        object
        :param opt_vars: dict, holder for all the optimization variables
        :param time: list, time horizon to dispatch over
        :returns: DispatchState, dispatch of the system
        '''
        # Initialize the dispatch
        dispatch = DispatchState(self.components, time)
        # Dispatch the fixed components
        fixed_comps = [c for c in self.components if c.dispatch_type == 'fixed']
        for f in fixed_comps:
            dispatch.set_activity(f, f.capacity_resource,
                                  f.capacity[start_i:end_i])
        # Dispatch the independent and dependent components using the vars
        disp_comps = [
            c for c in self.components if c.dispatch_type != 'fixed']
        for d in disp_comps:
            for i in range(len(time)):
                request = {d.capacity_resource: opt_vars[d.name][i]}
                bal, self.meta = d.transfer(request, self.meta)
                for res, value in bal.items():
                    dispatch.set_activity(d, res, value, i)
        return dispatch

    def _dispatch_pool(self) -> DispatchState:
        '''Dispatch the given system using a resource-pool method
        :returns: DispatchState, the optimal dispatch of the system
        :returns: dict, the storage levels of the storage components

         Steps:
           - Assemble all the vars into a vars dict
             A set of vars for each dispatchable component including storage elements
             include bound constraints
           - For each time window
               1) Build the pool constraint functions
                   Should have one constraint function for each pool
                   Each constraint will be a function of all the vars
               2) Set up the objective function as the double integral of the incremental dispatch
               3) Formulate the problem for pyOptSparse
                   a) Declare the variables
                   b) Declare the constraints
                   c) Declare the objective function
                   d) set the optimization configuration (IPOPT/SNOPT, CS/FD...)
               4) Run the optimization and handle failed/unfeasible runs
               5) Set the activities on each of the components and return the result
        '''

        self.start_time = time_lib.time()
        objval = 0.0

        # Step 1) Find the vars: 1 for each component input where dispatch is not fixed
        self.vs = {}  # Min/Max tuples of the various input
        self.storage_levels = {}
        for c in self.components:
            if c.dispatch_type == 'fixed':
                # Fixed dispatch components do not contribute to the variables
                continue
            else:  # Independent and dependent dispatch
                lower = c.min_capacity
                upper = c.capacity
                # Note: This assumes everything based off the first point
                if lower[0] < upper[0]:
                    self.vs[c.name] = [lower, upper]
                else:
                    self.vs[c.name] = [upper, lower]
            if c.stores:
                self.storage_levels[c.name] = np.zeros(len(self.time))

        full_dispatch = DispatchState(self.components, self.time)

        win_start_i = 0
        win_i = 0
        prev_win_end_i = 0
        prev_win_end = {} # a dict for tracking the final values of dispatchable components in a time window

        time_windows = []

        while win_start_i < len(self.time):
            win_end_i = win_start_i + self._window_length
            if win_end_i > len(self.time):
                win_end_i = len(self.time)

            # If the end time has not changed, then exit
            if win_end_i == prev_win_end_i:
                break

            # if self.verbose:
            print(f'win: {win_i}, start: {win_start_i}, end: {win_end_i}')
            time_windows.append([win_start_i, win_end_i])

            win_horizon = self.time[win_start_i:win_end_i]
            if self.verbose:
                print('Dispatching window', win_i)
            if win_i == 0:
                win_dispatch, win_obj_val = self._dispatch_window(
                    win_horizon, win_start_i, win_end_i)
            else:
                win_dispatch, win_obj_val = self._dispatch_window(
                    win_horizon, win_start_i, win_end_i, prev_win_end)
            if self.verbose:
                print(f'Optimal dispatch for win {win_i}:', win_dispatch)

            for comp in self.components:
                for res in comp.get_resources():
                    full_dispatch.set_activity_vector(
                        comp, res, win_start_i, win_end_i,
                        win_dispatch.get_activity(comp, res)
                    )
                if comp.dispatch_type != 'fixed':
                    prev_win_end[comp.name] = win_dispatch.get_activity(
                        comp, comp.capacity_resource, -1
                    )
                if comp.stores:
                    # Get the initial storage level
                    if win_start_i == 0:
                        init_level = comp.storage_init_level
                    else:
                        init_level = self.storage_levels[comp.name][win_start_i-1]

                    print(comp.name, 'storage init:', init_level)
                    # Update the storage_levels dict
                    activity = win_dispatch.get_activity(
                        comp, comp.capacity_resource)
                    activity = np.insert(activity, 0, init_level)
                    self.storage_levels[comp.name][win_start_i:win_end_i] = np.cumsum(activity)[1:]

            # Increment the window indexes
            prev_win_end_i = win_end_i
            win_i += 1
            objval += win_obj_val

            # This results in time windows that match up, but do not overlap
            win_start_i = win_end_i
            # print(full_dispatch)

        solution = Solution(self.time, full_dispatch.state, self.storage_levels,
                                False, objval, time_windows=time_windows)
        return solution

    def generate_objective(self) -> callable:
        '''Assembles an objective function to minimize the system cost'''
        if self.external_obj_func:
            return self.external_obj_func
        else:

            def objective(dispatch: DispatchState) -> float:
                '''The objective function. It is broken out to allow for easier scaling.
                :param dispatch: the full dispatch of the system
                :returns: float, value of the objective function
                '''
                obj = 0.0
                for c in self.components:
                    obj += c.cost_function(dispatch.state[c.name])
                return obj
            return objective

    def _dispatch_window(self, time_window: List[float],
                            start_i: int, end_i: int, prev_win_end: dict=None) -> Solution:
        '''Dispatch a time-window using a resource-pool method
        :param time_window: The time window to dispatch the system over
        :param start_i: The time-array index for the start of the window
        :param end_i: The time-array index for the end of the window
        :param prev_win_end: dict of the ending values for the previous time window used for consistency constraints
        :returns: DispatchState, the optimal dispatch over the time_window
        '''

        # Step 1) Build the resource pool constraint functions
        if self.verbose:
            print('Step 2) Build the resource pool constraints',
              time_lib.time() - self.start_time)
        pool_cons = self._build_pool_cons()

        # Step 2) Set up the objective function and constraint functions
        if self.verbose:
            print('Step 4) Assembling the big function',
              time_lib.time() - self.start_time)

        objective = self.generate_objective()

        obj_scale = 1.0
        if self.scale_objective:
            # Make an initial call to the objective function and scale it
            init_dispatch = {}
            for comp in self.components:
                if comp.dispatch_type != 'fixed':
                    init_dispatch[comp.name] = comp.guess[start_i:end_i+1]

            # get the initial dispatch so it can be used for scaling
            initdp = self.determine_dispatch(init_dispatch, time_window, start_i, end_i)
            obj_scale = objective(initdp)

        # Figure out the initial storage levels
        # if this is the first time window, use the 'storage_init_level' property.
        # Otherwise use the end storage level from the previous time window.
        storage_levels = {}
        for comp in self.components:
            if comp.stores:
                if start_i == 0:
                    storage_levels[comp.name] = comp.storage_init_level
                else:
                    storage_levels[comp.name] = self.storage_levels[comp.name][start_i-1]

        def optimize_me(stuff: dict) -> [dict, bool]:
            '''Objective function passed to pyOptSparse
            It returns a dict describing the values of the objective and constraint
            functions along with a bool indicating whether an error occured.
            :param stuff: dict of optimization vars from pyOptSparse
            :returns: [dict, bool]
            '''
            try:
                dispatch = self.determine_dispatch(stuff, time_window, start_i, end_i)
                # At this point the dispatch should be fully determined, so assemble the return object
                things = {}
                # Dispatch the components to generate the obj val
                things['objective'] = objective(dispatch)/obj_scale
                # Run the resource pool constraints
                things['resource_balance'] = [cons(dispatch) for cons in pool_cons]
                for comp in self.components:
                    if comp.dispatch_type != 'fixed':
                        if start_i == 0:
                            things[f'ramp_{comp.name}'] = np.diff(stuff[comp.name])
                        else: # Make sure subsequent windows start from the last point of the previous window
                            things[f'ramp_{comp.name}'] = np.diff(
                                np.insert(stuff[comp.name], 0, prev_win_end[comp.name]))
                    if comp.stores is not None:
                        # Add the initial storage level
                        storage = np.insert(dispatch.state[comp.name][comp.stores],
                                                0, storage_levels[comp.name])
                        things[f'{comp.name}_storage_level'] = np.cumsum(storage)[1:]
                return things, False
            except Exception: # If the input crashes the objective function
                return {}, True
        self.objective = optimize_me

        # Step 3) Formulate the problem for pyOptSparse
        if self.verbose:
            print('Step 5) Setting up pyOptSparse',
              time_lib.time() - self.start_time)
        optProb = pyoptsparse.Optimization('Dispatch', optimize_me)
        for comp in self.components:
            if comp.dispatch_type != 'fixed':
                bounds = [bnd[start_i:end_i] for bnd in self.vs[comp.name]]
                # FIXME: will need to find a way of generating the guess values
                guess = comp.guess[start_i:end_i]
                ramp_up = comp.ramp_rate_up[start_i:end_i]
                ramp_down = comp.ramp_rate_down[start_i:end_i]
                if start_i == 0: # The first window will have n-1 ramp points
                    optProb.addConGroup(f'ramp_{comp.name}', len(time_window)-1,
                                    lower=-1*ramp_down[:-1], upper=ramp_up[:-1])
                else:
                    optProb.addConGroup(f'ramp_{comp.name}', len(time_window),
                                        lower=-1*ramp_down, upper=ramp_up)

                if comp.stores is not None:
                    min_capacity = comp.min_capacity[start_i:end_i]
                    max_capacity = comp.capacity[start_i:end_i]
                    ramp_up = comp.ramp_rate_up[start_i:end_i]
                    ramp_down = -1*comp.ramp_rate_down[start_i:end_i]
                    print(f'{comp.name}', comp.stores, min_capacity, max_capacity)
                    print(ramp_down, ramp_up)
                    optProb.addConGroup(f'{comp.name}_storage_level', len(time_window),
                        lower=min_capacity, upper=max_capacity)
                    # Storage components can have negative activities
                    optProb.addVarGroup(comp.name, len(time_window), 'c',
                                        value=np.zeros(len(ramp_down)),
                                        lower=ramp_down,
                                        upper=ramp_up)

                else:
                    optProb.addVarGroup(comp.name, len(time_window), 'c',
                                        value=guess, lower=bounds[0], upper=bounds[1])

        optProb.addConGroup('resource_balance', len(
            pool_cons), lower=0, upper=0)
        optProb.addObj('objective')

        # Step 4) Run the optimization
        if self.verbose:
            print('Step 6) Running the dispatch optimization',
              time_lib.time() - self.start_time)
        try:
            opt = pyoptsparse.OPT('ipopt')
            sol = opt(optProb, sens='CD')
            # print(sol)
            if self.verbose:
                print('Dispatch optimization successful')
        except Exception as err:
            print('Dispatch optimization failed:')
            traceback.print_exc()
            raise err
        if self.verbose:
            print('Step 6.5) Completed the dispatch optimization',
                  time_lib.time() - self.start_time)

        # Step 5) Set the activities on each component
        if self.verbose:
            print('\nCompleted dispatch process\n\n\n')

        win_opt_dispatch = self.determine_dispatch(sol.xStar, time_window, start_i, end_i)
        if self.verbose:
            print('\nReturning the results', time_lib.time() - self.start_time)
        return win_opt_dispatch, sol.fStar

    def gen_slack_storage_trans(self, res) -> callable:
        def trans(data, meta):
            return data, meta
        return trans

    def gen_slack_storage_cost(self, res) -> callable:
        def cost(dispatch):
            return np.sum(1e10*dispatch[res])
        return cost

    def add_slack_storage(self) -> None:
        for res in self.resources:
            num = 1e10*np.ones(len(self.time))
            guess = np.zeros(len(self.time))

            trans = self.gen_slack_storage_trans(res)
            cost = self.gen_slack_storage_cost(res)

            c = PyOptSparseComponent(f'{res}_slack', num, num, num, res, trans,
                                        cost, stores=[res], guess=guess)
            self.components.append(c)
            self.slack_storage_added = True

    def dispatch(self, components: List[PyOptSparseComponent],
                    time: List[float], timeSeries: List[TimeSeries] = [],
                    external_obj_func: callable=None, meta=None,
                    verbose: bool=False, scale_objective: bool=True,
                    slack_storage: bool=False) -> Solution:
        """Optimally dispatch a given set of components over a time horizon
        using a list of TimeSeries

        :param components: List of components to dispatch
        :param time: time horizon to dispatch the components over
        :param timeSeries: list of TimeSeries objects needed for the dispatch
        :param external_obj_func: callable, An external objective function
        :param meta: stuff, an arbitrary object passed to the transfer functions
        :param verbose: Whether to print verbose dispatch
        :param scale_objective: Whether to scale the objective function by its initial value
        :param slack_storage: Whether to use artificial storage components as "slack" variables
        :returns: optDispatch, A dispatch-state object representing the optimal system dispatch
        :returns: storage_levels, the storage levels of the system components
        Note that use of `external_obj_func` will replace the use of all component cost functions
        """
        # FIXME: Should check to make sure that the components have arrays of the right length
        self.components = components
        self.time = time
        self.verbose = verbose
        self.timeSeries = timeSeries
        self.scale_objective = scale_objective
        self.external_obj_func = external_obj_func # Should be callable or None
        self.meta = meta

        resources = [c.get_resources() for c in self.components]
        self.resources = list(set(chain.from_iterable(resources)))

        if slack_storage and not self.slack_storage_added:
            self.add_slack_storage()

        return self._dispatch_pool()

# ToDo:
# - Get it to stop printing the annoying "Using option file" comment
# - Try priming the initial values for generic systems better
# - Calculate exact derivatives using JAX if possible
#   - Could use extra meta props to accomplish this
#   - Could also explicitly disable use of meta in user functions
# - Integrate storage into the dispatch
# - Handle infeasible cases clearly

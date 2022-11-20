from .Dispatcher import Dispatcher, DispatchState
from .Component import PyOptSparseComponent
from .TimeSeries import TimeSeries
from .Solution import Solution

import numpy as np
import sys
import os
import time as time_lib
import traceback
from typing import List
from itertools import chain
from pprint import pformat
from scipy.optimize import minimize, NonlinearConstraint

# Some formulation parameters
MAX_RESOURCE_BALANCE_CONSTRAINT_VIOLATION = 10000
MAX_RAMP_CONSTRAINT_VIOLATION = 10000
MAX_STORAGE_LIMIT_CONSTRAINT_VIOLATION = 10000

class ScipyDispatcher(Dispatcher):
    '''
    Dispatch using Scipty optimization algorithms and a pool-based method.
    '''

    slack_storage_added = False # In case slack storage is added in a loop

    def __init__(self, window_length=10):
        self.name = 'ScipyDispatcher'
        self._window_length = window_length

        # Defined on call to self.dispatch
        self.components = None
        self.case = None

    def _dispatch_pool(self) -> Solution:
        '''Dispatch the given system using a resource-pool method

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

            if self.verbose:
                print(f'win: {win_i}, start: {win_start_i}, end: {win_end_i}')
            time_windows.append([win_start_i, win_end_i])

            win_horizon = self.time[win_start_i:win_end_i]
            if self.verbose:
                print('Dispatching window', win_i)

            # Assemble the "initial storage levels" for the window
            init_store = {}
            storers = [comp for comp in self.components if comp.stores]
            for storer in storers:
                if win_start_i == 0:
                    init_store[storer.name] = storer.storage_init_level
                else:
                    init_store[storer.name] = self.storage_levels[storer.name][win_start_i-1]


            pass_in_prev_win_end = prev_win_end
            if win_i == 0:
                pass_in_prev_win_end = None
            #     win_dispatch, store_lvls, win_obj_val = self._dispatch_window(
            #         win_horizon, win_start_i, win_end_i, init_store)
            # else:
            #     win_dispatch, store_lvls, win_obj_val = self._dispatch_window(
            #         win_horizon, win_start_i, win_end_i, init_store, prev_win_end)

            dispatch_window = DispatchWindow(self.components, self.resources, win_horizon,
                                    win_start_i, win_end_i, init_store,
                                    pass_in_prev_win_end, self.external_obj_func)
            win_dispatch, store_lvls, win_obj_val = dispatch_window.optimize_dispatch(self.scale_objective)

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

                # Update the storage_levels dict
                if comp.stores:
                    self.storage_levels[comp.name][win_start_i:win_end_i] = store_lvls[comp.name]

            # Increment the window indexes
            prev_win_end_i = win_end_i
            win_i += 1
            objval += win_obj_val

            # This results in time windows that match up, but do not overlap
            win_start_i = win_end_i

        # FIXME: Return the total error
        solution = Solution(self.time, full_dispatch.state, self.storage_levels,
                                False, objval, time_windows=time_windows)
        return solution

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


class DispatchWindow(object):
    '''Class for dispatching a given window. Broken out from ScipyDispatcher to ease data access.'''
    def __init__(self, components, resources, time_window, start_i, end_i, init_store, prev_win_end, external_obj_func):
        self.components = components
        self.resources = resources
        self.time_window = time_window,
        self.start_i = start_i
        self.end_i = end_i
        self.init_store = init_store
        self.prev_win_end = prev_win_end
        self.external_obj_func = external_obj_func
        self.window_length = end_i - start_i

    def _gen_pool_cons(self, resource) -> callable:
        '''A closure for generating a pool constraint for a resource
        :param resource: the resource to evaluate
        :returns: a function representing the pool constraint
        '''

        def pool_cons(dispatch: DispatchState) -> float:
            '''A resource pool constraint
            Checks that the net amount of a resource being consumed, produced and
            stored is zero.
            :param dispatch: the dispatch to evaluate
            :returns: SSE of resource constraint violations
            '''
            n = len(self.time_window)
            err = np.zeros(n)

            # FIXME: This is an inefficient way of doing this. Find a better way
            cs = [c for c in self.components if resource in c.get_resources()]
            for i, _ in enumerate(self.time_window):
                for c in cs:
                    if c.stores:
                        err[i] += -dispatch.get_activity(c, resource, i)
                    else:
                        err[i] += dispatch.get_activity(c, resource, i)

            # FIXME: This simply returns the sum of the errors over time. There
            # are likely much better ways of handling this.
            # Maybe have a constraint for the resource at each point in time?
            # Maybe use the storage components as slack variables?
            return sum(err**2)
        return pool_cons

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

    def _convert_optvars_to_dispatch(self, opt_vars):
        stuff = {}
        disp_comps = [
            comp for comp in self.components if comp.dispatch_type != 'fixed']
        for i, comp in enumerate(disp_comps):
            if comp.dispatch_type != 'fixed':
                stuff[comp.name] = opt_vars[i:i+self.window_length]
        dispatch, store_lvl = self.determine_dispatch(stuff)
        return dispatch, store_lvl

    def generate_trust_constr_constraints(self) -> List[NonlinearConstraint]:
        '''Generate a list of NonlinearConstraint objects describing system
        constraints for use with Scipy's trust-constr method.'''

        # Functions are included as closures as access to self is needed,
        # but the signature must be as shown
        constraints = []

        # Add the resource balance constraints for each resource pool
        def resource_balance_constraint(opt_vars):
            imbalance = 0.0
            dispatch, _ = self._convert_optvars_to_dispatch(opt_vars)
            # These are all bunched together to reduce computational demand in
            # approximating the derivative
            for res in self.resources:
                imbalance += self._gen_pool_cons(res)(dispatch)

            return imbalance
        constraints.append(NonlinearConstraint(resource_balance_constraint,
                                               0, MAX_RESOURCE_BALANCE_CONSTRAINT_VIOLATION))

        # Add the ramp constraints
        def ramp_rate_constraint(opt_vars):
            dispatch, _ = self._convert_optvars_to_dispatch(opt_vars)
            error = 0.0
            dispatch_components = [
                c for c in self.components if c.dispatch_type != 'fixed']

            for d in dispatch_components:
                capres_dispatch = dispatch.get_activity(d, d.capacity_resource)
                if self.prev_win_end is None:
                    ramp = np.diff(capres_dispatch)
                else:
                    ramp = np.diff(
                        np.insert(capres_dispatch, 0, self.prev_win_end[d.name]))
                max_ramp_upr = d.ramp_rate_up[self.start_i:self.end_i]
                max_ramp_lwr = d.ramp_rate_down[self.start_i:self.end_i]
                for r, upr, lwr in zip(ramp, max_ramp_upr, max_ramp_lwr):
                    if r > upr:
                        error += r - upr
                    elif r < lwr:
                        error += lwr - r
            return error
        constraints.append(NonlinearConstraint(ramp_rate_constraint,
                                               0, MAX_RAMP_CONSTRAINT_VIOLATION))

        # Add the storage level constraints
        def storage_limit_constraint(opt_vars):
            _, store_lvl = self._convert_optvars_to_dispatch(opt_vars)
            error = 0.0
            for c in self.components:
                if c.stores:
                    c_store_lvl = store_lvl[c.name]
                    max_store_lvl = c.capacity[self.start_i:self.end_i]
                    for lvl, mx in zip(c_store_lvl, max_store_lvl):
                        error += max(0, lvl - mx)
            return error
        constraints.append(NonlinearConstraint(storage_limit_constraint,
                                               0, MAX_STORAGE_LIMIT_CONSTRAINT_VIOLATION))
        return constraints


    def determine_dispatch(self, opt_vars: dict) -> DispatchState:
        '''Determine the dispatch from a given set of optimization
        vars by running the transfer functions. Returns a Numpy dispatch
        object
        :param opt_vars: dict, holder for all the optimization variables
        :returns: DispatchState, dispatch of the system
        :returns: dict, storage levels of each storage component over time
        '''
        # Initialize the dispatch
        dispatch = DispatchState(self.components, self.time_window)
        store_lvls = {}
        # Dispatch the fixed components
        fixed_comps = [
            c for c in self.components if c.dispatch_type == 'fixed']
        for f in fixed_comps:
            dispatch.set_activity(f, f.capacity_resource,
                                  f.capacity[self.start_i:self.end_i])
        # Dispatch the independent and dependent components using the vars
        disp_comps = [
            c for c in self.components if c.dispatch_type != 'fixed']
        for d in disp_comps:
            dispatch.set_activity(d, d.capacity_resource, opt_vars[d.name])
            if d.stores:
                store_lvls[d.name] = d.transfer(
                    opt_vars[d.name], self.init_store[d.name])
            else:
                bal = d.transfer(opt_vars[d.name])
                for res, values in bal.items():
                    dispatch.set_activity(d, res, values)
        return dispatch, store_lvls


    def optimize_dispatch(self, scale_objective) -> Solution:
        '''Dispatch a time-window using a resource-pool method
        :param time_window: The time window to dispatch the system over
        :param start_i: The time-array index for the start of the window
        :param end_i: The time-array index for the end of the window
        :param init_store: dict of the initial storage values of the storage components
        :param prev_win_end: dict of the ending values for the previous time window used for consistency constraints
        :returns: DispatchState, the optimal dispatch over the time_window
        '''

        # Step 2) Setz up the objective function and constraint functions
        objective = self.generate_objective()

        obj_scale = 1.0
        if scale_objective:
            # Make an initial call to the objective function and scale it
            init_dispatch = {}
            for comp in self.components:
                if comp.dispatch_type != 'fixed':
                    init_dispatch[comp.name] = comp.guess[self.start_i:self.end_i]

            # get the initial dispatch so it can be used for scaling
            initdp, _ = self.determine_dispatch(init_dispatch)
            obj_scale = objective(initdp)

        # Figure out the initial storage levels
        # if this is the first time window, use the 'storage_init_level' property.
        # Otherwise use the end storage level from the previous time window.
        storage_levels = {}
        for comp in self.components:
            if comp.stores:
                if self.start_i == 0:
                    storage_levels[comp.name] = comp.storage_init_level
                else:
                    storage_levels[comp.name] = self.init_store[comp.name]

        def optimize_me(opt_vars: List[float]) -> float:
            '''Objective function passed to SciPy'''
            dispatch, _ = self._convert_optvars_to_dispatch(opt_vars)
            return objective(dispatch)/obj_scale
            # try:
            # except:  # In case an input crashes the objective function
            #     print('Infeasible objective function')
            #     return 1e8
        constraints = self.generate_trust_constr_constraints()

        # Step 3) Formulate the problem for SciPy
        guess = []
        bounds = []
        dispatch_components = [
            c for c in self.components if c.dispatch_type != 'fixed']
        for d in dispatch_components:
            guess.extend(d.guess[self.start_i:self.end_i])
            bounds.extend(
                zip(d.min_capacity[self.start_i:self.end_i], d.capacity[self.start_i:self.end_i]))

        sol = minimize(optimize_me, guess,
                       constraints=constraints,
                       bounds=bounds,
                       method='trust-constr',
                       options={'disp': True})

        # Shape the optimization arguments back into a dict
        opt_results = {}

        for i, d in enumerate(dispatch_components):
            opt_results[d.name] = sol.x[i*self.window_length:(i+1)*self.window_length]

        win_opt_dispatch, store_lvl = self.determine_dispatch(opt_results)
        return win_opt_dispatch, store_lvl, sol.fun

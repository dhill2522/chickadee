from .Dispatcher import Dispatcher

import pyoptsparse
import numpy as np
import sys
import os
import time as time_lib
import traceback
from itertools import chain
from pprint import pformat

class DispatchState(object):
    '''Modeled after idaholab/HERON NumpyState object'''
    def __init__(self, components, time):
        s = {
            'time': time
        }

        for c in components:
            s[c.name] = {}
            for resource in c.get_resources():
                s[c.name][resource] = np.zeros(len(time))

        self.state = s
        self.time = time

    def set_activity(self, component, resource, activity, i=None):
        if i is None:
            self.state[component.name][resource] = activity
        else:
            self.state[component.name][resource][i] = activity

    def get_activity(self, component, resource, i=None):
        try:
            if i is None:
                return self.state[component.name][resource]
            else:
                return self.state[component.name][resource][i]
        except Exception as err:
            print(i)
            raise err

    def set_activity_vector(self, component, resource, start, end, activity):
        self.state[component.name][resource][start:end] = activity

    def __repr__(self):
        return pformat(self.state)




class PyOptSparse(Dispatcher):
    '''
    Dispatch using pyOptSparse optimization package
    '''

    def __init__(self, window_length=10):
        print('PyOptDispatch:__init__')
        self.name = 'PyOptSparseDispatcher'
        self._window_length = window_length

        # Defined on call to self.dispatch
        self.components = None
        self.case = None

    def __gen_pool_cons(self, res):
        '''A closure for generating a pool constraint for a resource'''

        def pool_cons(dispatch_window: DispatchState):
            '''A resource pool constraint

            Ensures that the net amount of a resource being consumed, produced and
            stored is zero. Inteded to '''
            time = dispatch_window.time
            n = len(time)
            err = np.zeros(n)

            # FIXME: This is an inefficient way of doing this. Find a better way
            cs = [c for c in self.components if res in c.get_resources()]
            for i, t in enumerate(time):
                for c in cs:
                    err[i] += dispatch_window.get_activity(c, res, i)

            # FIXME: This simply returns the sum of the errors over time. There
            # are likely much better ways of handling this.
            # Maybe have a constraint for the resource at each point in time?
            # Maybe use the storage components as slack variables?
            return sum(err)

        return pool_cons

    def _build_pool_cons(self):
        '''Build the pool constraints
        Returns a list of `pool_cons` functions, one for each resource.'''

        cons = []
        for res in self.resources:
            # Generate the pool constraint here
            pool_cons = self.__gen_pool_cons(res)
            cons.append(pool_cons)
        return cons

    def determine_dispatch(self, opt_vars, time):
        '''Determine the dispatch from a given set of optimization
        vars by running the transfer functions. Returns a Numpy dispatch
        object
        @ In, opt_vars, dict, holder for all the optimization variables
        @ In, time, list, time horizon to dispatch over
        '''
        # Initialize the dispatch
        dispatch = DispatchState(self.components, time)
        # Dispatch the fixed components
        fixed_comps = [c for c in self.components if c.dispatch_type == 'fixed']
        for f in fixed_comps:
            capacity = f.capacity
            vals = np.ones(len(time)) * capacity
            dispatch.set_activity(f, f.capacity_resource, vals)
        # Dispatch the independent and dependent components using the vars
        disp_comps = [
            c for c in self.components if c.dispatch_type != 'fixed']
        for d in disp_comps:
            for i in range(len(time)):
                request = {d.capacity_resource: opt_vars[d.name][i]}
                bal, meta = d.transfer(request, {})
                for res, value in bal.items():
                    dispatch.set_activity(d, res, value, i)

        return dispatch, meta

    def _dispatch_pool(self):
        # Steps:
        #   1) Assemble all the vars into a vars dict
        #     A set of vars for each dispatchable component including storage elements
        #     include bound constraints
        #   2) Build the pool constraint functions
        #     Should have one constraint function for each pool
        #     Each constraint will be a function of all the vars
        #   3) Assemble the transfer function constraints
        #   4) Set up the objective function as the double integral of the incremental dispatch
        #   5) Assemble the parts for the optimizer function
        #     a) Declare the variables
        #     b) Declare the constraints
        #     c) Declare the objective function
        #     d) set the optimization configuration (IPOPT/SNOPT, CS/FD...)
        #   6) Run the optimization and handle failed/unfeasible runs
        #   7) Set the activities on each of the components and return the result

        print('\n\n\n\nDEBUG: pyOptSparse dispatcher')
        resources = [c.get_resources() for c in self.components]
        self.resources = list(set(chain.from_iterable(resources)))
        self.start_time = time_lib.time()

        # Step 1) Find the vars: 1 for each component input where dispatch is not fixed
        self.vs = {}  # Min/Max tuples of the various input
        for c in self.components:
            if c.dispatch_type == 'fixed':
                # Fixed dispatch components do not contribute to the variables
                continue
            else:  # Independent and dependent dispatch
                self.vs[c.name] = {}
                # FIXME: get real min capacity
                self.vs[c.name] = [0, c.capacity]
                self.vs[c.name].sort()  # The max capacities are often negative

        full_dispatch = DispatchState(self.components, self.time)

        # FIXME: Add constraints to ensure that the windows overlap

        win_start_i = 0
        win_i = 0
        while win_start_i < len(self.time):
            win_end_i = win_start_i + self._window_length
            if win_end_i > len(self.time):
                win_end_i = len(self.time)
            print(f'win: {win_i}, start: {win_start_i}, end: {win_end_i}')

            win_horizon = self.time[win_start_i:win_end_i]
            win_dispatch = self._dispatch_window(win_horizon, win_i)

            for comp in self.components:
                for res in comp.get_resources():
                    full_dispatch.set_activity_vector(
                        comp, res, win_start_i, win_end_i,
                        win_dispatch.get_activity(comp, res)
                    )

            win_i += 1
            win_start_i = win_end_i

            if win_i > 10:
                break

        return full_dispatch

        # return self._dispatch_window(time_horizon, 1)

    def _dispatch_window(self, time_window, win_i, verbose=False):
        if verbose:
            print('Dispatching window', win_i)

        # Step 2) Build the resource pool constraint functions
        if verbose:
            print('Step 2) Build the resource pool constraints',
              time_lib.time() - self.start_time)
        pool_cons = self._build_pool_cons()

        # Step 3) Assemble the transfer function constraints
        # this is taken into account by "determining the dispatch"

        # Step 4) Set up the objective function as the double integral of the incremental dispatch
        if verbose:
            print('Step 4) Assembling the big function',
              time_lib.time() - self.start_time)

        def objective(stuff):
            return 10

        def optimize_me(stuff):
            # nonlocal meta
            dispatch, self.meta = self.determine_dispatch(stuff, time_window)
            # At this point the dispatch should be fully determined, so assemble the return object
            things = {}
            # Dispatch the components to generate the obj val
            things['objective'] = objective(stuff)
            # Run the resource pool constraints
            things['resource_balance'] = [cons(dispatch) for cons in pool_cons]
            things['window_overlap'] = []
            # FIXME: Nothing is here to verify ramp rates!
            return things, False

        # Step 5) Assemble the parts for the optimizer function
        if verbose:
            print('Step 5) Setting up pyOptSparse',
              time_lib.time() - self.start_time)
        optProb = pyoptsparse.Optimization('Dispatch', optimize_me)
        for comp, bounds in self.vs.items():
            # FIXME: will need to find a way of generating the guess values
            optProb.addVarGroup(comp, len(time_window), 'c',
                                value=-1, lower=bounds[0], upper=bounds[1])
        optProb.addConGroup('resource_balance', len(
            pool_cons), lower=0, upper=0)
        # if win_i != 0:
        #   optProb.addConGroup('window_overlap', len())
        optProb.addObj('objective')

        # Step 6) Run the optimization
        if verbose:
            print('Step 6) Running the dispatch optimization',
              time_lib.time() - self.start_time)
        try:
            opt = pyoptsparse.OPT('IPOPT')
            sol = opt(optProb, sens='CD')
            # print(sol)
            if verbose:
                print('Dispatch optimization successful')
        except Exception as err:
            print('Dispatch optimization failed:')
            traceback.print_exc()
            raise err

        # Step 7) Set the activities on each component
        if verbose:
            print('\nCompleted dispatch process\n\n\n')

        win_opt_dispatch, self.meta = self.determine_dispatch(
            sol.xStar, time_window)
        if verbose:
            print(f'Optimal dispatch for win {win_i}:', win_opt_dispatch)
            print('\nReturning the results', time_lib.time() - self.start_time)
        return win_opt_dispatch

    def dispatch(self, components, time, tdep):
        self.components = components
        self.time = time
        self.tdep = tdep
        return self._dispatch_pool()

# Questions:
# - How should I raise exceptions? What is the raven way?
# - Is there a suggested way of multithreading in HERON?
# - Should I set things like meta as class members or pass them everywhere (functional vs oop)?
# - Tried internalParrallel to true for inner and it failed to import TEAL. Any ideas?
# - Best way of getting dispatch window length from user?

# ToDo:
# - Try priming the initial values better
# - Calculate exact derivatives using JAX
#   - Could use extra meta props to accomplish this
# - Scale the obj func inputs and outputs
# - Find a way to recover the optimal dispatch
# - Integrate storage into the dispatch
#   - formulate storage as a slack variable in the resource constraint
# - Determine the analytical solution for a benchmark problem


# Ideas
# - may need to time the obj func call itself
# - Could try linearizing the transfer function?
#   - Probably not a good idea as it would severely limit the functionality of the transfer
#   - Could be done by a researcher beforehand and used in the Pyomo dispatcher

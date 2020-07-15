"""A general function for computing a model's steady state variable and parameters values"""

import numpy as np
import scipy.optimize as opt
from copy import copy

import sequence_jacobian as sj


# Find the steady state solution
# Feature to-be-implemented: Add more flexible specification of numerical solver/required kwargs to be used for
# the numerical solution. For now, just add optional "bounds" arguments to the steady_state.
def steady_state(model_dag, dag_targets, idiosyncratic_grids, prespecified_variables_and_parameters,
                 calibration_set, analytic_solution=None, numerical_solution=None,
                 numerical_solution_initialization=None, walras_variable="C",
                 validity_check=True):

    if analytic_solution is None and numerical_solution is None:
        raise RuntimeError("Must provide either an analytic solution or a numerical solution method to solve"
                           " for non-pre-specified variables/parameters.")

    # Validity Checks:
    # The following code is to check the validity of the steady state calibration exercise
    # by counting all of the variables/parameters involved as inputs/outputs in the model dag
    # and ensuring the user has specified a means of solving for them.
    # Can be performed optionally (allow user to bypass if they don't want to check)
    # So probably best calculated the first time the steady state is solved for but not every time
    # *Feature to-be-implemented: For now omitting the edge case of models that do not require *any* numerical solution
    #   e.g. the RBC model.
    if validity_check:
        # Find the set of model variables and parameters that are candidates for calibration instruments and targets
        vars_and_paras = set([])
        for block in model_dag.blocks:
            vars_and_paras = vars_and_paras.union(block.inputs)
            vars_and_paras = vars_and_paras.union(block.outputs)

        # Remove all "*_grid" variables, "Pi" (assumed to be the standard name for the state transition matrix),
        # and any variables contained in 'dag_targets' from vars_and_paras
        for vp in copy(vars_and_paras):
            if "_grid" in vp or vp == "Pi" or vp in model_dag.targets or vp == walras_variable:
                vars_and_paras.remove(vp)

        # This is the total set of variables and parameters that must be solved for either
        # analytically or numerically
        vars_and_paras_to_be_solved = vars_and_paras.difference(prespecified_variables_and_parameters.keys())
        vars_and_paras_to_be_solved = vars_and_paras_to_be_solved.difference(calibration_set.get_instrument_names())
        vars_and_paras_to_be_solved = vars_and_paras_to_be_solved.difference(calibration_set.get_target_names())

        # If certain variables/parameters can be solved analytically and an analytic solution has been provided
        # remove them from the accounting of the set of variables/parameters that need to be solved.
        if analytic_solution is not None:
            vars_and_paras_to_be_solved = vars_and_paras_to_be_solved.difference(
                sj.utils.output_list(analytic_solution))

        # The variables/parameters found via numerical methods (in the set vars_and_paras_to_be_solved_num)
        # must be the same as the set of outputs from the function "numerical_solution"
        # further, the "calibration instrument"s must be arguments to the "numerical solution" function
        assert vars_and_paras_to_be_solved == set(sj.utils.output_list(numerical_solution))
        assert set(calibration_set.get_instrument_names()).issubset(set(sj.utils.input_list(numerical_solution)))

    # Method Body
    # Build up the dictionary of potential arguments available to use for computing the solution to the analytic
    # and numerical components of the steady state computation
    calibration_targets = dict(zip(calibration_set.get_target_names(), calibration_set.get_target_values()))
    potential_args = {**calibration_targets, **idiosyncratic_grids, **prespecified_variables_and_parameters}

    if analytic_solution is not None:
        analytic_input_arg_names = sj.utils.input_list(analytic_solution)
        analytic_output_arg_names = sj.utils.output_list(analytic_solution)
        analytic_input_args = [potential_args[arg_name] for arg_name in analytic_input_arg_names]
        analytic_output_args = analytic_solution(*analytic_input_args)

        potential_args.update(zip(analytic_output_arg_names, analytic_output_args))

    # *Feature to-be-implemented: Expect that numerical_solution_initialization is a function. Later allow
    #   for the possibility of providing an Array directly.
    # *Feature to-be-implemented: For now expect the numerical solution initialization to be a single value
    #   (so not compatible with two asset yet)
    # Construct the initial value for the numerical solution programmatically.
    # Expect that the arguments for the initialization are either grids, pre-specified variables/parameters
    # or are solved for in the analytic solution, so look through the dictionary of both sets of
    # values to pass into the numerical_solution_initialization function.
    num_init_input_arg_names = sj.utils.input_list(numerical_solution_initialization)
    num_init_output_arg_names = sj.utils.output_list(numerical_solution_initialization)
    num_init_input_args = [potential_args[arg_name] for arg_name in num_init_input_arg_names]
    num_init_output_args = numerical_solution_initialization(*num_init_input_args)

    potential_args.update(zip(num_init_output_arg_names, [num_init_output_args]))

    # Manually construct the ordered list of arguments to enter into the numerical solution
    # *Note on expected argument ordering: The expected argument ordering for the numerical_solution()
    #   method is: calibrated instruments first, then calibrated targets, then the initial value
    #   for the iteration procedure (e.g. V' for initializing value function iteration), and then
    #   other relevant grids, pre-specified variables/parameters, and solved variables/parameters
    #   (the latter 3 categories in arbitrary order)
    num_input_arg_names = list_difference(sj.utils.input_list(numerical_solution),
                                          calibration_set.get_instrument_names())
    num_output_arg_names = sj.utils.output_list(numerical_solution)
    num_input_args = [potential_args[arg_name] for arg_name in num_input_arg_names]

    # Build the numerical residual function programmatically, using the dag targets and the
    # numerical solution input arguments
    # *Feature to-be-implemented: Expect num_output_args to be a scalar. Generalize functionality to scalar and vectors
    def res(x):
        residuals = np.ones(len(dag_targets))
        target_to_block = find_target_blocks(model_dag, dag_targets)
        num_output_args = numerical_solution(x, *num_input_args)
        potential_args.update(zip(num_output_arg_names, [num_output_args]))

        for i, target in enumerate(dag_targets):
            block_input_arg_names = target_to_block[target].inputs
            block_input_args = [potential_args[arg_name] for arg_name in block_input_arg_names]
            residuals[i] = target_to_block[target].ss(*block_input_args)

        return residuals

    # *Feature to-be-implemented: Allow for generic numerical solvers and output handling.
    #   For now expect the numerical solution to be univariate (as in krusell smith)
    cal_instr, sol = opt.brentq(res, calibration_set.get_instrument_bounds()[0][0],
                                calibration_set.get_instrument_bounds()[0][1],
                                full_output=True)

    potential_args.update(zip(calibration_set.get_instrument_names(), [cal_instr]))

    if not sol.converged:
        raise ValueError("Steady-state solver did not converge.")

    # *Feature to-be-implemented: Handle Walras' Law as an additional simple block in the Model DAG and include
    # the walras variable (in this case, 'C') as one of the return arguments

    # Stop early for now to check functionality
    return potential_args

    # Check that the steady state solution is compatible with the dag representation
    # by computing the dag once through with the steady state unknown values, ignoring
    # all time displacement terms, and check that the targets are indeed hit.


# Find which blocks are associated with the targets.
# Assume that every target in the model's dag has an associated block,
# where the block.ss() function computes the relevant residual for the target.
# Note that this is NOT currently the case (for instance, the notebooks tend to group all of the market clearing
# conditions into a single block, whereas these methods expect they all be in separate blocks) so will need to be
# remedied before it can be used
def find_target_blocks(model_dag, dag_targets):
    target_to_block = {}
    for block in model_dag.blocks:
        # If there is more than one output from a model's block, then that block is assumed not to be associated
        # with a target, since the block's output should be the target residual.
        if len(block.outputs) != 1:
            continue
        else:
            if list(block.outputs)[0] in dag_targets:
                target_to_block[list(block.outputs)[0]] = block

    return target_to_block


# Need an additional utility to compute a set-like difference for lists, but preserving the order of
# the list that items are being deleted from
def list_difference(primary_list, *args):
    return [item for item in primary_list if item not in set().union(*args)]
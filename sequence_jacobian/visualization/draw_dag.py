"""Provides the functionality for basic DAG visualization"""

import warnings

# Force warnings.warn() to omit the source code line in the message
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
    formatwarning_orig(message, category, filename, lineno, line='')

from ..utilities.graph import block_sort, construct_output_map, construct_dependency_graph,\
    find_outputs_that_are_intermediate_inputs


# Implement DAG drawing functions as "soft" dependencies to not enforce the installation of graphviz, since
# it's not required for the rest of the sequence-jacobian code to run
try:
    """
    DAG Graph routine
    Requires installing graphviz package and executables
    https://www.graphviz.org/

    On a mac this can be done as follows:
    1) Download macports at:
    https://www.macports.org/install.php
    2) On the command line, install graphviz with macports by typing
    sudo port install graphviz

    """
    from graphviz import Digraph


    def draw_dag(block_list, exogenous=None, unknowns=None, targets=None, ignore_helpers=True, calibration=None,
                 showdag=False, debug=False, leftright=False, filename='modeldag'):
        """
        Visualizes a Directed Acyclic Graph (DAG) of a set of blocks, exogenous variables, unknowns, and targets

        block_list: `list`
            Blocks to be represented as nodes within a DAG
        exogenous: `list` (optional)
            Exogenous variables, to be represented on DAG
        unknowns: `list` (optional)
            Unknown variables, to be represented on DAG
        targets: `list` (optional)
            Target variables, to be represented on DAG
        ignore_helpers: `bool`
            A boolean indicating whether to also draw HelperBlocks contained in block_list into the DAG
        calibration: `dict` or `None`
            An optional dict of variable/parameter names and their pre-specified values to help resolve any cycles
            introduced by using HelperBlocks. Read `block_sort` docstring for more detail
        showdag: `bool`
            If True, export and plot pdf file. If false, export png file and do not plot
        debug: `bool`
            If True, returns list of candidate unknown and targets
        leftright: `bool`
            If True, plots DAG from left to right instead of top to bottom

        return: None
        """

        # To prevent having mutable variables as keyword arguments
        exogenous = [] if exogenous is None else exogenous
        unknowns = [] if unknowns is None else unknowns
        targets = [] if targets is None else targets

        # obtain the topological sort
        topsorted = block_sort(block_list, ignore_helpers=ignore_helpers, calibration=calibration)
        # get sorted list of blocks
        block_list_sorted = [block_list[i] for i in topsorted]
        # Obtain the dependency list of the sorted set of blocks
        dep_list_sorted = construct_dependency_graph(block_list_sorted, construct_output_map(block_list_sorted),
                                                     ignore_helpers=ignore_helpers, calibration=calibration)

        # Draw DAG
        dot = Digraph(comment='Model DAG')

        # Make left-to-right
        if leftright:
            dot.attr(rankdir='LR', ratio='compress', center='true')
        else:
            dot.attr(ratio='auto', center='true')

        # add initial nodes (one for exogenous, one for unknowns) provided those are not empty lists
        if exogenous:
            dot.node('exog', 'exogenous', shape='box')
        if unknowns:
            dot.node('unknowns', 'unknowns', shape='box')
        if targets:
            dot.node('targets', 'targets', shape='diamond')

        # add nodes sequentially in order
        for i in dep_list_sorted:
            if hasattr(block_list_sorted[i], 'hetinput'):
                # HA block
                dot.node(str(i), 'HA [' + str(i) + ']')
            elif hasattr(block_list_sorted[i], 'block_list'):
                # Solved block
                dot.node(str(i), block_list_sorted[i].block_list[0].f.__name__ + '[solved,' + str(i) + ']')
            else:
                # Simple block
                dot.node(str(i), block_list_sorted[i].f.__name__ + ' [' + str(i) + ']')

            # nodes from exogenous to i (figure out if needed and draw)
            if exogenous:
                edgelabel = block_list_sorted[i].inputs & set(exogenous)
                if len(edgelabel) != 0:
                    edgelabel_list = list(edgelabel)
                    edgelabel_str = ', '.join(str(e) for e in edgelabel_list)
                    dot.edge('exog', str(i), label=str(edgelabel_str))

            # nodes from unknowns to i (figure out if needed and draw)
            if unknowns:
                edgelabel = block_list_sorted[i].inputs & set(unknowns)
                if len(edgelabel) != 0:
                    edgelabel_list = list(edgelabel)
                    edgelabel_str = ', '.join(str(e) for e in edgelabel_list)
                    dot.edge('unknowns', str(i), label=str(edgelabel_str))

            #  nodes from i to final targets
            for target in targets:
                if target in block_list_sorted[i].outputs:
                    dot.edge(str(i), 'targets', label=target)

            # nodes from any interior block to i
            for j in dep_list_sorted[i]:
                # figure out inputs of i that are also outputs of j
                edgelabel = block_list_sorted[i].inputs & block_list_sorted[j].outputs
                edgelabel_list = list(edgelabel)
                edgelabel_str = ', '.join(str(e) for e in edgelabel_list)

                # draw edge from j to i
                dot.edge(str(j), str(i), label=str(edgelabel_str))

        if showdag:
            dot.render('dagexport/' + filename, view=True, cleanup=True)
        else:
            dot.render('dagexport/' + filename, format='png', cleanup=True)
            # print(dot.source)

        if debug:
            dep, inputs, outputs = block_sort(block_list_sorted, return_io=True, ignore_helpers=ignore_helpers,
                                              calibration=calibration)
            required = find_outputs_that_are_intermediate_inputs(block_list_sorted, ignore_helpers=ignore_helpers)
            # Candidate targets: outputs that are not inputs to any block
            print("Candidate targets :")
            cand_targets = outputs.difference(required)
            print(cand_targets)
            # Candidate exogenous and unknowns (also includes parameters)
            # inputs that are not outputs of any block
            print("Candidate exogenous/unknowns :")
            cand_xu = inputs.difference(required)
            print(cand_xu)


    def draw_solved(solvedblock, filename='solveddag'):
        # Inspects a solved block by drawing its DAG
        draw_dag([solvedblock.block_list[0]], unknowns=solvedblock.unknowns, targets=solvedblock.targets,
                 filename=filename, showdag=True)


    def inspect_solved(block_list):
        # Inspects all the solved blocks by running through each and drawing its DAG in turn
        for block in block_list:
            if hasattr(block, 'block_list'):
                draw_solved(block, filename=str(block.block_list[0].f.__name__))
except ImportError:
    def draw_dag(*args, **kwargs):
        warnings.warn("\nAttempted to use `draw_dag` when the package `graphviz` has not yet been installed. \n"
                      "DAG visualization tools, i.e. draw_dag, will not produce any figures unless this dependency has been installed. \n"
                      "Once installed, re-load sequence-jacobian to produce DAG figures.")
        pass


    def draw_solved(*args, **kwargs):
        warnings.warn("\nAttempted to use `draw_solved` when the package `graphviz` has not yet been installed. \n"
                      "DAG visualization tools, i.e. draw_dag, will not produce any figures unless this dependency has been installed. \n"
                      "Once installed, re-load sequence-jacobian to produce DAG figures.")
        pass


    def inspect_solved(*args, **kwargs):
        warnings.warn("\nAttempted to use `inspect_solved` when the package `graphviz` has not yet been installed. \n"
                      "DAG visualization tools, i.e. draw_dag, will not produce any figures unless this dependency has been installed. \n"
                      "Once installed, re-load sequence-jacobian to produce DAG figures.")
        pass

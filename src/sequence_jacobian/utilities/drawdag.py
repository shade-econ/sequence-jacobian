import warnings
from sequence_jacobian.utilities import graph

"""
Adrien's DAG Graph routine, updated for SSJ v1.0

Requires installing graphviz package and executables
https://www.graphviz.org/

On a mac this can be done as follows:
1) Download macports at:
https://www.macports.org/install.php
2) On the command line, install graphviz with macports by typing
sudo port install graphviz

"""

try:
    from graphviz import Digraph

    def drawdag(block_list,exogenous=[], unknowns=[], targets=[], showdag=False, leftright=False, filename='model'):
        '''
        Routine that draws DAG
        :param block_list: list of blocks to be represented
        :param exogenous: (optional) exogenous variables, to be represented on DAG
        :param unknowns:  (optional) unknown variables, to be represented on DAG
        :param unknowns:  (optional) target variables, to be represented on DAG
        :bool showdag: if True, export and plot pdf file. If false, export png file and do not plot
        :bool debug: if True, returns list of candidate unknown and targets
        :bool leftright: if True, plots dag from left to right instead of top to bottom
        :return: none
        '''

        # obtain the topological sort 
        topsorted = graph.block_sort(block_list)
        # reorder blocks according to this topological sort (NB: typically blocks will already be sorted)
        block_list_sorted = [block_list[i] for i in topsorted]
        # Obtain the dependency list of the sorted set of blocks
        #inmap = graph.get_input_map(block_list_sorted)
        #adj = graph.get_block_adjacency_list(block_list_sorted, inmap)
        # Obtain the dependency list of the sorted set of blocks
        outmap = graph.get_output_map(block_list_sorted)
        revadj = graph.get_block_reverse_adjacency_list(block_list_sorted, outmap)

        # Start DAG
        dot = Digraph(comment='Model DAG')

        # Make it left-to-right if asked
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
        for i in topsorted:
            #if hasattr(block_list_sorted[i], 'hetinput'):
            if "HetBlock" in str(block_list_sorted[i].__class__):
                # HA block
                dot.node(str(i), block_list_sorted[i].name + ' [HA, ' + str(i) + ']')
            elif "SolvedBlock" in str(block_list_sorted[i].__class__):
                # Solved block
                dot.node(str(i), block_list_sorted[i].name + ' [solved,' + str(i) + ']')
            else:
                # Simple block
                dot.node(str(i), block_list_sorted[i].name + ' [' + str(i) + ']')

            # nodes from exogenous to i (figure out if needed and draw)
            if exogenous:
                edgelabel = block_list_sorted[i].inputs & set(exogenous)
                if len(edgelabel) != 0:
                    edgelabel_list = list(edgelabel)
                    edgelabel_str = ', '.join(str(e) for e in edgelabel_list)
                    dot.edge('exog', str(i), label=str(edgelabel_str))

            # nodes from unknowns to i (figure out if needed, then draw)
            if unknowns:
                edgelabel = block_list_sorted[i].inputs & set(unknowns)
                if len(edgelabel) != 0:
                    edgelabel_list = list(edgelabel)
                    edgelabel_str = ', '.join(str(e) for e in edgelabel_list)
                    dot.edge('unknowns', str(i), label=str(edgelabel_str))
            
            # nodes from i to final targets
            for target in targets:
                if target in block_list_sorted[i].outputs:
                    dot.edge(str(i), 'targets', label=target)
                        
            # nodes from any interior block to i
            for j in revadj[i]:
                # figure out inputs of i that are also outputs of j
                edgelabel = block_list_sorted[i].inputs & block_list_sorted[j].outputs
                edgelabel_list = list(edgelabel)
                edgelabel_str = ', '.join(str(e) for e in edgelabel_list)
                
                # draw edge from j to i
                dot.edge(str(j), str(i), label=str(edgelabel_str))

        if showdag:
            dot.render('dag/' + filename, view=True, cleanup=True)
        else:
            dot.render('dag/' + filename, format='png', cleanup=True)
            #print(dot.source)

except ImportError:
    def draw_dag(*args, **kwargs):
        warnings.warn("\nAttempted to use `draw_dag` when the package `graphviz` has not yet been installed. \n"
                      "DAG visualization tools, i.e. draw_dag, will not produce any figures unless this dependency has been installed. \n"
                      "Once installed, re-load sequence-jacobian to produce DAG figures.")

import warnings
from sequence_jacobian.blocks.solved_block import SolvedBlock
from sequence_jacobian.blocks.het_block import HetBlock

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
    from IPython.display import display

    def drawdag(model, exogenous=[], unknowns=[], targets=[], leftright=False, save=False, savepath=None):
        '''
        Routine that draws DAG
        :param model: combined block to be represented as dag
        :param exogenous: (optional) exogenous variables, to be represented on DAG
        :param unknowns:  (optional) unknown variables, to be represented on DAG
        :param unknowns:  (optional) target variables, to be represented on DAG
        :bool leftright: if True, plots dag from left to right instead of top to bottom
        :return: none
        '''
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
        for i, b in enumerate(model.blocks):
            if isinstance(b, HetBlock):
                dot.node(str(i), b.name + ' [HA, ' + str(i) + ']')
            elif isinstance(b, SolvedBlock) :
                dot.node(str(i), b.name + ' [solved,' + str(i) + ']')
            else:
                dot.node(str(i), b.name + ' [' + str(i) + ']')

            # nodes from exogenous to i (figure out if needed and draw)
            if exogenous:
                edgelabel = b.inputs & set(exogenous)
                if len(edgelabel) != 0:
                    edgelabel_list = list(edgelabel)
                    edgelabel_str = ', '.join(str(e) for e in edgelabel_list)
                    dot.edge('exog', str(i), label=str(edgelabel_str))

            # nodes from unknowns to i (figure out if needed, then draw)
            if unknowns:
                edgelabel = b.inputs & set(unknowns)
                if len(edgelabel) != 0:
                    edgelabel_list = list(edgelabel)
                    edgelabel_str = ', '.join(str(e) for e in edgelabel_list)
                    dot.edge('unknowns', str(i), label=str(edgelabel_str))
            
            # nodes from i to final targets
            for target in targets:
                if target in b.outputs:
                    dot.edge(str(i), 'targets', label=target)
                        
            # nodes from any interior block to i
            for j in model.revadj[i]:
                # figure out inputs of i that are also outputs of j
                edgelabel = b.inputs & model.blocks[j].outputs
                edgelabel_list = list(edgelabel)
                edgelabel_str = ', '.join(str(e) for e in edgelabel_list)
                
                # draw edge from j to i
                dot.edge(str(j), str(i), label=str(edgelabel_str))

        if save:
            if savepath is None:
                savepath = 'dag/' + model.name
            dot.render(savepath, format='png', cleanup=True)
        display(dot)

except ImportError:
    def drawdag(*args, **kwargs):
        warnings.warn("\nAttempted to use `drawdag` when the package `graphviz` has not yet been installed. \n"
                      "DAG visualization tools, i.e. drawdag, will not produce any figures unless this dependency has been installed. \n"
                      "If you want to install, try typing 'conda install -c conda-forge python-graphviz' at the terminal,\n"
                      "or see README for more instructions. Once installed, re-load sequence-jacobian to produce DAG figures.")

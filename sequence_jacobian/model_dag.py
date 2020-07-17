# Not currently being used due to refactoring. Delete if this ends up not being necessary
"""The Directed Acyclic Graph (DAG) representation of a Model"""

from typing import List


class ModelDAG:
    blocks: List  # Handle more elegantly later w/ a list of Block types, where simple/het/solved inherit from it
    exogenous_variables: List[str]
    unknown_variables: List[str]
    targets: List[str]

    def __init__(self, blocks, exogenous_variables, unknown_variables, targets):
        self.blocks = blocks
        self.exogenous_variables = exogenous_variables
        self.unknown_variables = unknown_variables
        self.targets = targets
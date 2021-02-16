"""Type aliases, custom functions and errors for the base-level functionality of the package"""

from .primitives import Block
from .blocks.combined_block import CombinedBlock, combine


# Useful type aliases
Model = CombinedBlock


# Useful functional aliases
def create_model(*args, **kwargs):
    return combine(*args, model_alias=True, **kwargs)

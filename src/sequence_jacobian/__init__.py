"""Public-facing objects."""

from . import estimation, utilities, grids, interpolate, misc, hetblocks

from .blocks.simple_block import simple
from .blocks.het_block import het
from .blocks.solved_block import solved
from .blocks.combined_block import combine, create_model
from .blocks.support.simple_displacement import apply_function
from .classes.steady_state_dict import SteadyStateDict
from .classes.impulse_dict import ImpulseDict
from .classes.jacobian_dict import JacobianDict
from .utilities.drawdag import drawdag

from .utilities.distributions import *
from .utilities.shocks import AR, ARMA, MA, ShockDict
from .estimation import DensityModel

# Ensure warning uniformity across package
import warnings

# Force warnings.warn() to omit the source code line in the message
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
    formatwarning_orig(message, category, filename, lineno, line='')

# deprecation of old ways for calling things
def agrid(*args, **kwargs):
    warnings.warn("The function 'agrid' is deprecated and will be removed in a subsequent version.\n"
                  "Please call sj.grids.asset_grid(amin, amax, n) instead.")
    return utilities.discretize.agrid(*args, **kwargs)

def markov_rouwenhorst(*args, **kwargs):
    warnings.warn("Calling sj.markov_rouwenhorst() is deprecated and will be disallowed in a subsequent version.\n"
                  "Please call sj.grids.markov_rouwenhorst() instead.")
    return grids.markov_rouwenhorst(*args, **kwargs)

def markov_tauchen(*args, **kwargs):
    warnings.warn("Calling sj.markov_tauchen() is deprecated and will be disallowed in a subsequent version.\n"
                  "Please call sj.grids.markov_tauchen() instead.")
    return grids.markov_tauchen(*args, **kwargs)

def interpolate_y(*args, **kwargs):
    warnings.warn("Calling sj.interpolate_y() is deprecated and will be disallowed in a subsequent version.\n"
                  "Please call sj.interpolate.interpolate_y() instead.")
    return interpolate.interpolate_y(*args, **kwargs)

def setmin(*args, **kwargs):
    warnings.warn("Calling sj.setmin() is deprecated and will be disallowed in a subsequent version.\n"
                  "Please call sj.misc.setmin() instead.")
    misc.setmin(*args, **kwargs)

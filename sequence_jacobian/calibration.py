"""Some useful functionality for steady state calibration exercises"""
from typing import Optional, Dict


# Feature to-be-implemented: Allow init_value/lower_bound/upper_bound to be functions of the target corresponding to the
# instrument's value. e.g. in the ks_ss() code, beta_min = lb/(1+r) and beta_max = ub/(1+r), where "r" is the target
# corresponding to the instrument, beta
# Requirement: Imposing that there is a target that directly corresponds to an instrument and vice versa (for some
# consistency checks when constructing a CalibrationSet later on).
class CalibrationInstrument:
    name: str
    target_name: str
    init_value: Optional[float]
    lower_bound: Optional[float]
    upper_bound: Optional[float]

    def __init__(self, name, target_name, init_value=None, lower_bound=None, upper_bound=None):
        if init_value is None and lower_bound is None and upper_bound is None:
            raise ValueError("Must specify either an initial value, or a lower/upper bound to properly instantiate"
                             " this instrument.")
        if init_value is None and (lower_bound is None or upper_bound is None):
            raise ValueError("If instantiating an instrument with bounds, must provide both lower_bound"
                             " and upper_bound.")

        self.name = name
        self.target_name = target_name
        if init_value is not None:
            self.init_value = init_value
        if lower_bound is not None and upper_bound is not None:
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

    def __repr__(self):
        return self.name + ": An instrument for calibrating a model's steady state values/parameters."


class CalibrationTarget:
    name: str
    instrument_name: str
    target_value: float

    def __init__(self, name, instrument_name, target_value):
        self.name = name
        self.instrument_name = instrument_name
        self.target_value = target_value

    def __repr__(self):
        return self.name + ": A target for calibrating a model's steady state values/parameters."


class CalibrationSet:
    instruments: Dict[str, CalibrationInstrument]
    targets: Dict[str, CalibrationTarget]

    def __init__(self, instruments, targets):
        self.instruments = instruments
        self.targets = targets

    def __repr__(self):
        return "The set of instruments and targets for calibrating a model's steady state values/parameters"

    # Define some useful getters
    # To check:
    def get_instrument_names(self):
        return self.instruments.keys()

    def get_target_names(self):
        return self.targets.keys()

    def get_instrument_init_values(self):
        return [instr.init_value for instr in self.instruments.values()]

    def get_instrument_bounds(self):
        return [(instr.lower_bound, instr.upper_bound) for instr in self.instruments.values()]

    def get_target_values(self):
        return [target.target_value for target in self.targets.values()]



# Define some useful setters
# The primary user-facing constructor for the CalibrationSet
def construct_calibration_set(target_names, target_values, instrument_names,
                              instrument_init_values=None, instrument_bounds=None):
    assert len(target_names) == len(target_values)

    if instrument_init_values is None and instrument_bounds is None:
        raise ValueError("Must specify either a set of initial values or a set of instrument_bounds to properly"
                         "instantiate a list of instruments.")
    elif instrument_init_values is not None and instrument_bounds is None:
        assert len(instrument_init_values) == len(instrument_names)
        instrument_list = {name: CalibrationInstrument(name, target_name, init_value=init_value)
                           for name, target_name, init_value
                           in zip(instrument_names, target_names, instrument_init_values)}
    elif instrument_init_values is None and instrument_bounds is not None:
        assert len(instrument_bounds) == len(instrument_names)
        instrument_list = {name: CalibrationInstrument(name, target_name, lower_bound=bds[0], upper_bound=bds[1])
                           for name, target_name, bds in zip(instrument_names, target_names, instrument_bounds)}
    else:
        instrument_list = {name: CalibrationInstrument(name, init_value=init_value, lower_bound=bds[0], upper_bound=bds[1])
                           for name, target_name, init_value, bds
                           in zip(instrument_names, target_names, instrument_init_values, instrument_bounds)}

    target_list = {name: CalibrationTarget(name, instrument_name, target_value)
                   for name, instrument_name, target_value in zip(target_names, instrument_names, target_values)}

    return CalibrationSet(instrument_list, target_list)

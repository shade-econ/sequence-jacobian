import numpy as np
import sequence_jacobian as sj


def test_jacobian_accumulation():

    # Define two blocks. Notice: Second one does not use output from the first!
    @sj.solved(unknowns={'p': (-10, 1000)}, targets=['valuation'] , solver="brentq")
    def equity(r1, p, Y):
        valuation = Y + p(+1) / (1 + r1) - p
        return valuation

    @sj.simple
    def mkt_clearing(r0, r1, A0, A1, Y, B):
        asset_mkt_0 = A0 + (r0 + Y - 0.5*r1) - B
        asset_mkt_1 = A1 + (r1 + Y - 0.5*r0) - B
        return asset_mkt_0, asset_mkt_1

    both_blocks = sj.create_model([equity, mkt_clearing])
    only_second = sj.create_model([mkt_clearing])

    calibration = {'B': 0, 'Y': 0, 'r0': 0.01/4, 'r1': 0.01/4, 'A0': 1, 'A1': 1}
    ss_both = both_blocks.steady_state(calibration)
    ss_second = only_second.steady_state(calibration)

    # Second block alone gives us Jacobian without issues.

    unknowns_td = ['Y', 'r1']
    targets_td = ['asset_mkt_0', 'asset_mkt_1']
    T = 300
    shock = {'r0': 0.95**np.arange(T)}
    irf = only_second.solve_impulse_linear(ss_second, unknowns_td, targets_td, shock)
    G = only_second.solve_jacobian(ss_second, unknowns_td, targets_td, ['r0'], T=T)

    # Both blocks give us trouble. Even though solve_impulse_linear runs through...

    unknowns_td = ['Y', 'r1']
    targets_td = ['asset_mkt_0', 'asset_mkt_1']
    T = 300
    shock = {'r0': 0.95**np.arange(T)}
    irf = both_blocks.solve_impulse_linear(ss_both, unknowns_td, targets_td, shock)
    G = both_blocks.solve_jacobian(ss_both, unknowns_td, targets_td, ['r0'], T=T)
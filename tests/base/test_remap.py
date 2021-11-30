import numpy as np
from sequence_jacobian import simple, solved, combine


@simple
def matching(theta, ell, kappa):
    f = theta / (1 + theta ** ell) ** (1 / ell)
    qfill = f / theta
    hiring_cost = kappa / qfill
    return f, qfill, hiring_cost


@solved(unknowns={'h': (0, 1)}, targets=['jc_res'])
def job_creation(h, w, beta, s, hiring_cost):
    jc_res = h - w + beta * (1 - s(+1)) * hiring_cost(+1) - hiring_cost
    return jc_res


@solved(unknowns={'N': (0.5, 1)}, targets=['N_lom'])
def labor_lom(h, w, N, s, qfill, f, theta, hiring_cost):
    N_lom = (1 - s * (1 - f)) * N(-1) + f * (1 - N(-1)) - N
    U = 1 - N
    v = theta * U
    vacancy_cost = hiring_cost * qfill * v
    Div_labor = (h - w) * N - vacancy_cost
    return N_lom, U, v, vacancy_cost, Div_labor


@simple
def dmp_aggregate(U_men, U_women, Div_labor_men, Div_labor_women, vacancy_cost_men, vacancy_cost_women):
    U = (U_men + U_women) / 2
    Div_labor = (Div_labor_men + Div_labor_women) / 2
    vacancy_cost = (vacancy_cost_men + vacancy_cost_women) / 2
    return U, Div_labor, vacancy_cost


def test_remap_combined_block():
    dmp = combine([matching, job_creation, labor_lom], name='DMP')
    dmp_men = dmp.rename('_men')
    dmp_women = dmp.rename('_women')

    # remap some inputs and all outputs
    to_remap = ['theta', 'ell', 's'] + list(dmp_men.outputs)
    dmp_men = dmp_men.remap({k: k + '_men' for k in to_remap})
    dmp_women = dmp_women.remap({k: k + '_women' for k in to_remap})

    # combine remapped blocks
    dmp_all = combine([dmp_men, dmp_women, dmp_aggregate], name='dmp_all')
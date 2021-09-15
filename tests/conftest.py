"""Fixtures used by tests."""

import pytest

from sequence_jacobian.examples import rbc, krusell_smith, hank, two_asset


@pytest.fixture(scope='session')
def rbc_dag():
    return rbc.dag()


@pytest.fixture(scope='session')
def krusell_smith_dag():
    return krusell_smith.dag()


@pytest.fixture(scope='session')
def one_asset_hank_dag():
    return hank.dag()


@pytest.fixture(scope='session')
def two_asset_hank_dag():
    return two_asset.dag()


@pytest.fixture(scope='session')
def ks_remapped_dag():
    return krusell_smith.remapped_dag()

import numpy as np
import pytest

from scripts.simulate_data import psi


def test_psi_identical_and_shifted_distributions():
    reference = np.linspace(0, 100, 1000)
    assert psi(reference, reference) == pytest.approx(0)
    assert psi(reference, reference + 200) > 0.25


def test_constant_reference_detects_new_values():
    reference = np.ones(100)
    assert psi(reference, reference) == pytest.approx(0)
    assert psi(reference, np.zeros(100)) > 0.25
    assert psi(reference, np.full(100, 2)) > 0.25


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf]])
def test_invalid_samples(values):
    with pytest.raises(ValueError):
        psi([1, 2, 3], values)

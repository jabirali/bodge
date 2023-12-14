import pytest

from bodge.common import *


def test_pauli():
    # Test that the quaternion identities hold.
    assert np.allclose(σ1 @ σ1, σ0)
    assert np.allclose(σ2 @ σ2, σ0)
    assert np.allclose(σ3 @ σ3, σ0)

    assert np.allclose(σ1 @ σ2, jσ3)
    assert np.allclose(σ2 @ σ3, jσ1)
    assert np.allclose(σ3 @ σ1, jσ2)

    assert np.allclose(σ1 @ σ2 @ σ3, jσ0)

import pytest
import numpy as np
from app import lorenz_system, rk4_step

def test_lorenz_system():
    # Test steady state for trivial parameters
    state = np.array([0.0, 0.0, 0.0])
    derivatives = lorenz_system(state, 0.0, 10.0, 28.0, 8.0/3.0)
    assert np.allclose(derivatives, np.zeros(3))

def test_rk4_step():
    state = np.array([1.0, 1.0, 1.0])
    dt = 0.01
    next_state = rk4_step(state, 0.0, dt, 10.0, 28.0, 8.0/3.0)
    # The output should not be completely unchanged for non-steady states
    assert not np.allclose(state, next_state)

import pytest
import numpy as np

# Global seeding for deterministic behavior
@pytest.fixture(autouse=True)
def set_global_seed():
    np.random.seed(123)
    yield


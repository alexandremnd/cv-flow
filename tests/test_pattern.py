import pytest

from cvflow.command import N, E, M, X, Z
from cvflow.pattern import Pattern

def test_invalid_action_before_initialisation():
    for cmd in [E(nodes=(0, 1)), M(node=0), X(node=0), Z(node=0)]:
        with pytest.raises(ValueError):
            Pattern([cmd, N(node=0), N(node=1)])

def test_invalid_action_different_node():
    for cmd in [E(nodes=(1, 2)), M(node=2), X(node=2), Z(node=2)]:
        with pytest.raises(ValueError):
            Pattern([cmd, N(node=0), N(node=1)])

def test_valid_pattern():
    Pattern([
        N(node=0),
        N(node=1),
        E(nodes=(0, 1)),
        M(node=0),
        X(node=1),
        Z(node=1)
    ])
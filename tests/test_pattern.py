import pytest

from cvflow.command import N, E, M, X, Z
from cvflow.pattern import Pattern

def test_measurement_domain_violation():
    with pytest.raises(ValueError):
        Pattern([
            N(node=0),
            N(node=1),
            E(nodes=(0, 1)),
            M(node=0, x_domain={1: 0.0}),
        ], input_nodes=[])

    with pytest.raises(ValueError):
        Pattern([
            N(node=0),
            N(node=1),
            E(nodes=(0, 1)),
            M(node=0, x_domain={2: 0.0}),
        ], input_nodes=[])

    with pytest.raises(ValueError):
        Pattern([
            N(node=0),
            N(node=1),
            E(nodes=(0, 1)),
            M(node=0, z_domain={1: 0.0}),
        ], input_nodes=[])

    with pytest.raises(ValueError):
        Pattern([
            N(node=0),
            N(node=1),
            E(nodes=(0, 1)),
            M(node=0, z_domain={2: 0.0}),
        ], input_nodes=[])

def test_correction_domain_violation():
    with pytest.raises(ValueError):
        Pattern([
            N(node=0),
            N(node=1),
            E(nodes=(0, 1)),
            X(node=1, x_domain={0: 0.0}),
        ], input_nodes=[])

    with pytest.raises(ValueError):
        Pattern([
            N(node=0),
            N(node=1),
            E(nodes=(0, 1)),
            X(node=1, x_domain={2: 0.0}),
        ], input_nodes=[])

    with pytest.raises(ValueError):
        Pattern([
            N(node=0),
            N(node=1),
            E(nodes=(0, 1)),
            Z(node=1, z_domain={0: 0.0}),
        ], input_nodes=[])

    with pytest.raises(ValueError):
        Pattern([
            N(node=0),
            N(node=1),
            E(nodes=(0, 1)),
            Z(node=1, z_domain={2: 0.0}),
        ], input_nodes=[])

def test_invalid_action_before_initialisation():
    for cmd in [E(nodes=(0, 1)), M(node=0), X(node=0), Z(node=0)]:
        with pytest.raises(ValueError):
            Pattern([cmd, N(node=0), N(node=1)], input_nodes=[])

def test_invalid_action_different_node():
    for cmd in [E(nodes=(1, 2)), M(node=2), X(node=2), Z(node=2)]:
        with pytest.raises(ValueError):
            Pattern([cmd, N(node=0), N(node=1)], input_nodes=[])

def test_valid_pattern():
    Pattern([
        N(node=0),
        N(node=1),
        E(nodes=(0, 1)),
        M(node=0),
        X(node=1),
        Z(node=1)
    ], input_nodes=[])

def test_invalid_pattern():
    with pytest.raises(ValueError):
        Pattern([
            N(node=0),
            N(node=1),
            E(nodes=(0, 1)),
            M(node=0),
            X(node=1),
            Z(node=1)
        ], input_nodes=[0])
import networkx as nx
import pytest
from cvflow.graph import OpenGraph
from cvflow.flow import check_flow

import numpy as np

def test_check_flow_cycle_graph():
    # ===== Test case 1: Cycle graph =====
    cycle = nx.cycle_graph(6)

    og = OpenGraph(
        graph=cycle,
        input_nodes=[1, 3, 5],
        output_nodes=[0, 2, 4]
    )

    res, corrections = check_flow(og, [1, 3, 5])

    expected_corrections = [
        np.array([0.5, 0, 0.5, 0, 0, 0]),  # Correction for measuring node 1
        np.array([-0.33333333, 0, 0.33333333, 0, 0.66666667, 0]),  # Correction for measuring node 3
        np.array([0.5, 0, -0.5, 0, 0.5, 0])   # Correction for measuring node 5
    ]
    assert(res)
    for _, (actual, expected) in enumerate(zip(corrections, expected_corrections)):
        assert actual == pytest.approx(expected)

def test_check_flow_hexagon_graph():
    edges = [
        (0, 1), (0, 2),
        (1, 3), (2, 3),
        (1, 4), (2, 5),
        (3, 0), (3, 6),
        (4, 6), (5, 6),
    ]
    hexagon = nx.Graph(edges)

    og = OpenGraph(
        graph=hexagon,
        input_nodes=[5, 4, 0, 3],
        output_nodes=[1, 2, 6]
    )
    res, _ = check_flow(og, [5, 4, 0, 3])

    assert not res

def test_check_flow_linear_graph():
    linear = nx.path_graph(5)
    og = OpenGraph(
        graph=linear,
        input_nodes=[0],
        output_nodes=[4]
    )
    res, corrections = check_flow(og, [0, 1, 2, 3])

    expected_corrections = [
        np.array([0, 1, 0, 0, 0]),
        np.array([0, 0, 1, 0, 0]),
        np.array([0, 0, 0, 1, 0]),
        np.array([0, 0, 0, 0, 1])
    ]

    assert res
    for _, (actual, expected) in enumerate(zip(corrections, expected_corrections)):
        assert actual == pytest.approx(expected)
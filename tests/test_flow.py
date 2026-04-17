import networkx as nx
import pytest

from cvflow.graph import OpenGraph
from cvflow.flow import check_flow, find_cvflow


# ──────────────────────────── fixtures ────────────────────────────

@pytest.fixture
def path_og():
    """Path graph 0-1-2-3-4, input=[0], output=[4]."""
    return OpenGraph(graph=nx.path_graph(5), input_nodes=[0], output_nodes=[4])


@pytest.fixture
def cycle_og():
    """Cycle graph on 6 nodes, input=[1,3,5], output=[0,2,4]."""
    return OpenGraph(graph=nx.cycle_graph(6), input_nodes=[1, 3, 5], output_nodes=[0, 2, 4])


@pytest.fixture
def no_flow_og():
    """Graph on 7 nodes with no CV-flow."""
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 3),
        (1, 4), (2, 5), (3, 0), (3, 6),
        (4, 6), (5, 6),
    ]
    return OpenGraph(graph=nx.Graph(edges), input_nodes=[5, 4, 0, 3], output_nodes=[1, 2, 6])


# ──────────────────────── path graph ─────────────────────────────
# Simple chain: corrections are always 1.0 on the single next node.

class TestPathGraph:
    MEASUREMENTS = [0, 1, 2, 3]

    def test_check_flow_exists(self, path_og):
        has_flow, _, _ = check_flow(path_og, self.MEASUREMENTS)
        assert has_flow

    def test_check_flow_corrections(self, path_og):
        _, g, _ = check_flow(path_og, self.MEASUREMENTS)
        assert g[0] == pytest.approx({1: 1.0})
        assert g[1] == pytest.approx({2: 1.0})
        assert g[2] == pytest.approx({3: 1.0})
        assert g[3] == pytest.approx({4: 1.0})

    def test_check_flow_layers(self, path_og):
        _, _, layer = check_flow(path_og, self.MEASUREMENTS)
        assert layer == {4: 0, 3: 1, 2: 2, 1: 3, 0: 4}

    def test_check_flow_wrong_order_fails(self, path_og):
        # Measuring in reverse order cannot satisfy the flow constraint.
        has_flow, g, layer = check_flow(path_og, [3, 2, 1])
        assert not has_flow
        assert g == {}
        assert layer == {}

    def test_find_cvflow_exists(self, path_og):
        has_flow, _, _ = find_cvflow(path_og)
        assert has_flow

    def test_find_cvflow_corrections(self, path_og):
        _, g, _ = find_cvflow(path_og)
        assert g[3] == pytest.approx({4: 1.0})
        assert g[2] == pytest.approx({3: 1.0})
        assert g[1] == pytest.approx({2: 1.0})
        assert g[0] == pytest.approx({1: 1.0})

    def test_find_cvflow_layers(self, path_og):
        _, _, layer = find_cvflow(path_og)
        assert layer == {4: 0, 3: 1, 2: 2, 1: 3, 0: 4}


# ──────────────────────── cycle graph ─────────────────────────────
# Corrections are non-trivial fractions; we verify structure and key values.

class TestCycleGraph:
    MEASUREMENTS = [1, 3, 5]

    def test_check_flow_exists(self, cycle_og):
        has_flow, _, _ = check_flow(cycle_og, self.MEASUREMENTS)
        assert has_flow

    def test_check_flow_corrections(self, cycle_og):
        _, g, _ = check_flow(cycle_og, self.MEASUREMENTS)
        assert g[1] == pytest.approx({0: 0.5, 2: 0.5})
        assert g[3] == pytest.approx({0: -1 / 3, 2: 1 / 3, 4: 2 / 3})
        assert g[5] == pytest.approx({0: 0.5, 2: -0.5, 4: 0.5})

    def test_find_cvflow_exists(self, cycle_og):
        has_flow, _, _ = find_cvflow(cycle_og)
        assert has_flow

    def test_find_cvflow_corrections(self, cycle_og):
        _, g, _ = find_cvflow(cycle_og)
        # Since we measure simulataneously on 1, 3, 5, the corrections are not always the same as in check_flow
        assert g[1] == pytest.approx({0: 0.5, 2: 0.5, 4: -0.5})
        assert g[3] == pytest.approx({0: -0.5, 2: 0.5, 4: 0.5})
        assert g[5] == pytest.approx({0: 0.5, 2: -0.5, 4: 0.5})



# ──────────────────────── no-flow graph ───────────────────────────

class TestNoFlowGraph:
    MEASUREMENTS = [5, 4, 0, 3]

    def test_check_flow_not_exists(self, no_flow_og):
        has_flow, g, layer = check_flow(no_flow_og, self.MEASUREMENTS)
        assert not has_flow
        assert g == {}
        assert layer == {}

    def test_find_cvflow_not_exists(self, no_flow_og):
        has_flow, g, layer = find_cvflow(no_flow_og)
        assert not has_flow
        assert g == {}
        assert layer == {}


# ──────────────────────── error handling ──────────────────────────

def test_check_flow_raises_on_unknown_node(path_og):
    with pytest.raises(ValueError):
        check_flow(path_og, [99])

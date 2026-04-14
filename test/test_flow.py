import networkx as nx
import pytest
from cvflow.graph import OpenGraph
from cvflow.flow import check_flow

def test_check_flow():
    # ===== Test case 1: Simple cycle graph =====
    cycle = nx.cycle_graph(6)

    og = OpenGraph(
        graph=cycle,
        input_nodes=[1, 3, 5],
        output_nodes=[0, 2, 4]
    )

    res = check_flow(og, [1, 3, 5])

    assert(res)

    # ===== Test case 2: Hexagon graph =====
    edges = [
        (0, 1), (0, 2),       # top to top-left and top-right
        (1, 3), (2, 3),       # top-left and top-right to center
        (1, 4), (2, 5),       # top-left to bottom-left, top-right to bottom-right
        (3, 0), (3, 6),       # center to bottom-left and bottom-right
        (4, 6), (5, 6),       # bottom-left and bottom-right to bottom
    ]
    hexagon = nx.Graph(edges)

    og = OpenGraph(
        graph=hexagon,
        input_nodes=[5, 4, 0, 3],
        output_nodes=[1, 2, 6]
    )
    res = check_flow(og, [5, 4, 0, 3])
    assert(not res)
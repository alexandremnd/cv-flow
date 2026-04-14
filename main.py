from cvflow.pattern import Pattern
from cvflow.command import N, M, E, X, Z
from cvflow.flow import find_cvflow, check_flow
from cvflow.graph import OpenGraph
import networkx as nx

star_edges = [
        (0, 1), (0, 2),       # top to top-left and top-right
        (1, 3), (2, 3),       # top-left and top-right to center
        (1, 4), (2, 5),       # top-left to bottom-left, top-right to bottom-right
        (3, 0), (3, 6),       # center to bottom-left and bottom-right
        (4, 6), (5, 6),       # bottom-left and bottom-right to bottom
    ]

def main():
    cycle = nx.cycle_graph(6)

    og = OpenGraph(
        graph=cycle,
        input_nodes=[1, 3, 5],
        output_nodes=[0, 2, 4]
    )
    res = check_flow(og, [1, 3, 5])

    # star = nx.Graph(star_edges)
    # og = OpenGraph(
    #     graph=hexagon,
    #     input_nodes=[5, 4, 0, 3],
    #     output_nodes=[1, 2, 6]
    # )
    # res = check_flow(og, [5, 4, 0, 3])

    # print(res)
    # print(g)
    # print(layer)

    og.visualise()


if __name__ == "__main__":
    main()

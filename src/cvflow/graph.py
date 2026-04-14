from typing import Iterable

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class OpenGraph:
    def __init__(self, graph: nx.Graph, input_nodes: Iterable[int], output_nodes: Iterable[int]):
        """Initializes an OpenGraph with the given graph, input nodes, and output nodes.

        Args:
            graph (nx.Graph): The graph to initialize with.
            input_nodes (Iterable[int]): The input nodes.
            output_nodes (Iterable[int]): The output nodes.
        """

        self.graph: nx.Graph                = graph
        self._nodes : set[int]              = set(graph.nodes)
        self._input_nodes: set[int]         = set(input_nodes)
        self._output_nodes: set[int]        = set(output_nodes)
        self._adjacency_matrix: np.ndarray  = nx.to_numpy_array(self.graph)

        self.__validate__()

    def __validate__(self):
        """Validates the graph structure, ensuring that input and output nodes are correctly defined."""
        if not all(node in self._nodes for node in self._input_nodes):
            raise ValueError("All input nodes must be present in the graph.")

        if not all(node in self._nodes for node in self._output_nodes):
            raise ValueError("All output nodes must be present in the graph.")

        if min(self._nodes) != 0:
            raise ValueError("Node indices must start from 0.")

    @property
    def number_of_nodes(self) -> int:
        """Returns the number of nodes in the graph.

        Returns:
            int: The number of nodes in the graph.
        """
        return len(self._nodes)

    @property
    def number_of_input_nodes(self) -> int:
        """Returns the number of input nodes in the graph.

        Returns:
            int: The number of input nodes in the graph.
        """
        return len(self._input_nodes)

    @property
    def number_of_output_nodes(self) -> int:
        """Returns the number of output nodes in the graph.

        Returns:
            int: The number of output nodes in the graph.
        """
        return len(self._output_nodes)

    @property
    def non_input_nodes(self) -> list[int]:
        """Returns the list of non-input nodes in the graph.

        Returns:
            list[int]: The list of non-input nodes in the graph.
        """
        return self.get_nodes_complement(self.input_nodes)

    @property
    def output_nodes(self) -> list[int]:
        """Returns the list of output nodes.

        Returns:
            list[int]: The list of output nodes.
        """
        return list(self._output_nodes) # copy of the output nodes list to avoid external modifications

    @property
    def input_nodes(self) -> list[int]:
        """Returns the list of input nodes.

        Returns:
            list[int]: The list of input nodes.
        """
        return list(self._input_nodes) # copy of the input nodes list to avoid external modifications

    @property
    def nodes(self) -> list[int]:
        """Returns a list of all nodes in the graph.

        Returns:
            list[int]: A list of all nodes in the graph.
        """
        return list(self._nodes) # copy of the nodes list to avoid external modifications

    def get_adjacency_matrix(self, rows_id: list[int], cols_id: list[int]) -> np.ndarray:
        """Returns the adjacency matrix for the specified rows and columns.

        Args:
            rows_id (list[int]): The row indices for the adjacency matrix.
            cols_id (list[int]): The column indices for the adjacency matrix.

        Returns:
            np.ndarray: The adjacency matrix for the specified rows and columns.
        """
        return self._adjacency_matrix[np.ix_(rows_id, cols_id)]

    def get_nodes_complement(self, nodes: Iterable[int]) -> list[int]:
        """Returns the list of nodes that are not in the given list of nodes.

        Args:
            nodes (Iterable[int]): The list of nodes to find the complimentary nodes for.
        """
        return list(self._nodes - set(nodes)) # copy of the nodes list to avoid external modifications


    def __visualise__(self):
        """Internal method to visualize the graph using Matplotlib.
        Nodes are colored based on their type (input, output, or interior). Edges are labeled with their weights.
        """
        pos = nx.spring_layout(self.graph, seed=42)

        node_colors = []
        for node in self.graph.nodes:
            if node in self.input_nodes and node in self.output_nodes:
                node_colors.append("#a855f7")  # purple: both input and output
            elif node in self.input_nodes:
                node_colors.append("#22c55e")  # green: input
            elif node in self.output_nodes:
                node_colors.append("#ef4444")  # red: output
            else:
                node_colors.append("#3b82f6")  # blue: interior (measured)

        edge_labels = {(u, v): f"{self.graph[u][v].get('weight', 1.0):.2f}" for u, v in self.graph.edges}

        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=600)
        nx.draw_networkx_labels(self.graph, pos, font_color="white", font_weight="bold")
        nx.draw_networkx_edges(self.graph, pos, width=2.0, alpha=0.7)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#22c55e", markersize=10, label="Input"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#ef4444", markersize=10, label="Output"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#3b82f6", markersize=10, label="Interior (Measured)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#a855f7", markersize=10, label="Input & Output"),
        ]

        plt.legend(handles=legend_elements, loc="best", frameon=False)
        plt.title("OpenGraph")
        plt.axis("off")
        plt.tight_layout()

    def output_svg(self, filename: str):
        """Outputs the graph as an SVG file.

        Args:
            filename (str): The name of the file to save the SVG to.
        """
        self.__visualise__()
        plt.savefig(filename, format="svg")
        plt.close()

    def visualise(self):
        """Visualizes the graph using Matplotlib."""
        self.__visualise__()
        plt.show()
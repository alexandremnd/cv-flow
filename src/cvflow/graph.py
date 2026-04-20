from typing import Iterable

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class OpenGraph:
    def __init__(self, graph: nx.Graph, input_nodes: Iterable[int], output_nodes: Iterable[int]):
        """Initializes an OpenGraph around a given NetworkX graph, input nodes, and output nodes.

        Parameters
        ----------
        graph : nx.Graph
            The underlying NetworkX graph.
        input_nodes : Iterable[int]
            The input nodes.
        output_nodes : Iterable[int]
            The output nodes.
        """

        self._graph: nx.Graph                = graph
        self._nodes : set[int]              = set(graph.nodes)
        self._input_nodes: set[int]         = set(input_nodes)
        self._output_nodes: set[int]        = set(output_nodes)
        self._adjacency_matrix: np.ndarray  = nx.to_numpy_array(self._graph)

        self.__validate__()

    def __validate__(self):
        """Validate the graph structure.

        Ensures that input and output nodes are present in the graph and that
        node indices start from 0.

        Raises
        ------
        ValueError
            If any input or output node is not in the graph, or if the minimum
            node index is not 0.
        """
        if not all(node in self._nodes for node in self._input_nodes):
            raise ValueError("All input nodes must be present in the graph.")

        if not all(node in self._nodes for node in self._output_nodes):
            raise ValueError("All output nodes must be present in the graph.")

        if min(self._nodes) != 0:
            raise ValueError("Node indices must start from 0.")

    @property
    def number_of_nodes(self) -> int:
        """Total number of nodes in the graph.

        Returns
        -------
        number_of_nodes : int
            The number of nodes in the graph.
        """
        return len(self._nodes)

    @property
    def number_of_input_nodes(self) -> int:
        """Number of input nodes.

        Returns
        -------
        number_of_input_nodes : int
            The number of input nodes in the graph.
        """
        return len(self._input_nodes)

    @property
    def number_of_output_nodes(self) -> int:
        """Number of output nodes.

        Returns
        -------
        number_of_output_nodes : int
            The number of output nodes in the graph.
        """
        return len(self._output_nodes)

    @property
    def non_input_nodes(self) -> list[int]:
        """List of non-input nodes, guaranteed duplicate-free.

        Returns
        -------
        non_input_nodes : list[int]
            The list of non-input nodes in the graph.
        """
        return self.get_nodes_complement(self.input_nodes)

    @property
    def non_output_nodes(self) -> list[int]:
        """List of non-output nodes, guaranteed duplicate-free.

        Returns
        -------
        non_output_nodes : list[int]
            The list of non-output nodes in the graph.
        """
        return self.get_nodes_complement(self.output_nodes)

    @property
    def output_nodes(self) -> list[int]:
        """List of output nodes, guaranteed duplicate-free.

        Returns
        -------
        output_nodes : list[int]
            The list of output nodes.
        """
        return list(self._output_nodes) # copy of the output nodes list to avoid external modifications

    @property
    def input_nodes(self) -> list[int]:
        """List of input nodes, guaranteed duplicate-free.

        Returns
        -------
        input_nodes : list[int]
            The list of input nodes.
        """
        return list(self._input_nodes) # copy of the input nodes list to avoid external modifications

    @property
    def nodes(self) -> list[int]:
        """List of all nodes in the graph, guaranteed duplicate-free.

        Returns
        -------
        nodes : list[int]
            A list of all nodes in the graph.
        """
        return list(self._nodes) # copy of the nodes list to avoid external modifications

    def get_correction_matrix(self, rows_id: list[int], cols_id: list[int]) -> np.ndarray:
        """Return the correction matrix for the given rows and columns.

        Parameters
        ----------
        rows_id : list[int]
            Row node indices.
        cols_id : list[int]
            Column node indices.

        Returns
        -------
        correction_matrix : np.ndarray
            The correction matrix with shape ``(len(rows_id), len(cols_id))``.
        """
        return self._adjacency_matrix[np.ix_(rows_id, cols_id)]

    def get_nodes_complement(self, nodes: Iterable[int]) -> list[int]:
        """Return all nodes not in the given collection.

        Parameters
        ----------
        nodes : Iterable[int]
            The nodes to exclude.

        Returns
        -------
        complement : list[int]
            The list of nodes that are not in the given collection.
        """
        return list(self._nodes - set(nodes)) # copy of the nodes list to avoid external modifications


    def __visualise__(self):
        """Render the graph with Matplotlib.

        Nodes are coloured by role (input, output, interior, or both).
        Edges are labelled with their weights.
        """
        pos = nx.spring_layout(self._graph, seed=42)

        node_colors = []
        for node in self._graph.nodes:
            if node in self.input_nodes and node in self.output_nodes:
                node_colors.append("#a855f7")  # purple: both input and output
            elif node in self.input_nodes:
                node_colors.append("#22c55e")  # green: input
            elif node in self.output_nodes:
                node_colors.append("#ef4444")  # red: output
            else:
                node_colors.append("#3b82f6")  # blue: interior (measured)

        edge_labels = {(u, v): f"{self._graph[u][v].get('weight', 1.0):.2f}" for u, v in self._graph.edges}

        nx.draw_networkx_nodes(self._graph, pos, node_color=node_colors, node_size=600)
        nx.draw_networkx_labels(self._graph, pos, font_color="white", font_weight="bold")
        nx.draw_networkx_edges(self._graph, pos, width=2.0, alpha=0.7)
        nx.draw_networkx_edge_labels(self._graph, pos, edge_labels=edge_labels, font_size=8)

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
        """Save the graph visualisation as an SVG file.

        Parameters
        ----------
        filename : str
            Path of the output SVG file.
        """
        self.__visualise__()
        plt.savefig(filename, format="svg")
        plt.close()

    def visualise(self):
        """Display the graph visualisation in a Matplotlib window."""
        self.__visualise__()
        plt.show()

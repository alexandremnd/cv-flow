from typing import Iterable

import numpy as np
from .graph import OpenGraph

def check_flow(graph: OpenGraph, measurements: Iterable[int]) -> tuple[bool, list[np.ndarray]]:
    """Check if an OpenGraph satisfies the flow property for a given sequence of measurements.

    Args:
        graph (OpenGraph):
            An OpenGraph object representing the graph structure, including its nodes, edges, input nodes, and output nodes.

        measurements (Iterable[int]):
            An ordered sequence of node that will be "measured".
            measurements[0] is the first node to be measured, measurements[1] the second, and so on.

    Returns:
        bool: True if the measurement sequence satisfies the flow property, False otherwise.
    """
    if not all(node in graph.nodes for node in measurements):
        raise ValueError("Each node measured must be present in the graph.")

    past_nodes = []
    corrections = []

    for node in measurements:
        past_nodes.append(node)
        future_nodes = graph.get_nodes_complement(past_nodes + graph.input_nodes)

        correction_matrix = graph.get_adjacency_matrix(rows_id=past_nodes, cols_id=future_nodes)
        target_vector = np.zeros(correction_matrix.shape[0])
        target_vector[-1] = 1.0

        augmented_correction_matrix = np.hstack((correction_matrix, target_vector.reshape(-1, 1)))
        rank_augmented = np.linalg.matrix_rank(augmented_correction_matrix)
        c, _, rank_correction, _ = np.linalg.lstsq(correction_matrix, target_vector)

        if rank_augmented != rank_correction:
            return False, []

        corrections.append(c)


    return True, corrections

def find_cvflow(graph: OpenGraph, verbose: bool = False) -> tuple[bool, dict[int, np.ndarray], dict[int, int]]:
    """Finds the CVFlow for the given OpenGraph.

    Args:
        graph (OpenGraph):
            The OpenGraph to find the CVFlow for.
        verbose (bool, optional):
            Whether to print verbose output. Defaults to False.

    Returns:
        g: dict[int, np.ndarray]: A dictionary mapping each node to the required correction to achieve the CV-flow.
        layer: dict[int, int]: A dictionary mapping each node to its layer in the CV-flow. The lowest layer corresponding to the output nodes.
    """

    layer: dict[int, int] = {node: 0 for node in graph.output_nodes}
    g: dict[int, np.ndarray] = {node: np.zeros(graph.number_of_nodes) for node in graph.nodes}

    measured_nodes: list[int] = []
    non_correctable_nodes: set[int] = set(graph.output_nodes)

    has_converged: bool = False
    iteration: int = 1
    while not has_converged:
        has_converged: bool = True

        current_corrected_nodes: list[int] = []
        past_nodes = measured_nodes.copy()
        past_nodes.append(-1)

        for node in graph.nodes:
            if node in non_correctable_nodes:
                continue

            past_nodes[-1] = node
            futures_nodes = graph.get_nodes_complement(past_nodes + graph.input_nodes)

            correction_matrix = graph.get_adjacency_matrix(rows_id=past_nodes, cols_id=futures_nodes)
            target_vector = np.zeros(correction_matrix.shape[0])
            target_vector[-1] = 1.0

            augmented_correction_matrix = np.hstack((correction_matrix, target_vector.reshape(-1, 1)))
            rank_augmented = np.linalg.matrix_rank(augmented_correction_matrix)
            correction_vector, _, rank_correction, _ = np.linalg.lstsq(correction_matrix, target_vector)

            if rank_augmented == rank_correction:
                for fnode in futures_nodes:
                    g[node][fnode] = correction_vector[futures_nodes.index(fnode)]

                layer[node] = iteration
                current_corrected_nodes.append(node)

        if current_corrected_nodes == []:
            if non_correctable_nodes == set(graph.nodes):
                return True, g, layer
            else:
                return False, g, layer
        else:
            has_converged = False
            non_correctable_nodes.update(current_corrected_nodes)
            measured_nodes += current_corrected_nodes
            iteration += 1

    return False, g, layer
import numpy as np

from cvflow.graph import OpenGraph

# TODO: The current implementation of check_flow and find_cvflow
# uses least squares and thus minimises the L2 norm of the correction vector
# which spread the correction onto many nodes instead of concentrating it on a few nodes.
# A better solution would be to use a L1 minimisation from scipy.optimize.linprog (or even offer a choice of the norm L1/L2)

def create_g_entry(correction_vector: np.ndarray, nodes: list[int]) -> dict[int, float]:
    """Build a sparse g-map entry from a correction vector.

    Zips each node in `nodes` with the corresponding value in `correction_vector`
    and keeps only the pairs whose absolute value exceeds 1e-6, discarding
    numerically negligible corrections.

    Parameters
    ----------
    correction_vector : np.ndarray
        Solution vector returned by ``np.linalg.lstsq``. Each element is
        the correction weight assigned to the node at the same position in
        `nodes`.
    nodes : list[int]
        Ordered list of node IDs corresponding to the entries of
        `correction_vector`.

    Returns
    -------
    g_entry : dict[int, float]
        Mapping from node ID to its non-negligible correction weight.
        Entries whose absolute correction is ≤ 1e-6 are omitted.
    """
    return {node: float(correction) for node, correction in zip(nodes, correction_vector) if abs(correction) > 1e-6}

def check_flow(graph: OpenGraph, measurements: list[int]) -> tuple[bool, dict[int, float], dict[int, int]]:
    """Check if an OpenGraph satisfies the flow property for a given sequence of measurements.

    Parameters
    ----------
    graph : OpenGraph
        The graph structure, including its nodes, edges, input nodes, and output nodes.
    measurements : Iterable[int]
        An ordered sequence of nodes to be measured.
        ``measurements[0]`` is the first node to be measured,
        ``measurements[1]`` the second, and so on.

    Returns
    -------
    has_flow : bool
        True if the measurement sequence satisfies the flow property, False otherwise.
    g : dict[int, float]
        A dictionary mapping each measured node to its corresponding correction value on the graph.
    layer : dict[int, int]
        A dictionary mapping each measured node to its layer in the flow.
        The lowest layer corresponds to the output nodes.
    """
    if not all(node in graph.nodes for node in measurements):
        raise ValueError("Each node measured must be present in the graph.")

    past_nodes = []
    g = {}
    layer = {node: 0 for node in graph.output_nodes}

    for i, node in enumerate(measurements):
        past_nodes.append(node)
        future_nodes = graph.get_nodes_complement(past_nodes + graph.input_nodes)

        correction_matrix = graph.get_adjacency_matrix(rows_id=past_nodes, cols_id=future_nodes)
        target_vector = np.zeros(correction_matrix.shape[0])
        target_vector[-1] = 1.0

        augmented_correction_matrix = np.hstack((correction_matrix, target_vector.reshape(-1, 1)))
        rank_augmented = np.linalg.matrix_rank(augmented_correction_matrix)
        correction_vector, _, rank_correction, _ = np.linalg.lstsq(correction_matrix, target_vector)

        if rank_augmented != rank_correction:
            return False, {}, {}

        g[node] = create_g_entry(correction_vector, future_nodes)
        layer[node] = len(measurements) - i

    return True, g, layer

def find_cvflow(graph: OpenGraph) -> tuple[bool, dict[int, dict[int, float]], dict[int, int]]:
    """Find the CV-flow of the given OpenGraph.

    Parameters
    ----------
    graph : OpenGraph
        The graph to find the CV-flow for.

    Returns
    -------
    has_flow : bool
        True if the CV-flow exists, False otherwise.
    g : dict[int, dict[int, float]]
        A dictionary mapping each node to the required correction to achieve the CV-flow.
    layer : dict[int, int]
        A dictionary mapping each node to its layer in the CV-flow.
        The lowest layer corresponds to the output nodes.
    """

    layer: dict[int, int] = {node: 0 for node in graph.output_nodes}
    g: dict[int, dict[int, float]] = {}

    past_nodes: list[int] = graph.non_output_nodes
    resolved_nodes: list[int] = graph.output_nodes

    iteration: int = 1
    while True:
        nodes_in_layer: list[int] = []

        correction_matrix = graph.get_adjacency_matrix(rows_id=past_nodes, cols_id=resolved_nodes)
        augmented_correction_matrix = np.hstack((correction_matrix, np.zeros((correction_matrix.shape[0], 1))))

        for node in graph.nodes:
            if node in resolved_nodes:
                continue

            node_index = past_nodes.index(node)
            augmented_correction_matrix[node_index, -1] = 1.0

            rank_augmented = np.linalg.matrix_rank(augmented_correction_matrix)
            correction_vector, _, rank_correction, _ = np.linalg.lstsq(correction_matrix, augmented_correction_matrix[:, -1])

            if rank_augmented == rank_correction:
                g[node] = create_g_entry(correction_vector, resolved_nodes)
                layer[node] = iteration
                nodes_in_layer.append(node)

            # We reset the augmented matrix RHS
            augmented_correction_matrix[node_index, -1] = 0.0

        if not nodes_in_layer :
            if set(resolved_nodes) == set(graph.nodes):
                return True, g, layer
            else:
                return False, {}, {}
        else:
            resolved_nodes.extend(nodes_in_layer)

            # Remove the nodes that have been corrected in this iteration
            # They now become part of the output nodes
            for node in nodes_in_layer:
                past_nodes.remove(node)

            if iteration > graph.number_of_nodes + 1:
                raise RuntimeError("Exceeded maximum iterations while finding CVFlow. A flow should have been found but was not.")

            iteration += 1
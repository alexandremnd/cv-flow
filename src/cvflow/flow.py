import numpy as np

from cvflow.graph import OpenGraph

def solve_l1(correction_matrix: np.ndarray, target_vector: np.ndarray) -> tuple[int, np.ndarray]:
    """Solve the L1 minimisation problem for the given correction matrix and target vector.

    Parameters
    ----------
    correction_matrix : np.ndarray
        The matrix representing the corrections.
    target_vector : np.ndarray
        The target vector for the corrections.

    Returns
    -------
    rank : int
        The rank of the correction matrix.
    solution : np.ndarray
        The solution vector that minimizes the L1 norm of the corrections.
    """
    from scipy.optimize import linprog

    num_vars = correction_matrix.shape[1]
    c = np.ones(2 * num_vars)
    A_eq = np.hstack((correction_matrix, -correction_matrix))
    bounds = [(0, None)] * (2 * num_vars)

    res = linprog(c, A_eq=A_eq, b_eq=target_vector, bounds=bounds, method='highs')
    rank = np.linalg.matrix_rank(correction_matrix)

    if res.success:
        return rank, res.x[:num_vars] - res.x[num_vars:]  # Return the difference of positive and negative parts
    else:
        raise ValueError("L1 minimization failed: " + res.message)

def solve_l2(correction_matrix: np.ndarray, target_vector: np.ndarray) -> tuple[int, np.ndarray]:
    """Solve the L2 minimisation problem for the given correction matrix and target vector.

    Parameters
    ----------
    correction_matrix : np.ndarray
        The matrix representing the corrections.
    target_vector : np.ndarray
        The target vector for the corrections.

    Returns
    -------
    rank : int
        The rank of the correction matrix.
    solution : np.ndarray
        The solution vector that minimizes the L2 norm of the corrections.
    """
    solution_vector, _, rank, _ = np.linalg.lstsq(correction_matrix, target_vector, rcond=None)
    return int(rank), solution_vector

def create_g_entry(correction_vector: np.ndarray, nodes: list[int]) -> dict[int, float]:
    """Build a sparse g-map entry from a correction vector.

    Zips each node in `nodes` with the corresponding value in `correction_vector`
    and keeps only the pairs whose absolute value exceeds 1e-6, discarding
    numerically negligible corrections.

    Parameters
    ----------
    correction_vector : np.ndarray
        Solution vector. Each element is the correction weight
        assigned to the node at the same position in `nodes`.
        Ordered list of node IDs corresponding to the entries of
        `correction_vector`.

    Returns
    -------
    g_entry : dict[int, float]
        Mapping from node ID to its non-negligible correction weight.
        Entries whose absolute correction is ≤ 1e-6 are omitted.
    """
    return {node: float(correction) for node, correction in zip(nodes, correction_vector) if abs(correction) > 1e-6}

def check_flow(graph: OpenGraph, measurements: list[int], method: str = "l2") -> tuple[bool, dict[int, float], dict[int, int]]:
    """Check if an OpenGraph satisfies the flow property for a given sequence of measurements.

    Parameters
    ----------
    graph : OpenGraph
        The graph structure, including its nodes, edges, input nodes, and output nodes.
    measurements : list[int]
        An ordered sequence of nodes to be measured.
        ``measurements[0]`` is the first node to be measured,
        ``measurements[1]`` the second, and so on.
    method : str, optional
        The method used to solve the correction vector. Can be "l1" or "l2".
            - "l2" (default) uses least squares minimisation, which may spread
            the correction across many nodes.

            - "l1" uses linear programming to minimise the L1 norm of the
            correction vector, which may yield sparser corrections.

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

        correction_matrix = graph.get_correction_matrix(rows_id=past_nodes, cols_id=future_nodes)
        target_vector = np.zeros(correction_matrix.shape[0])
        target_vector[-1] = 1.0

        augmented_correction_matrix = np.hstack((correction_matrix, target_vector.reshape(-1, 1)))
        rank_augmented = np.linalg.matrix_rank(augmented_correction_matrix)

        if method == "l2":
            rank_correction, correction_vector = solve_l2(correction_matrix, target_vector)
        else:
            rank_correction, correction_vector = solve_l1(correction_matrix, target_vector)

        if rank_augmented != rank_correction:
            return False, {}, {}

        g[node] = create_g_entry(correction_vector, future_nodes)
        layer[node] = len(measurements) - i

    return True, g, layer

def find_cvflow(graph: OpenGraph, method: str = "l2") -> tuple[bool, dict[int, dict[int, float]], dict[int, int]]:
    """Find the CV-flow of the given OpenGraph.

    Parameters
    ----------
    graph : OpenGraph
        The graph to find the CV-flow for.
    method : str, optional
        The method used to solve the correction vector. Can be "l1" or "l2".
            - "l2" (default) uses least squares minimisation, which may spread
            the correction across many nodes.

            - "l1" uses linear programming to minimise the L1 norm of the
            correction vector, which may yield sparser corrections.

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

        correction_matrix = graph.get_correction_matrix(rows_id=past_nodes, cols_id=resolved_nodes)
        augmented_correction_matrix = np.hstack((correction_matrix, np.zeros((correction_matrix.shape[0], 1))))

        for node in graph.nodes:
            if node in resolved_nodes:
                continue

            node_index = past_nodes.index(node)
            augmented_correction_matrix[node_index, -1] = 1.0

            rank_augmented = np.linalg.matrix_rank(augmented_correction_matrix)
            if method == "l2":
                rank_correction, correction_vector = solve_l2(correction_matrix, augmented_correction_matrix[:, -1])
            else:
                rank_correction, correction_vector = solve_l1(correction_matrix, augmented_correction_matrix[:, -1])

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
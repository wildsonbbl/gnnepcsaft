import numpy as np
import jraph
import jax.numpy as jnp
import jax


def _nearest_bigger_power_of_two(x: int) -> int:
    """Computes the nearest power of two greater than x for padding."""
    y = 2
    while y < x:
        y *= 2
    return y


def pad_graph_to_nearest_power_of_two(
    graphs_tuple: jraph.GraphsTuple,
) -> jraph.GraphsTuple:
    """Pads a batched `GraphsTuple` to the nearest power of two.

    For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
    would pad the `GraphsTuple` nodes and edges:
        7batch_sizedes --> 8 nodes (2^3)
        5 edges --> 8 edges (2^3)

    And since padding is accomplished using `jraph.pad_with_graphs`, an extra
    graph and node is added:
        8 nodes --> 9 nodes
        3 graphs --> 4 graphs

    Args:
        graphs_tuple: a batched `GraphsTuple` (can be batch size 1).

    Returns:
        A graphs_tuple batched to the nearest power of two.
    """
    # Add 1 since we need at least one padding node for pad_with_graphs.
    pad_nodes_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_node)) + 1
    pad_edges_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_edge))
    # Add 1 since we need at least one padding graph for pad_with_graphs.
    # We do not pad to nearest power of two because the batch size is fixed.
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
    return jraph.pad_with_graphs(
        graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to
    )


def get_batched_padded_graph_tuples(batch) -> jraph.GraphsTuple:
    graphs = jraph.GraphsTuple(
        nodes=np.array(batch.node_attr),
        edges=np.array(batch.edges),
        n_node=np.array(batch.n_node),
        n_edge=np.array(batch.n_edge),
        senders=np.array(batch.senders),
        receivers=np.array(batch.receivers),
        globals=np.array(batch.globals),
    )

    graphs = pad_graph_to_nearest_power_of_two(graphs)  # padd the whole batch once
    return graphs


def get_padded_array(
    arrays: list, subkey: jax.random.KeyArray, max_pad: int
) -> tuple[jnp.ndarray, str]:
    (ids, _, _) = arrays[0]
    states = [
        jnp.concatenate(
            [jnp.asarray(state, dtype=jnp.float_), jnp.asarray([y], dtype=jnp.float_)]
        )[None, ...]
        for _, state, y in arrays
    ]

    states = jnp.concatenate(states, 0)
    states = jax.random.permutation(subkey, states, 0, True)

    pad_size = _nearest_bigger_power_of_two(states.shape[0])

    states = states.repeat(pad_size // states.shape[0] + 1, 0)
    return states[:pad_size,:]

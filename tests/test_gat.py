"""Attention math, target extraction, graph immutability, and source parity."""

import dgl
import pytest
import torch
from torch.nn import functional as F

from src.models import GATEncoder, GATLayer


def test_sigmoid_sum_and_edge_direction_by_hand():
    # Node 0 receives 0 and 1; node 1 receives only itself. No reverse 0 -> 1.
    graph = dgl.graph(([0, 1, 1], [0, 0, 1]), num_nodes=2)
    layer = GATLayer(1, 1)
    with torch.no_grad():
        layer.projection.weight.fill_(1)
        layer.attention.weight.zero_()
    output = layer(torch.tensor([[2.0], [4.0]]), graph)
    # Sigmoid(0) = 0.5 per incoming edge; softmax would give node 1 a weight of 1.
    torch.testing.assert_close(output, torch.tensor([[3.0], [2.0]]))


def test_source_destination_order_leaky_relu_and_shared_projection():
    graph = dgl.graph(([0, 1, 1], [0, 0, 1]), num_nodes=2)
    layer = GATLayer(1, 1)
    with torch.no_grad():
        layer.projection.weight.fill_(2)
        layer.attention.weight.copy_(torch.tensor([[1.0, -2.0]]))
    output = layer(torch.tensor([[1.0], [3.0]]), graph)
    expected = torch.tensor([
        [2 * torch.sigmoid(torch.tensor(-0.02)) + 6 * torch.sigmoid(torch.tensor(2.0))],
        [6 * torch.sigmoid(torch.tensor(-0.06))],
    ])
    torch.testing.assert_close(output, expected)


def test_gat_shapes_target_selection_and_no_mutation(graph_batch):
    model = GATEncoder().eval()
    graph_batch.ndata["z"] = torch.randn(8, 2)
    graph_batch.ndata["h"] = torch.randn(8, 3)
    graph_batch.edata["e"] = torch.randn(graph_batch.num_edges(), 2)
    original_nodes = {key: value.clone() for key, value in graph_batch.ndata.items()}
    original_edges = {key: value.clone() for key, value in graph_batch.edata.items()}
    all_embeddings = model.attention_layer(graph_batch.ndata["feat"], graph_batch)
    assert graph_batch.ndata["feat"].shape == (8, 13)
    assert all_embeddings.shape == (8, 64)
    selected = all_embeddings[graph_batch.ndata["is_target"] == 1]
    expected = F.relu(model.output_batch_norm(
        model.target_projection(F.relu(model.target_batch_norm(selected)))
    ))
    actual = model(graph_batch)
    assert actual.shape == (3, 32)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model(graph_batch), actual)
    assert set(graph_batch.ndata) == set(original_nodes)
    assert set(graph_batch.edata) == set(original_edges)
    for key, value in original_nodes.items():
        torch.testing.assert_close(graph_batch.ndata[key], value)
    for key, value in original_edges.items():
        torch.testing.assert_close(graph_batch.edata[key], value)


@pytest.mark.parametrize("training", [False, True])
def test_gat_source_parity(upstream, weight_transfer, graph_batch, training):
    original = upstream.GAT_L().train(training)
    clean = GATEncoder().train(training)
    weight_transfer.gat(clean, original)
    # Upstream overwrites 'feat'; local_scope isolates that side effect.
    with graph_batch.local_scope():
        graph_batch.ndata["id"] = graph_batch.ndata["is_target"]
        expected = original(graph_batch.ndata["feat"], graph_batch)
    actual = clean(graph_batch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_bad_node_feature_shape(graph_batch):
    graph_batch.ndata["feat"] = torch.randn(8, 12)
    with pytest.raises(ValueError, match="node features must have shape"):
        GATEncoder()(graph_batch)


@pytest.mark.parametrize("mask", [
    [1, 1, 0, 0, 0, 0, 0, 1],  # Total is B, but two in one graph and none in another.
    [0, 0, 0, 0, 0, 1, 0, 1],
    [0, 2, 0, 0, 0, 1, 0, 1],
])
def test_bad_target_mask(graph_batch, mask):
    graph_batch.ndata["is_target"] = torch.tensor(mask)
    with pytest.raises(ValueError, match="target"):
        GATEncoder()(graph_batch)


def test_missing_and_wrong_rank_target_masks(graph_batch):
    graph_batch.ndata["is_target"] = torch.zeros(8, 1)
    with pytest.raises(ValueError, match=r"shape \[N\]"):
        GATEncoder()(graph_batch)
    del graph_batch.ndata["is_target"]
    with pytest.raises(ValueError, match="must contain"):
        GATEncoder()(graph_batch)


def test_single_self_loop_graph_in_eval():
    graph = dgl.graph(([0], [0]), num_nodes=1)
    graph.ndata["feat"] = torch.randn(1, 13)
    graph.ndata["is_target"] = torch.ones(1)
    model = GATEncoder()
    with pytest.raises(ValueError, match="BatchNorm"):
        model(graph)
    assert model.eval()(graph).shape == (1, 32)

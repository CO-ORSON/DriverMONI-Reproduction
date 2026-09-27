"""Active fusion architecture, source parity, gradients, and device smoke tests."""

import pytest
import torch
from torch.nn import functional as F

from src.models import DriverMONI


def test_fusion_shapes_order_and_raw_logits(graph_batch):
    model = DriverMONI().eval()
    features = torch.randn(3, 12)
    combined = model.encode(features, graph_batch)
    assert combined.shape == (3, 96)
    torch.testing.assert_close(combined[:, :64], model.dnn(features))
    expected_gat = F.relu(model.gat_batch_norm(model.gat_projection(model.gat(graph_batch))))
    torch.testing.assert_close(combined[:, 64:], expected_gat)
    logits = model(features, graph_batch)
    assert logits.shape == (3, 2)
    torch.testing.assert_close(logits, model.classifier(combined))
    # Set a known classifier output to distinguish logits from probabilities.
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([-2.0, 3.0]))
    torch.testing.assert_close(model(features, graph_batch), torch.tensor([[-2.0, 3.0]] * 3))


@pytest.mark.parametrize("training", [False, True])
def test_full_model_source_logits_loss_and_gradient_parity(
    upstream, weight_transfer, graph_batch, training,
):
    original = upstream.gcn_linc_embedder_transfer(upstream.GAT_L(), "BRCA").train(training)
    clean = DriverMONI().train(training)
    weight_transfer.model(clean, original)
    features = torch.randn(3, 12)
    original_features = features.clone().requires_grad_(True)
    clean_features = features.clone().requires_grad_(True)
    original_nodes = graph_batch.ndata["feat"].clone().requires_grad_(True)
    clean_nodes = original_nodes.detach().clone().requires_grad_(True)
    torch.manual_seed(25)
    with graph_batch.local_scope():
        graph_batch.ndata["feat"] = original_nodes
        graph_batch.ndata["id"] = graph_batch.ndata["is_target"]
        expected = original(original_features, graph_batch)
    torch.manual_seed(25)
    with graph_batch.local_scope():
        graph_batch.ndata["feat"] = clean_nodes
        actual = clean(clean_features, graph_batch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    labels = torch.tensor([0, 1, 0])
    weights = torch.tensor([0.75, 1.5], dtype=torch.float64)
    expected_loss = F.nll_loss(F.log_softmax(expected, dim=1, dtype=torch.float64), labels, weight=weights)
    actual_loss = F.nll_loss(F.log_softmax(actual, dim=1, dtype=torch.float64), labels, weight=weights)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    expected_loss.backward()
    actual_loss.backward()
    torch.testing.assert_close(clean_features.grad, original_features.grad)
    torch.testing.assert_close(clean_nodes.grad, original_nodes.grad)

    mapping = {
        "dnn.linear1": "feat_enc.ll", "dnn.linear2": "feat_enc.ll2", "dnn.linear3": "feat_enc.ll3",
        "dnn.batch_norm1": "feat_enc.bn1", "dnn.batch_norm2": "feat_enc.bn2", "dnn.batch_norm3": "feat_enc.bn3",
        "gat.attention_layer.projection": "gcn_encoder.GCN.fc",
        "gat.attention_layer.attention": "gcn_encoder.GCN.attn_fc",
        "gat.target_batch_norm": "gcn_encoder.bn", "gat.target_projection": "gcn_encoder.ll",
        "gat.output_batch_norm": "gcn_encoder.bn1", "gat_projection": "gat_ll",
        "gat_batch_norm": "bn", "classifier": "l_classify",
    }
    original_parameters = dict(original.named_parameters())
    for name, parameter in clean.named_parameters():
        module, suffix = name.rsplit(".", 1)
        reference = original_parameters[f"{mapping[module]}.{suffix}"]
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, reference.grad)
    assert original.gcn_encoder.ll2.weight.grad is None  # Dead upstream layer.


def test_batch_and_feature_validation(graph_batch):
    model = DriverMONI()
    with pytest.raises(ValueError, match="number of graphs"):
        model(torch.randn(2, 12), graph_batch)
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(3, 13), graph_batch)


def test_fixed_seed_reproduces_clean_model_initialization():
    torch.manual_seed(9)
    first = DriverMONI()
    torch.manual_seed(9)
    second = DriverMONI()
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable on this host")
def test_cuda_forward_and_backward(graph_batch):
    model = DriverMONI().to("cuda")
    graph = graph_batch.to("cuda")  # Requires a CUDA-enabled DGL installation.
    logits = model(torch.randn(3, 12, device="cuda"), graph)
    assert logits.shape == (3, 2)
    assert logits.device.type == "cuda"
    F.cross_entropy(logits, torch.tensor([0, 1, 0], device="cuda")).backward()
    assert model.gat.attention_layer.attention.weight.grad is not None

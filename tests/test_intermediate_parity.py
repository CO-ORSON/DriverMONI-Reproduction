"""Trace intermediate values and compare one Adam update under matched conditions."""

import pytest
import torch
from torch.nn import functional as F

from src.models import DriverMONI
from src.train import weighted_nll


MODULES = {
    "dnn.linear1": "feat_enc.ll", "dnn.linear2": "feat_enc.ll2", "dnn.linear3": "feat_enc.ll3",
    "dnn.batch_norm1": "feat_enc.bn1", "dnn.batch_norm2": "feat_enc.bn2", "dnn.batch_norm3": "feat_enc.bn3",
    "dnn.relu": "feat_enc.relu", "dnn.dropout1": "feat_enc.dr1", "dnn.dropout2": "feat_enc.dr2",
    "gat.attention_layer.projection": "gcn_encoder.GCN.fc",
    "gat.attention_layer.attention": "gcn_encoder.GCN.attn_fc",
    "gat.attention_layer": "gcn_encoder.GCN",
    "gat.target_batch_norm": "gcn_encoder.bn", "gat.target_projection": "gcn_encoder.ll",
    "gat.output_batch_norm": "gcn_encoder.bn1", "gat.relu": "gcn_encoder.relu",
    "gat_projection": "gat_ll", "gat_batch_norm": "bn", "relu": "relu", "classifier": "l_classify",
}


def record(model, mapping):
    trace, handles = [], []
    for name, module_name in mapping.items():
        def hook(module, inputs, output, name=name):
            trace.append((name, inputs[0].detach().clone(), output.detach().clone()))
        handles.append(model.get_submodule(module_name).register_forward_hook(hook))
    return trace, handles


@pytest.mark.parametrize("training", [False, True])
def test_intermediate_tensors_and_adam_step(upstream, weight_transfer, graph_batch, training):
    original = upstream.gcn_linc_embedder_transfer(upstream.GAT_L(), "BRCA").train(training)
    clean = DriverMONI().train(training)
    weight_transfer.model(clean, original)
    original_trace, original_handles = record(original, MODULES)
    clean_trace, clean_handles = record(clean, {key: key for key in MODULES})
    features = torch.randn(3, 12)
    original_optimizer = torch.optim.Adam(original.parameters(), lr=0.001)
    clean_optimizer = torch.optim.Adam(clean.parameters(), lr=0.001)
    try:
        torch.manual_seed(314)
        with graph_batch.local_scope():
            graph_batch.ndata["id"] = graph_batch.ndata["is_target"]
            expected = original(features, graph_batch)
            original_edges = graph_batch.edata["e"].detach().clone()
        torch.manual_seed(314)
        actual = clean(features, graph_batch)
    finally:
        for handle in original_handles + clean_handles:
            handle.remove()
    assert [item[0] for item in clean_trace] == [item[0] for item in original_trace]
    for (name, inputs, output), (_, ref_inputs, ref_output) in zip(clean_trace, original_trace):
        assert inputs.shape == ref_inputs.shape, name
        assert output.shape == ref_output.shape, name
        torch.testing.assert_close(inputs, ref_inputs, rtol=0, atol=0, msg=name)
        torch.testing.assert_close(output, ref_output, rtol=0, atol=0, msg=name)
    captured = {name: (inputs, output) for name, inputs, output in clean_trace}
    assert captured["dnn.linear1"][0].shape == (3, 12)
    assert captured["dnn.linear3"][1].shape == (3, 64)
    assert captured["gat.attention_layer"][0].shape == (8, 13)
    assert captured["gat.attention_layer"][1].shape == (8, 64)
    target_input = captured["gat.target_batch_norm"][0]
    assert target_input.shape == (3, 64)
    torch.testing.assert_close(target_input,
        captured["gat.attention_layer"][1][graph_batch.ndata["is_target"] == 1], rtol=0, atol=0)
    assert captured["gat.output_batch_norm"][1].shape == (3, 32)
    assert captured["classifier"][0].shape == (3, 96)
    assert actual.shape == (3, 2)
    # Compare edge logits and independent sigmoid coefficients, before reduction.
    clean_edges = F.leaky_relu(captured["gat.attention_layer.attention"][1], negative_slope=0.01)
    torch.testing.assert_close(clean_edges, original_edges, rtol=0, atol=0)
    torch.testing.assert_close(torch.sigmoid(clean_edges), F.sigmoid(original_edges), rtol=0, atol=0)
    labels = torch.tensor([0, 1, 0])
    weights = torch.tensor([0.75, 1.5], dtype=torch.float64)
    expected_loss = F.nll_loss(F.log_softmax(expected, dim=1, dtype=float), labels, weight=weights)
    actual_loss = weighted_nll(actual, labels, weights)
    expected_loss.backward()
    actual_loss.backward()
    original_optimizer.step()
    clean_optimizer.step()
    original_parameters = dict(original.named_parameters())
    for name, parameter in clean.named_parameters():
        module, suffix = name.rsplit(".", 1)
        reference = original_parameters[f"{MODULES[module]}.{suffix}"]
        torch.testing.assert_close(parameter, reference, rtol=1e-5, atol=1e-6, msg=name)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(clean_optimizer.state[parameter][key],
                original_optimizer.state[reference][key], rtol=1e-5, atol=1e-6, msg=f"{name}.{key}")

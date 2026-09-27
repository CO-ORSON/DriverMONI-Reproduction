"""DNN shapes, input errors, initialization, and numerical source parity."""

import pytest
import torch

from src.models import DNNEncoder


@pytest.mark.parametrize("batch_size", [2, 5])
def test_dnn_shapes(batch_size):
    model = DNNEncoder()
    assert model(torch.randn(batch_size, 12)).shape == (batch_size, 64)
    assert (model.dropout1.p, model.dropout2.p) == (0.7, 0.6)


@pytest.mark.parametrize("training", [False, True])
def test_dnn_source_parity(upstream, weight_transfer, training):
    original = upstream.feat_L_BRCA().train(training)
    clean = DNNEncoder().train(training)
    weight_transfer.dnn(clean, original)
    x = torch.randn(5, 12)
    torch.manual_seed(91)
    expected = original(x)
    torch.manual_seed(91)
    actual = clean(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for new, old in [(clean.batch_norm1, original.bn1),
                     (clean.batch_norm2, original.bn2),
                     (clean.batch_norm3, original.bn3)]:
        torch.testing.assert_close(new.running_mean, old.running_mean)
        torch.testing.assert_close(new.running_var, old.running_var)


def test_dnn_initialization_matches_source(upstream):
    torch.manual_seed(7)
    original = upstream.feat_L_BRCA()
    torch.manual_seed(7)
    clean = DNNEncoder()
    for new, old in [(clean.linear1, original.ll), (clean.linear2, original.ll2),
                     (clean.linear3, original.ll3)]:
        torch.testing.assert_close(new.weight, old.weight, rtol=0, atol=0)
        torch.testing.assert_close(new.bias, old.bias, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(3, 13), (12,), (2, 3, 12)])
def test_dnn_rejects_wrong_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        DNNEncoder()(torch.randn(*shape))


def test_dnn_singleton_is_supported_only_in_eval():
    model = DNNEncoder()
    with pytest.raises(ValueError, match="BatchNorm"):
        model(torch.randn(1, 12))
    assert model.eval()(torch.randn(1, 12)).shape == (1, 64)


def test_dnn_rejects_empty_and_integer_features():
    with pytest.raises(ValueError, match="nonempty"):
        DNNEncoder()(torch.empty(0, 12))
    with pytest.raises(TypeError, match="floating"):
        DNNEncoder()(torch.ones(2, 12, dtype=torch.long))

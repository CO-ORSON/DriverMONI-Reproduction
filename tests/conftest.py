"""Small directed graph fixtures and explicit upstream weight mappings."""

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("DGLBACKEND", "pytorch")

import dgl
import pytest
import torch
from tests.upstream_reference import load_reference


def pytest_addoption(parser):
    parser.addoption("--brca-data-root", default=None, help="Original BRCA data root for integration comparisons")


@pytest.fixture(scope="session")
def upstream_source_root(request):
    value = request.config.getoption("--brca-data-root")
    if value is None or not (Path(value) / "CV_multimodal_compare.py").is_file():
        pytest.skip("Original-source parity requires --brca-data-root pointing to the pinned checkout")
    return Path(value)


@pytest.fixture(scope="session")
def upstream_dataset(upstream_source_root):
    return load_reference(upstream_source_root, dataset=True)


@pytest.fixture(autouse=True)
def reproducible_test_seed():
    torch.manual_seed(1729)


@pytest.fixture(scope="session")
def upstream(upstream_source_root):
    return load_reference(upstream_source_root)


@pytest.fixture
def graph_batch():
    """Three directed ego graphs, including an isolated self-loop target.

    Targets deliberately occur at different local indices to catch first-node
    assumptions. Every node has a self-loop, as in upstream graph construction.
    """
    graphs = []
    for size, target in [(3, 1), (4, 2), (1, 0)]:
        sources = list(range(size))
        destinations = list(range(size))
        if size > 1:
            sources += [0, size - 1, target]
            destinations += [target, target, 0]
        graph = dgl.graph((sources, destinations), num_nodes=size)
        graph.ndata["feat"] = torch.randn(size, 13)
        graph.ndata["is_target"] = torch.zeros(size)
        graph.ndata["is_target"][target] = 1
        graphs.append(graph)
    return dgl.batch(graphs)


def copy_dnn_weights(clean, original):
    for new, old in [
        ("linear1", "ll"), ("linear2", "ll2"), ("linear3", "ll3"),
        ("batch_norm1", "bn1"), ("batch_norm2", "bn2"), ("batch_norm3", "bn3"),
    ]:
        getattr(clean, new).load_state_dict(getattr(original, old).state_dict())


def copy_gat_weights(clean, original):
    clean.attention_layer.projection.load_state_dict(original.GCN.fc.state_dict())
    clean.attention_layer.attention.load_state_dict(original.GCN.attn_fc.state_dict())
    clean.target_batch_norm.load_state_dict(original.bn.state_dict())
    clean.target_projection.load_state_dict(original.ll.state_dict())
    clean.output_batch_norm.load_state_dict(original.bn1.state_dict())


@pytest.fixture
def weight_transfer():
    def transfer(clean, original):
        copy_dnn_weights(clean.dnn, original.feat_enc)
        copy_gat_weights(clean.gat, original.gcn_encoder)
        clean.gat_projection.load_state_dict(original.gat_ll.state_dict())
        clean.gat_batch_norm.load_state_dict(original.bn.state_dict())
        clean.classifier.load_state_dict(original.l_classify.state_dict())
    return SimpleNamespace(dnn=copy_dnn_weights, gat=copy_gat_weights, model=transfer)

"""Verify reference provenance and execute selected expressions from actual source.

Requires --brca-data-root pointing to the pinned authors' checkout. No upstream
module is imported: its CLI/data-loading side effects are never executed.
"""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.model_selection import StratifiedKFold
from torch.nn import functional as F

from src.dataset import DNN_FEATURES, GAT_FEATURES
from src.train import class_weights, weighted_nll
from src.utils import load_config
from tests.upstream_reference import adapted_definition, verified_source


@pytest.fixture(scope="module")
def source_root(request):
    value = request.config.getoption("--brca-data-root")
    if value is None or not (Path(value) / "CV_multimodal_compare.py").is_file():
        pytest.skip("Source audit requires --brca-data-root with original Python files")
    return Path(value)


def definition(text, name):
    return next(node for node in ast.parse(text).body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)


@pytest.mark.parametrize("name,filename", [
    ("feat_L_BRCA", "torch_models_linear.py"),
    ("GATLayer", "CV_multimodal_compare.py"),
    ("GAT_L", "CV_multimodal_compare.py"),
    ("gcn_linc_embedder_transfer", "CV_multimodal_compare.py"),
    ("classify_graph_data_encoder_humannet", "CV_multimodal_compare.py"),
])
def test_reference_classes_match_actual_source(source_root, name, filename):
    original = ast.unparse(definition(verified_source(source_root, filename), name))
    if name == "gcn_linc_embedder_transfer":
        old = "graph.ndata['feat'].type(torch.FloatTensor).cuda()"
        new = "graph.ndata['feat'].to(dtype=torch.float32, device=feat.device)"
        assert original.count(old) == 1
        original = original.replace(old, new)
    if name == "classify_graph_data_encoder_humannet":
        adaptations = {
            "nodes_l = list(set(node_l))": "nodes_l = IndexedNodes(set(node_l))",
            "g_dgl = dgl.DGLGraph()\n    g_dgl.from_networkx(g_nx)":
                "g_dgl = dgl.from_networkx(g_nx)",
            "dgl.DGLGraph(self.graph.subgraph(new_nodes))":
                "dgl.node_subgraph(self.graph, new_nodes)",
        }
        # Class-level unparse indents method bodies by eight spaces.
        adaptations = {key.replace("\n    ", "\n        "): value
                       for key, value in adaptations.items()}
        for old, new in adaptations.items():
            assert original.count(old) == 1, old
            original = original.replace(old, new)
    reference = adapted_definition(source_root, filename, name)
    assert ast.dump(ast.parse(original).body[0]) == ast.dump(reference)


def test_feature_columns_read_directly_from_source(source_root):
    dataset = definition(verified_source(source_root, "CV_multimodal_compare.py"),
                         "classify_graph_data_encoder_humannet")
    columns = [tuple(ast.literal_eval(node)) for node in ast.walk(dataset)
               if isinstance(node, ast.List) and node.elts
               and all(isinstance(item, ast.Constant) and isinstance(item.value, str)
                       for item in node.elts)]
    assert DNN_FEATURES in columns and GAT_FEATURES in columns
    assert len(DNN_FEATURES) == 12 and len(GAT_FEATURES) == 13


def expression(function, variable, environment):
    matches = [node.value for node in ast.walk(function) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == variable
                       for target in node.targets)]
    assert len(matches) == 1, variable
    return eval(compile(ast.Expression(matches[0]), "<upstream-expression>", "eval"), environment)


def test_actual_source_weights_loss_adam_and_cv(source_root):
    function = definition(verified_source(source_root, "CV_multimodal_compare.py"), "train_data")
    config = load_config(Path(__file__).parents[1] / "configs/brca.yaml")
    labels = np.array([0] * 18 + [1] * 6)
    env = {"neg_num": 18, "pos_num": 6, "total__": 24}
    expected_weights = torch.tensor([expression(function, name, env)
                                     for name in ("weight_for_0", "weight_for_1")], dtype=float)
    torch.testing.assert_close(class_weights(labels), expected_weights, rtol=0, atol=0)
    logits = torch.tensor([[1.3, -0.4], [-0.2, 0.8], [0.1, 0.3]])
    batch_labels = torch.tensor([0, 1, 0])
    expected_loss = expression(function, "loss_contrastive", {
        "torch": torch, "F": F, "output": logits, "label": batch_labels, "classw_w": expected_weights,
    })
    torch.testing.assert_close(weighted_nll(logits, batch_labels, expected_weights), expected_loss,
                               rtol=0, atol=0)
    model = torch.nn.Linear(2, 2)
    optimizer = expression(function, "optimizer", {"optim": torch.optim, "meta_model": model})
    for source_key, config_key in [("lr", "learning_rate"), ("weight_decay", "weight_decay"),
                                   ("eps", "eps"), ("amsgrad", "amsgrad")]:
        assert optimizer.defaults[source_key] == config["training"][config_key]
    assert optimizer.defaults["betas"] == tuple(config["training"]["betas"])
    assert expression(function, "epochs", {}) == config["training"]["epochs"]
    params = expression(function, "params", {"collate_dgl": None})
    for key in ("batch_size", "num_workers", "drop_last"):
        assert params[key] == config["training"][key]
    assert params["shuffle"] == config["training"]["shuffle_train"] == config["training"]["shuffle_test"]
    original_splitter = expression(function, "skf", {"StratifiedKFold": StratifiedKFold})
    assert original_splitter.random_state is None  # A deliberate clean difference.
    assert original_splitter.n_splits == 3 and original_splitter.shuffle
    state = np.random.get_state()
    try:
        # Condition the unspecified upstream RNG for this experiment only.
        np.random.seed(config["seed"])
        expected = list(original_splitter.split(np.zeros(len(labels)), labels))
    finally:
        np.random.set_state(state)
    actual = list(StratifiedKFold(3, shuffle=True, random_state=config["seed"])
                  .split(np.zeros(len(labels)), labels))
    for (a_train, a_test), (b_train, b_test) in zip(actual, expected):
        np.testing.assert_array_equal(a_train, b_train)
        np.testing.assert_array_equal(a_test, b_test)

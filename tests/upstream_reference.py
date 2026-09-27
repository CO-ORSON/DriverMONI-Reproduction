"""Load selected classes from a user's local upstream checkout, only in memory.

No authors' class bodies are redistributed with this project. Hash verification
limits execution to the two inspected files; imports/CLI code are not executed.
"""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import dgl
import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data


SOURCE_HASHES = {
    "CV_multimodal_compare.py": "9ece040b35d708e67eeddd74689e19300206cf10a2a85d07f0ef71024606911f",
    "torch_models_linear.py": "a8ed2d98ec78ea20ff48e788a725555031d9a8f213255f047b4da5414f6e7e20",
}
MODEL_NAMES = ("GATLayer", "GAT_L", "gcn_linc_embedder_transfer")
DATASET_NAME = "classify_graph_data_encoder_humannet"


def verified_source(root: Path, filename: str) -> str:
    content = (root / filename).read_bytes()
    if hashlib.sha256(content).hexdigest() != SOURCE_HASHES[filename]:
        raise ValueError(f"Unrecognized upstream source: {filename}; use the documented pinned revision")
    return content.decode("utf-8")


def adapted_definition(root: Path, filename: str, name: str) -> ast.ClassDef:
    tree = ast.parse(verified_source(root, filename))
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)
    text = ast.unparse(node)
    changes = {}
    if name == "gcn_linc_embedder_transfer":
        changes["graph.ndata['feat'].type(torch.FloatTensor).cuda()"] = (
            "graph.ndata['feat'].to(dtype=torch.float32, device=feat.device)"
        )
    if name == DATASET_NAME:
        changes = {
            "nodes_l = list(set(node_l))": "nodes_l = IndexedNodes(set(node_l))",
            "g_dgl = dgl.DGLGraph()\n        g_dgl.from_networkx(g_nx)": "g_dgl = dgl.from_networkx(g_nx)",
            "dgl.DGLGraph(self.graph.subgraph(new_nodes))": "dgl.node_subgraph(self.graph, new_nodes)",
        }
    for old, new in changes.items():
        if text.count(old) != 1:
            raise ValueError(f"Expected exactly one compatibility adaptation in {name}")
        text = text.replace(old, new)
    return ast.parse(text).body[0]


class IndexedNodes(list):
    """Preserve a unique list's indices while avoiding repeated linear scans."""

    def __init__(self, values):
        super().__init__(values)
        self.indices = {value: index for index, value in enumerate(self)}

    def index(self, value, *args):
        if args:
            return super().index(value, *args)
        try:
            return self.indices[value]
        except KeyError:
            raise ValueError(value) from None


def load_reference(root: Path, *, dataset: bool = False) -> SimpleNamespace:
    namespace = dict(torch=torch, nn=nn, F=F, dgl=dgl, nx=nx, np=np, data=data,
                     IndexedNodes=IndexedNodes, tqdm=lambda values: values)
    selections = [("CV_multimodal_compare.py", DATASET_NAME)] if dataset else [
        ("torch_models_linear.py", "feat_L_BRCA"),
        *(("CV_multimodal_compare.py", name) for name in MODEL_NAMES),
    ]
    for filename, name in selections:
        module = ast.Module(body=[adapted_definition(root, filename, name)], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(root / filename), "exec"), namespace)
    if not dataset:
        namespace["torch_models_linear"] = SimpleNamespace(feat_L_BRCA=namespace["feat_L_BRCA"])
    return SimpleNamespace(**namespace)

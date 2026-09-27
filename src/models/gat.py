"""Custom sigmoid attention from upstream ``GATLayer`` and ``GAT_L``.

Edges are source -> destination. Graph construction (including self-loops and
edge direction) is the caller's responsibility; this module never adds edges.
"""

from typing import Dict

import dgl
import torch
from dgl.udf import EdgeBatch, NodeBatch
from torch import Tensor, nn
from torch.nn import functional as F


class GATLayer(nn.Module):
    """Aggregate [N, in_dim] into [N, out_dim] with unnormalized sigmoid weights.

    z = W x; e(src, dst) = LeakyReLU(a([z_src || z_dst]));
    h_dst = sum_src(sigmoid(e(src, dst)) * z_src).
    """

    def __init__(self, in_dim: int = 13, out_dim: int = 64) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("attention dimensions must be positive")
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Upstream uses default initialization for both bias-free projections.
        self.projection = nn.Linear(in_dim, out_dim, bias=False)
        self.attention = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges: EdgeBatch) -> Dict[str, Tensor]:
        """Compute one attention logit per edge, preserving src/dst order."""
        paired_features = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        return {"e": F.leaky_relu(self.attention(paired_features), negative_slope=0.01)}

    @staticmethod
    def message_func(edges: EdgeBatch) -> Dict[str, Tensor]:
        """Send projected source features and edge logits to destinations."""
        return {"z": edges.src["z"], "e": edges.data["e"]}

    @staticmethod
    def reduce_func(nodes: NodeBatch) -> Dict[str, Tensor]:
        """Sum incoming messages; sigmoid is deliberately not a softmax."""
        attention_weights = torch.sigmoid(nodes.mailbox["e"])
        return {"h": torch.sum(attention_weights * nodes.mailbox["z"], dim=1)}

    def forward(self, node_features: Tensor, graph: dgl.DGLGraph) -> Tensor:
        """Return node embeddings [N, out_dim] without mutating graph data."""
        if not graph.is_homogeneous:
            raise ValueError("GATLayer requires a homogeneous graph")
        expected_shape = (graph.num_nodes(), self.in_dim)
        if node_features.ndim != 2 or tuple(node_features.shape) != expected_shape:
            raise ValueError(
                f"node features must have shape [N, {self.in_dim}] = "
                f"{expected_shape}, got {tuple(node_features.shape)}"
            )
        if not torch.is_floating_point(node_features):
            raise TypeError("node features must be floating-point tensors")
        if node_features.device != graph.device:
            raise ValueError("node features and graph must be on the same device")
        if node_features.device != self.projection.weight.device:
            raise ValueError("model, node features, and graph must be on the same device")
        with graph.local_scope():
            graph.ndata["z"] = self.projection(node_features)
            graph.apply_edges(self.edge_attention)
            graph.update_all(self.message_func, self.reduce_func)
            return graph.ndata["h"]


class GATEncoder(nn.Module):
    """Encode B ego graphs with node features [N, 13] into targets [B, 32].

    ``graph.ndata['feat']`` contains features; ``graph.ndata['is_target']`` is a binary
    [N] mask with exactly one target per component, in the DGL batch order.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attention_layer = GATLayer(13, 64)
        self.target_batch_norm = nn.BatchNorm1d(64)
        self.target_projection = nn.Linear(64, 32)
        nn.init.xavier_uniform_(self.target_projection.weight)
        self.output_batch_norm = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()

    def forward(self, graph: dgl.DGLGraph) -> Tensor:
        """Message-pass over all nodes, then select and encode B targets."""
        if not graph.is_homogeneous:
            raise ValueError("GATEncoder requires a homogeneous graph")
        if "feat" not in graph.ndata or "is_target" not in graph.ndata:
            raise ValueError("graph.ndata must contain 'feat' and the target mask 'is_target'")
        target_ids = graph.ndata["is_target"]
        if target_ids.ndim != 1 or target_ids.shape[0] != graph.num_nodes():
            raise ValueError("graph.ndata['is_target'] must have shape [N]")
        if not bool(torch.all((target_ids == 0) | (target_ids == 1))):
            raise ValueError("target mask 'is_target' must contain only 0 and 1")

        component_sizes = graph.batch_num_nodes().tolist()
        if not component_sizes or any(size == 0 for size in component_sizes):
            raise ValueError("each ego graph must contain at least one node")
        target_mask = target_ids == 1
        component_masks = torch.split(target_mask, component_sizes)
        if any(int(mask.sum()) != 1 for mask in component_masks):
            raise ValueError("each graph must have exactly one target in the 'is_target' mask")
        if self.training and len(component_sizes) < 2:
            raise ValueError("GAT training requires at least two graphs for BatchNorm")

        node_embeddings = self.attention_layer(graph.ndata["feat"], graph)
        target_gene_embedding = node_embeddings[target_mask]
        target_gene_embedding = self.relu(self.target_batch_norm(target_gene_embedding))
        target_gene_embedding = self.target_projection(target_gene_embedding)
        return self.relu(self.output_batch_norm(target_gene_embedding))

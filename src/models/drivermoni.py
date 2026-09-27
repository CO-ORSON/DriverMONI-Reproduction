"""BRCA fusion model matching upstream ``gcn_linc_embedder_transfer``."""

import dgl
import torch
from torch import Tensor, nn

from .dnn import DNNEncoder
from .gat import GATEncoder


class DriverMONI(nn.Module):
    """Combine gene features [B, 12] and B ego graphs into logits [B, 2]."""

    def __init__(self) -> None:
        super().__init__()
        self.gat = GATEncoder()
        self.dnn = DNNEncoder()
        # The active upstream classifier uses nn.Linear's default initialization.
        self.classifier = nn.Linear(96, 2)
        self.gat_projection = nn.Linear(32, 32)
        self.gat_batch_norm = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        nn.init.xavier_uniform_(self.gat_projection.weight)

    def encode(self, gene_features: Tensor, graph: dgl.DGLGraph) -> Tensor:
        """Return fused embeddings [B, 96], ordered DNN first, GAT second."""
        if gene_features.ndim != 2 or gene_features.shape[1] != 12:
            raise ValueError(
                f"gene features must have shape [B, 12], got {tuple(gene_features.shape)}"
            )
        if gene_features.shape[0] != graph.batch_size:
            raise ValueError("gene feature batch size must match the number of graphs")
        if gene_features.device != self.classifier.weight.device:
            raise ValueError("gene features and model must be on the same device")

        dnn_embedding = self.dnn(gene_features)
        gat_embedding = self.gat(graph)
        # Required by the active training path, although absent from the paper diagram.
        gat_embedding = self.relu(self.gat_batch_norm(self.gat_projection(gat_embedding)))
        return torch.cat([dnn_embedding, gat_embedding], dim=1)

    def forward(self, gene_features: Tensor, graph: dgl.DGLGraph) -> Tensor:
        """Return unnormalized logits [B, 2]; loss/evaluation applies softmax."""
        return self.classifier(self.encode(gene_features, graph))

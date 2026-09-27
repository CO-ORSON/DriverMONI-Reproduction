"""BRCA DNN from upstream ``torch_models_linear.feat_L_BRCA``."""

import torch
from torch import Tensor, nn


class DNNEncoder(nn.Module):
    """Encode target-gene features [B, 12] as embeddings [B, 64]."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(12, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.7)
        self.dropout2 = nn.Dropout(p=0.6)

        # Only weights are overwritten upstream; preserve default linear biases.
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Return [B, 64]; training BatchNorm requires B >= 2."""
        if x.ndim != 2 or x.shape[1] != 12:
            raise ValueError(f"gene features must have shape [B, 12], got {tuple(x.shape)}")
        if x.shape[0] == 0:
            raise ValueError("gene feature batch must be nonempty")
        if not torch.is_floating_point(x):
            raise TypeError("gene features must be floating-point tensors")
        if self.training and x.shape[0] < 2:
            raise ValueError("DNN training requires at least two genes for BatchNorm")

        target_gene_embedding = self.linear1(x)
        target_gene_embedding = self.batch_norm1(target_gene_embedding)
        target_gene_embedding = self.dropout1(self.relu(target_gene_embedding))
        target_gene_embedding = self.linear2(target_gene_embedding)
        target_gene_embedding = self.batch_norm2(target_gene_embedding)
        target_gene_embedding = self.dropout2(self.relu(target_gene_embedding))
        target_gene_embedding = self.linear3(target_gene_embedding)
        target_gene_embedding = self.batch_norm3(target_gene_embedding)
        return self.relu(target_gene_embedding)

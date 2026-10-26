"""Feature order and faithful bounded graph rendering for the research app."""

import hashlib
from types import SimpleNamespace

import dgl
import pandas as pd
import pytest

from src.app_data import feature_rows, graph_preview_dot, read_features
from src.feature_schema import DNN_FEATURES, GAT_FEATURES


def feature_frame():
    return pd.DataFrame([
        {"gene_ens": gene, "gene_symbol": symbol,
         **{name: row + column / 20 for column, name in enumerate(GAT_FEATURES)}}
        for row, (gene, symbol) in enumerate((("ENSG1", "ONE"), ("ENSG2", "TWO")))
    ])


def test_feature_display_uses_model_order_despite_csv_column_order(tmp_path):
    path = tmp_path / "features.csv"
    source = feature_frame()
    source.loc[:, list(reversed(source.columns))].to_csv(path, index=False)
    actual, digest = read_features(path)
    assert list(actual.columns) == ["gene_ens", "gene_symbol", *GAT_FEATURES]
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    rows = feature_rows(actual.iloc[1])
    assert rows["Feature"].tolist() == list(GAT_FEATURES)
    assert rows["Value"].tolist() == pytest.approx([1 + i / 20 for i in range(13)])
    assert rows["Branch"].tolist() == ["DNN + GAT"] * len(DNN_FEATURES) + ["GAT only"]


@pytest.mark.parametrize("invalid", ["missing", "nan", "infinity", "overflow", "duplicate_id", "blank_symbol"])
def test_rejects_ambiguous_or_invalid_features(tmp_path, invalid):
    frame = feature_frame()
    if invalid == "missing":
        frame = frame.drop(columns="PPI")
    elif invalid == "duplicate_id":
        frame.loc[1, "gene_ens"] = frame.loc[0, "gene_ens"]
    elif invalid == "blank_symbol":
        frame.loc[1, "gene_symbol"] = " "
    else:
        frame.loc[1, "PPI"] = {"nan": float("nan"), "infinity": float("inf"), "overflow": 1e40}[invalid]
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_features(path)


def test_symbol_deduplication_matches_dataset_contract(tmp_path):
    frame = feature_frame()
    frame.loc[1, "gene_symbol"] = "ONE"
    path = tmp_path / "features.csv"
    frame.to_csv(path, index=False)
    actual, _ = read_features(path)
    assert actual["gene_ens"].tolist() == ["ENSG1"]


def test_graph_preview_preserves_actual_direction_duplicate_counts_and_target():
    # Target is deliberately last, and the displayed subset omits node 1.
    graph = dgl.graph(([0, 0, 1, 2, 0, 1, 2], [2, 2, 2, 0, 0, 1, 2]), num_nodes=3)
    sample = SimpleNamespace(gene_id="C", node_gene_ids=("A", "B", "C"), graph=graph)
    before = graph.edges()
    dot = graph_preview_dot(sample, {"A": 'symbol"with quotes', "C": "TARGET"}, max_nodes=2)
    assert 'n2 [label="TARGET (target)"' in dot
    assert 'n0 -> n2 [label="×2"]' in dot
    assert 'n2 -> n0;' in dot
    assert 'symbol\\"with quotes' in dot
    assert 'n1' not in dot
    assert 'n2 -> n2' not in dot
    assert graph.num_edges() == 7
    assert all((a == b).all() for a, b in zip(before, graph.edges(), strict=True))

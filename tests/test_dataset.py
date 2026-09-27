"""Independent upstream graph comparisons, feature contracts, and batching."""

from collections import Counter
from pathlib import Path
import pickle
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import torch
import dgl
from torch.utils.data import DataLoader

from src.dataset import (
    DNN_FEATURES, GAT_FEATURES, DriverMONIDataset, collate_gene_samples,
    load_brca_dataset, prepare_gene_table,
)
from src.graph_utils import PPIGraph, load_humannet_edges, load_neighbor_cache
from src.models import DriverMONI


@pytest.fixture
def toy_data():
    records = []
    for index, gene in enumerate(["A", "B", "C", "D"]):
        record = {name: float(index * 100 + offset) for offset, name in enumerate(GAT_FEATURES)}
        record.update(gene_ens=gene, gene_symbol=gene, label=int(gene == "A"))
        records.append(record)
    features = pd.DataFrame(records)
    edges = pd.DataFrame({
        "entrez_id_1": ["A", "A", "B", "C", "B", "C"],
        "entrez_id_2": ["B", "B", "C", "A", "B", "Z"],
        "LLS": [0.1] * 6,  # Low scores must not filter these edges.
    })
    neighbors = {"A": ["C", "B", "A", "Z", "B"], "B": ["A", "B", "C"],
                 "C": ["C", "A", "B", "Z"], "D": []}
    return features, edges, neighbors


def edge_counter(graph, genes):
    source, destination = graph.edges()
    return Counter((genes[s], genes[d]) for s, d in zip(source.tolist(), destination.tolist()))


def assert_sample_matches_original(clean, original, index):
    sample = clean[index]
    gene_id, dnn_features, graph, one_hot_label = original[index]
    original_genes = tuple(original.all_nodes[node] for node in graph.ndata["idx"].tolist())
    assert sample.gene_id == gene_id
    assert sample.label == one_hot_label[1]
    assert set(sample.node_gene_ids) == set(original_genes)
    assert edge_counter(sample.graph, sample.node_gene_ids) == edge_counter(graph, original_genes)
    torch.testing.assert_close(sample.gene_features, torch.tensor(np.asarray(dnn_features, dtype=np.float32)), rtol=0, atol=0)
    # Compare feature/mask rows by biological gene ID, allowing deterministic reindexing.
    new_index = {gene: index for index, gene in enumerate(sample.node_gene_ids)}
    order = [new_index[gene] for gene in original_genes]
    torch.testing.assert_close(sample.graph.ndata["feat"][order], graph.ndata["feat"], rtol=0, atol=0)
    torch.testing.assert_close(sample.graph.ndata["is_target"][order], graph.ndata["id"].bool())
    expected_target = [gene for gene, flag in zip(original_genes, graph.ndata["id"].tolist()) if flag]
    assert expected_target == [sample.gene_id]
    # direct_con is unused by GAT but its renamed equivalent is retained.
    source, destination = sample.graph.edges()
    expected_edge_mask = torch.tensor([
        sample.node_gene_ids[s] == sample.gene_id or sample.node_gene_ids[d] == sample.gene_id
        for s, d in zip(source.tolist(), destination.tolist())
    ])
    torch.testing.assert_close(sample.graph.edata["is_target_edge"], expected_edge_mask)


def test_exact_feature_order():
    assert DNN_FEATURES == (
        "synonymous_variant", "stop_gained", "missense_variant", "frameshift_variant",
        "splice", "inframe", "lost_stop and start", "deg", "related_pathway",
        "dir_pathway", "muta_count", "miss_ratio",
    )
    assert len(DNN_FEATURES) == 12
    assert GAT_FEATURES == DNN_FEATURES + ("PPI",)
    assert len(GAT_FEATURES) == 13


def test_duplicate_edges_induced_edges_and_extra_csv_loop(toy_data):
    features, edges, neighbors = toy_data
    sample = DriverMONIDataset(features, PPIGraph(edges, features.gene_ens), neighbors)[0]
    assert sample.node_gene_ids == ("A", "B", "C")
    assert edge_counter(sample.graph, sample.node_gene_ids) == Counter({
        ("A", "A"): 1, ("B", "B"): 2, ("C", "C"): 1,
        ("A", "B"): 2, ("B", "C"): 1, ("C", "A"): 1,
    })
    assert sample.graph.num_edges() == 8
    assert sample.gene_features.shape == (12,)
    assert sample.graph.ndata["feat"].shape == (3, 13)
    assert "label" not in sample.graph.ndata and "id" not in sample.graph.ndata


def test_toy_samples_match_authors_dataset(toy_data, upstream_dataset):
    features, edges, neighbors = toy_data
    clean = DriverMONIDataset(features, PPIGraph(edges, features.gene_ens), neighbors)
    original = upstream_dataset.classify_graph_data_encoder_humannet(edges, neighbors, features, features)
    for index in range(len(clean)):
        assert_sample_matches_original(clean, original, index)


def test_graph_reindexing_preserves_model_scores(toy_data, upstream_dataset, upstream, weight_transfer):
    """Compare complete dataset-to-logits paths despite different node numbering."""
    features, edges, neighbors = toy_data
    dataset = DriverMONIDataset(features, PPIGraph(edges, features.gene_ens), neighbors)
    original_dataset = upstream_dataset.classify_graph_data_encoder_humannet(
        edges, neighbors, features, features,
    )
    samples = [original_dataset[index] for index in range(len(dataset))]
    original_features = torch.tensor(np.asarray([sample[1] for sample in samples], dtype=np.float32))
    original_graph = dgl.batch([sample[2] for sample in samples])
    clean_batch = collate_gene_samples([dataset[index] for index in range(len(dataset))])
    original = upstream.gcn_linc_embedder_transfer(upstream.GAT_L(), "BRCA").eval()
    clean = DriverMONI().eval()
    weight_transfer.model(clean, original)
    with torch.no_grad():
        expected = original(original_features, original_graph)
        actual = clean(clean_batch.gene_features, clean_batch.graph)
    # Reordering incoming messages can change floating-point summation order.
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_fallback_neighbors_are_undirected(toy_data):
    features, edges, _ = toy_data
    neighbors = PPIGraph(edges, features.gene_ens).discover_neighbors(["A", "D", "Z"])
    assert neighbors == {"A": ["A", "B", "C"], "D": [], "Z": ["C", "Z"]}


def test_empty_cache_entry_is_preserved_even_when_edges_exist(toy_data):
    features, edges, neighbors = toy_data
    neighbors["A"] = []
    sample = DriverMONIDataset(features, PPIGraph(edges, features.gene_ens), neighbors)[0]
    assert sample.node_gene_ids == ("A",)
    assert sample.graph.num_edges() == 1


def test_target_subset_retains_other_fold_features(toy_data):
    features, edges, neighbors = toy_data
    dataset = DriverMONIDataset(features, PPIGraph(edges, features.gene_ens), neighbors, ["A"])
    assert len(dataset) == 1
    assert dataset[0].node_gene_ids == ("A", "B", "C")


def test_collation_connects_to_model_and_is_independent(toy_data):
    features, edges, neighbors = toy_data
    dataset = DriverMONIDataset(features, PPIGraph(edges, features.gene_ens), neighbors)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_gene_samples)
    batch = next(iter(loader)).to("cpu")
    assert batch.gene_features.shape == (4, 12)
    assert batch.labels.dtype == torch.long
    assert batch.labels.tolist() == [1, 0, 0, 0]
    assert batch.gene_ids == ("A", "B", "C", "D")
    assert int(batch.graph.ndata["is_target"].sum()) == 4
    assert DriverMONI()(batch.gene_features, batch.graph).shape == (4, 2)
    changed = dataset[0]
    changed.gene_features.fill_(999)
    changed.graph.ndata["feat"].fill_(999)
    assert dataset[0].gene_features[0] == 0
    assert dataset[0].graph.ndata["feat"][0, 0] == 0


def test_filter_order_and_labels(toy_data):
    features, _, _ = toy_data
    duplicate = features.iloc[[0]].copy()
    duplicate["gene_ens"] = "A_SECOND_ID"
    features = pd.concat([features, duplicate], ignore_index=True)
    result = prepare_gene_table(features, ["A"], ["A_SECOND_ID", "B", "C"], ["A", "B"])
    assert result.gene_ens.tolist() == ["A", "B"]
    assert result.label.tolist() == [1, 0]
    with pytest.raises(ValueError, match="absent genes"):
        prepare_gene_table(features, ["A"], ["NOT_PRESENT"], ["A"])


def test_invalid_data_and_missing_cache_fail_clearly(toy_data):
    features, edges, neighbors = toy_data
    ppi = PPIGraph(edges, features.gene_ens)
    with pytest.raises(ValueError, match="missing target"):
        DriverMONIDataset(features, ppi, {})
    features.loc[0, "PPI"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        DriverMONIDataset(features, ppi, neighbors)
    with pytest.raises(ValueError, match="missing columns"):
        DriverMONIDataset(features.drop(columns="miss_ratio"), ppi, neighbors)


def test_cache_and_archive_loading(tmp_path, toy_data):
    _, edges, neighbors = toy_data
    cache = tmp_path / "neighbors.pickle"
    cache.write_bytes(pickle.dumps(neighbors))
    assert load_neighbor_cache(cache) == neighbors
    cache.write_bytes(pickle.dumps({"A": 1}))
    with pytest.raises(ValueError, match="string lists"):
        load_neighbor_cache(cache)
    reference_dir = tmp_path / "for_ref"
    reference_dir.mkdir()
    with ZipFile(reference_dir / "humannet_v2_and_v3.zip", "w") as archive:
        archive.writestr("humannet_v2_and_v3/humannetv3_ens_ppi.csv", edges.to_csv(index=False))
    pd.testing.assert_frame_equal(load_humannet_edges(tmp_path), edges)


@pytest.fixture(scope="module")
def brca_pair(request, upstream_dataset):
    root_arg = request.config.getoption("--brca-data-root")
    if root_arg is None:
        pytest.skip("Pass --brca-data-root to compare the actual authors' BRCA data")
    root = Path(root_arg)
    clean = load_brca_dataset(root)
    # Independently transcribe the upstream pre-CV filter, preserving its lookup order.
    raw = pd.read_csv(root / "sample_preprocessed_file/BRCA_input_data.csv")
    expected = raw.drop_duplicates("gene_symbol").reset_index(drop=True)
    comparison = pd.read_csv(root / "for_concat_data/after_concat_BRCA_genes2.csv")["gene"].tolist()
    lookup = raw.drop_duplicates("gene_ens").set_index("gene_ens")
    symbols = set(lookup.loc[comparison, "gene_symbol"])
    allowed = set(pd.read_csv(root / "gene_info_for_GOSemSim.csv")["Symbol"])
    indices = [i for i in range(len(expected)) if expected.loc[i, "gene_symbol"] in symbols & allowed]
    expected = expected.loc[indices].reset_index(drop=True)
    drivers = pd.read_csv(root / "using_Data/BRCA_label.csv")["gene_label"].tolist()
    expected["label"] = [1 if gene in drivers else 0 for gene in expected.gene_ens]
    pd.testing.assert_frame_equal(clean.gene_table, expected)
    assert len(clean) == 15028 and int(expected.label.sum()) == 42
    original = upstream_dataset.classify_graph_data_encoder_humannet(
        load_humannet_edges(root), clean.neighbors, expected, expected,
    )
    return clean, original


@pytest.mark.parametrize("gene", [
    "ENSG00000000003",  # TSPAN6: ordinary negative.
    "ENSG00000012048",  # BRCA1: positive, larger neighborhood.
    "ENSG00000141510",  # TP53: positive, high connectivity.
    "ENSG00000005810",  # MYCBP2: empty supplied neighborhood.
    "ENSG00000056586",  # RC3H2: anomalously empty cache despite PPI neighbors.
])
def test_real_brca_samples_match_original(brca_pair, gene):
    clean, original = brca_pair
    index = clean.target_gene_ids.index(gene)
    assert_sample_matches_original(clean, original, index)

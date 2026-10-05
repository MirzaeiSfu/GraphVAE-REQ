"""Export attributed TU graphs in the GraphVAE-REQ interchange format."""

import math
from pathlib import Path

import numpy as np


def _load_graphvae_req():
    try:
        from ggm_eval import evaluate_with_trained_gnns, save_pyg_collection
        from ggm_eval.adapters import attributed_arrays_to_pyg
    except ImportError as exc:
        raise RuntimeError(
            "GraphVAE-REQ's graph_evaluation package is required for export."
        ) from exc
    return evaluate_with_trained_gnns, save_pyg_collection, attributed_arrays_to_pyg


def _quantile_thresholds(values, bins):
    values = sorted(float(value) for value in values)
    if not values:
        return []
    thresholds = []
    for index in range(1, bins):
        rank = (len(values) - 1) * index / bins
        lower, upper = math.floor(rank), math.ceil(rank)
        value = values[lower] if lower == upper else (
            values[lower] * (upper - rank) + values[upper] * (rank - lower)
        )
        if not thresholds or value > thresholds[-1]:
            thresholds.append(float(value))
    return thresholds


def _digitize(attrs, thresholds):
    if attrs is None:
        return None
    if attrs.shape[1] == 0:
        return np.zeros((attrs.shape[0], 0), dtype=np.int64)
    return np.stack([
        np.digitize(attrs[:, column], thresholds[column], right=False)
        for column in range(attrs.shape[1])
    ], axis=1).astype(np.int64)


def _dense_edges(edge_attr, num_nodes):
    matrix = np.asarray(edge_attr).reshape(num_nodes, num_nodes, -1)
    rows = []
    for source in range(num_nodes):
        for target in range(source + 1, num_nodes):
            if int(matrix[source, target, 0]) > 0:
                # Native edge fields use zero for absent pairs and are shifted
                # by one in the prepared bundle.  Convert them back here.
                features = tuple(int(value) - 1
                                 for value in matrix[source, target, 1:])
                rows.append((source, target, features))
    return rows


def _loader_records(loader, split):
    records = []
    for batch_index, batch in enumerate(loader):
        batch = batch.to("cpu")
        node_offset = edge_offset = 0
        attrs = getattr(batch, "x_attr", None)
        for graph_index in range(int(batch.num_graphs)):
            num_nodes = int(batch.num_node_per_graph[graph_index])
            node_slice = slice(node_offset, node_offset + num_nodes)
            edge_slice = slice(edge_offset, edge_offset + num_nodes ** 2)
            edge_rows = _dense_edges(batch.edge_attr[edge_slice], num_nodes)
            records.append({
                "name": f"{split}-{batch_index}-{graph_index}",
                "node_categories": batch.x[node_slice].reshape(num_nodes, -1).numpy().astype(np.int64),
                "node_attrs": None if attrs is None else attrs[node_slice].numpy().astype(np.float32),
                "num_nodes": num_nodes,
                "edge_rows": edge_rows,
            })
            node_offset += num_nodes
            edge_offset += num_nodes ** 2
    return records


def _generated_records(graphs):
    records = []
    for graph_index, graph in enumerate(graphs):
        nodes = sorted(graph.nodes())
        if len(nodes) < 1:
            continue
        mapping = {node: index for index, node in enumerate(nodes)}
        attrs = [graph.nodes[node].get("attr") for node in nodes]
        have_attrs = all(attr is not None for attr in attrs)
        edge_rows = []
        for source, target, data in graph.edges(data=True):
            if source == target:
                continue
            source, target = sorted((mapping[source], mapping[target]))
            features = data.get("categorical_features")
            if features is None:
                features = [] if "label" not in data else [int(data["label"])]
            edge_rows.append((source, target, tuple(int(value) for value in features)))
        node_categories = []
        for node in nodes:
            values = graph.nodes[node].get("categorical_features")
            if values is None:
                values = [int(graph.nodes[node].get("label", 0))]
            node_categories.append([int(value) for value in values])
        records.append({
            "name": f"generated-{graph_index}",
            "node_categories": np.asarray(node_categories, dtype=np.int64),
            "node_attrs": np.asarray(attrs, dtype=np.float32) if have_attrs else None,
            "num_nodes": len(nodes),
            "edge_rows": sorted(set(edge_rows)),
        })
    return records


def _onehot_columns(columns, value_sets):
    width = sum(len(values) for values in value_sets)
    result = np.zeros((columns.shape[0], width), dtype=np.float32)
    offset = 0
    for column, values in enumerate(value_sets):
        lookup = {value: offset + index for index, value in enumerate(values)}
        for row, value in enumerate(columns[:, column]):
            if int(value) in lookup:
                result[row, lookup[int(value)]] = 1.0
        offset += len(values)
    return result


def _records_to_pyg(records, thresholds, node_value_sets, edge_value_sets, save_adapter):
    attr_dim = len(thresholds)
    attr_values = [list(range(0, len(thresholds[column]) + 1))
                   for column in range(attr_dim)]
    value_sets = node_value_sets + attr_values
    graphs = []
    for record in records:
        quantized = _digitize(record["node_attrs"], thresholds) if attr_dim else None
        columns = record["node_categories"]
        if quantized is not None:
            columns = np.concatenate((columns, quantized), axis=1)
        node_features = _onehot_columns(columns, value_sets)
        edges = np.asarray([(source, target) for source, target, _ in record["edge_rows"]], dtype=np.int64)
        edge_features = None
        if len(record["edge_rows"]) and edge_value_sets:
            edge_columns = np.asarray(
                [features for _, _, features in record["edge_rows"]],
                dtype=np.int64,
            )
            edge_features = _onehot_columns(edge_columns, edge_value_sets)
        if len(edges):
            graphs.append(save_adapter(edges, node_features, edge_features,
                                       np.arange(record["num_nodes"]), name=record["name"]))
    return graphs, {"node_category_values": node_value_sets, "attribute_dim": attr_dim,
                    "attribute_bins": [len(values) for values in attr_values],
                    "edge_category_values": edge_value_sets}


def export_tu_pyg_collections(train_loader, reference_loader, generated_graphs,
                               output_dir, dataset_name, bins=8):
    """Write train/reference/generated PyG collections for GraphVAE-REQ."""
    _, save_collection, save_adapter = _load_graphvae_req()
    train_records = _loader_records(train_loader, "train")
    reference_records = _loader_records(reference_loader, "reference")
    generated_records = _generated_records(generated_graphs)
    all_records = (train_records, reference_records, generated_records)
    source_records = train_records or reference_records
    attr_dim = next((record["node_attrs"].shape[1] for record in source_records
                     if record["node_attrs"] is not None), 0)
    thresholds = [
        _quantile_thresholds([
            float(record["node_attrs"][row, column])
            for record in source_records if record["node_attrs"] is not None
            for row in range(record["node_attrs"].shape[0])
        ], bins) for column in range(attr_dim)
    ]
    node_field_count = source_records[0]["node_categories"].shape[1]
    node_value_sets = [sorted({
        int(record["node_categories"][row, field])
        for records in all_records for record in records
        for row in range(record["node_categories"].shape[0])
    }) for field in range(node_field_count)]
    edge_field_count = max(
        (len(features) for records in all_records for record in records
         for _, _, features in record["edge_rows"]), default=0
    )
    edge_value_sets = [sorted({
        int(features[field])
        for records in all_records for record in records
        for _, _, features in record["edge_rows"]
        if len(features) > field
    }) for field in range(edge_field_count)]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    exports = {}
    for name, records in (("train", train_records), ("reference", reference_records),
                          ("generated", generated_records)):
        graphs, metadata = _records_to_pyg(records, thresholds, node_value_sets,
                                           edge_value_sets, save_adapter)
        path = destination / f"{name}.pt"
        save_collection(path, graphs, metadata={"dataset": dataset_name, **metadata})
        exports[name] = str(path)
    return {"dataset": dataset_name, "exports": exports, "thresholds": thresholds}


def evaluate_exported_tu_pyg_collections(summary, dataset_name, output_dir, device="auto"):
    evaluate, _, _ = _load_graphvae_req()
    return evaluate(
        dataset=dataset_name,
        generated=summary["exports"]["generated"],
        reference=summary["exports"]["reference"],
        output_dir=output_dir,
        device=device,
    )

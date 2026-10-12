#!/usr/bin/env python3
"""Import a DGL GIN benchmark into a FactorBase-ready MySQL database.

This module is shared by the MUTAG and PTC entry points.  It deliberately
matches the dataset source used by GraphVAE-MM: ``dgl.data.GINDataset`` with
``self_loop=False``.  All graphs are flattened into one disconnected union,
and graph classification labels are excluded from the SQL schema so they
cannot leak into learned motifs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data_raw" / "dgl"
SUPPORTED_DATASETS = {"MUTAG", "PTC"}


@dataclass(frozen=True)
class GINGraph:
    node_features: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    graph_label: int


def parse_args(dataset_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Load the DGL GIN {dataset_name} dataset into MySQL for FactorBase."
        )
    )
    parser.add_argument("--db-name", default=f"{dataset_name.lower()}_undir_feat")
    edge_group = parser.add_mutually_exclusive_group()
    edge_group.add_argument(
        "--directed",
        action="store_true",
        help="Store exactly the directed edge rows exposed by DGL.",
    )
    edge_group.add_argument(
        "--undirected",
        action="store_true",
        help="Ensure both A->B and B->A rows exist for every edge pair (default).",
    )
    parser.add_argument("--mysql-host", default="localhost")
    parser.add_argument("--mysql-user", default="fbuser")
    parser.add_argument("--mysql-password", default="")
    parser.add_argument("--mysql-port", type=int, default=3306)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate the dataset without changing MySQL.",
    )
    args = parser.parse_args()
    if not args.directed and not args.undirected:
        args.undirected = True
    return args


def scalar_int(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def node_labels(graph) -> tuple[int, ...]:
    for key in ("label", "attr", "feat"):
        if key not in graph.ndata:
            continue
        values = graph.ndata[key]
        labels = []
        for node_id in range(graph.num_nodes()):
            value = values[node_id]
            if hasattr(value, "numel") and value.numel() > 1:
                # DGL GINDataset exposes ``attr`` as a one-hot encoding and
                # ``label`` as the corresponding categorical state.
                value = value.argmax()
            labels.append(scalar_int(value) + 1)
        return tuple(labels)
    raise KeyError(
        "GIN graph has no supported node label; "
        f"available keys are {sorted(graph.ndata.keys())}"
    )


def load_dataset(dataset_name: str) -> list[GINGraph]:
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported GIN dataset {dataset_name!r}; "
            f"choose from {sorted(SUPPORTED_DATASETS)}"
        )

    import dgl

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    dataset = dgl.data.GINDataset(
        name=dataset_name,
        self_loop=False,
        raw_dir=str(DATA_ROOT),
    )
    graphs = []
    for graph, label in zip(dataset.graphs, dataset.labels):
        sources, targets = graph.edges()
        edges = tuple(
            (int(source), int(target))
            for source, target in zip(sources.tolist(), targets.tolist())
            if int(source) != int(target)
        )
        graphs.append(
            GINGraph(
                node_features=node_labels(graph),
                edges=edges,
                graph_label=scalar_int(label),
            )
        )
    return graphs


def normalized_edges(
    edges: Iterable[tuple[int, int]],
    *,
    undirected: bool,
) -> list[tuple[int, int]]:
    rows = set()
    for source, target in edges:
        rows.add((source, target))
        if undirected:
            rows.add((target, source))
    return sorted(rows)


def dataset_summary(dataset_name: str, graphs: list[GINGraph], undirected: bool) -> dict:
    graph_sizes = [len(graph.node_features) for graph in graphs]
    source_rows = sum(len(set(graph.edges)) for graph in graphs)
    undirected_pairs = sum(
        len({tuple(sorted((source, target))) for source, target in graph.edges})
        for graph in graphs
    )
    stored_rows = sum(
        len(normalized_edges(graph.edges, undirected=undirected)) for graph in graphs
    )
    missing_reverse = sum(
        sum(1 for source, target in set(graph.edges) if (target, source) not in set(graph.edges))
        for graph in graphs
    )
    summary = {
        "dataset": dataset_name,
        "graphs": len(graphs),
        "nodes": sum(graph_sizes),
        "nodes_min": min(graph_sizes),
        "nodes_mean": sum(graph_sizes) / len(graph_sizes),
        "nodes_max": max(graph_sizes),
        "source_edge_rows": source_rows,
        "unique_undirected_edge_pairs": undirected_pairs,
        "stored_edge_rows": stored_rows,
        "missing_reverse_source_rows": missing_reverse,
        "node_features": dict(
            sorted(Counter(value for graph in graphs for value in graph.node_features).items())
        ),
        "graph_labels_not_imported": dict(
            sorted(Counter(graph.graph_label for graph in graphs).items())
        ),
    }
    # This named section and its stable labels are parsed into
    # run_metadata/rule_manifest.json by run_factorbase_pipeline.py.
    print("=" * 60)
    print("SOURCE EDGE DIRECTION ANALYSIS")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Graphs analyzed: {len(graphs):,}")
    print(f"Graphs with edges: {sum(bool(graph.edges) for graph in graphs):,}")
    print(f"Source edge rows: {source_rows:,}")
    print(f"Unique undirected edge pairs: {undirected_pairs:,}")
    print(f"Rows missing reverse edge: {missing_reverse:,}")
    print("=" * 60)
    print(f"{dataset_name} DGL GIN DATASET SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


def quote_identifier(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def populate_mysql(
    args: argparse.Namespace,
    graphs: list[GINGraph],
) -> tuple[int, int]:
    from pymysql import connect

    connection = connect(
        host=args.mysql_host,
        port=args.mysql_port,
        user=args.mysql_user,
        password=args.mysql_password,
    )
    try:
        with connection.cursor() as cursor:
            database = quote_identifier(args.db_name)
            cursor.execute(f"DROP DATABASE IF EXISTS {database}")
            cursor.execute(
                f"CREATE DATABASE {database} "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            cursor.execute(f"USE {database}")
            cursor.execute("SET FOREIGN_KEY_CHECKS=1")
            cursor.execute(
                """
                CREATE TABLE nodes (
                    node_id INT PRIMARY KEY,
                    node_feature INT NOT NULL,
                    INDEX idx_node_feature (node_feature)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE edges (
                    source_node_id INT NOT NULL,
                    target_node_id INT NOT NULL,
                    PRIMARY KEY (source_node_id, target_node_id),
                    FOREIGN KEY (source_node_id) REFERENCES nodes(node_id),
                    FOREIGN KEY (target_node_id) REFERENCES nodes(node_id)
                )
                """
            )

            node_offset = 0
            inserted_edges = 0
            for graph_index, graph in enumerate(graphs):
                node_rows = [
                    (node_offset + local_id, feature)
                    for local_id, feature in enumerate(graph.node_features)
                ]
                cursor.executemany(
                    "INSERT INTO nodes (node_id, node_feature) VALUES (%s, %s)",
                    node_rows,
                )
                edge_rows = [
                    (node_offset + source, node_offset + target)
                    for source, target in normalized_edges(
                        graph.edges,
                        undirected=args.undirected,
                    )
                ]
                if edge_rows:
                    cursor.executemany(
                        "INSERT INTO edges (source_node_id, target_node_id) VALUES (%s, %s)",
                        edge_rows,
                    )
                    inserted_edges += len(edge_rows)
                node_offset += len(graph.node_features)
                if (graph_index + 1) % 100 == 0:
                    connection.commit()
                    print(f"Import progress: {graph_index + 1}/{len(graphs)} graphs")

            connection.commit()
            cursor.execute("SELECT COUNT(*) FROM nodes")
            node_count = int(cursor.fetchone()[0])
            cursor.execute("SELECT COUNT(*) FROM edges")
            edge_count = int(cursor.fetchone()[0])
            if node_count != node_offset or edge_count != inserted_edges:
                raise RuntimeError(
                    "Database verification failed: "
                    f"nodes={node_count}/{node_offset}, edges={edge_count}/{inserted_edges}"
                )
    finally:
        connection.close()

    print(f"MySQL database: {args.db_name}")
    print(f"Imported nodes: {node_count:,}")
    print(f"Imported edge rows: {edge_count:,}")
    return node_count, edge_count


def main(dataset_name: str) -> None:
    dataset_name = dataset_name.upper()
    args = parse_args(dataset_name)
    graphs = load_dataset(dataset_name)
    dataset_summary(dataset_name, graphs, undirected=args.undirected)
    if args.dry_run:
        print("Dry run complete; MySQL was not modified.")
        return
    populate_mysql(args, graphs)

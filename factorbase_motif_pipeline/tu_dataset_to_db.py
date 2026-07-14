#!/usr/bin/env python3
"""Shared TU Dortmund dataset importer used by the AIDS/ENZYMES builders.

The importer deliberately writes only ``nodes`` and ``edges`` source tables.
Graphs are flattened into one disconnected union, matching the convention used
by the existing GraphVAE/FactorBase importers. Graph IDs and graph labels are
reported for auditing but are not placed in the source tables, since doing so
would let FactorBase learn rules from dataset membership or prediction labels.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data_raw"
TU_DOWNLOAD_ROOT = "https://www.chrsmrrs.com/graphkerneldatasets"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    node_attribute_count: int
    has_node_labels: bool
    has_edge_labels: bool
    default_db_name: str


TU_DATASET_SPECS = {
    "AIDS": DatasetSpec(
        name="AIDS",
        node_attribute_count=4,
        has_node_labels=True,
        has_edge_labels=True,
        default_db_name="aids_undir_feat",
    ),
    "ENZYMES": DatasetSpec(
        name="ENZYMES",
        node_attribute_count=18,
        has_node_labels=True,
        has_edge_labels=False,
        default_db_name="enzymes_undir_feat",
    ),
}


@dataclass
class TUGraph:
    node_labels: list[int]
    node_attributes: list[tuple[float, ...]]
    edges: list[tuple[int, int, int | None]]
    graph_label: int


@dataclass
class PreparedDataset:
    graphs: list[TUGraph]
    attribute_rows: list[list[tuple[int | float, ...]]]
    attribute_sql_type: str | None
    quantile_thresholds: list[list[float]]
    source: str


def quote_identifier(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def parse_args(spec: DatasetSpec) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Load the TU Dortmund {spec.name} dataset into MySQL for FactorBase."
    )
    parser.add_argument("--db-name", default=None, help="MySQL database to replace/create")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            f"Directory containing {spec.name}_A.txt and companion files. "
            "If omitted, repository and TUDATASET_ROOT locations are searched."
        ),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail instead of downloading the official TU Dortmund ZIP when data is absent.",
    )
    edge_group = parser.add_mutually_exclusive_group()
    edge_group.add_argument(
        "--directed",
        action="store_true",
        help="Preserve exactly the edge rows present in the TU source files (default).",
    )
    edge_group.add_argument(
        "--undirected",
        action="store_true",
        help="Ensure A->B and B->A rows exist for every source edge pair.",
    )
    parser.add_argument(
        "--node-attribute-mode",
        choices=("quantile", "raw", "omit"),
        default="quantile",
        help=(
            "How to store continuous node attributes. quantile (default) converts each "
            "dimension into low-cardinality integer bins suitable for FactorBase; raw "
            "stores source floats; omit keeps only categorical node labels."
        ),
    )
    parser.add_argument(
        "--attribute-bins",
        type=int,
        default=8,
        help="Maximum number of bins per node-attribute dimension in quantile mode.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Optionally omit graphs with more than this many nodes.",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="Optionally keep only the first N source graphs (mainly for smoke tests).",
    )
    parser.add_argument("--mysql-host", default="localhost")
    parser.add_argument("--mysql-user", default="fbuser")
    parser.add_argument("--mysql-password", default="")
    parser.add_argument("--mysql-port", type=int, default=3306)
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="Print SQL for the selected feature mode and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load, validate, and summarize the dataset without changing MySQL.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional path for loader statistics and quantile thresholds.",
    )
    args = parser.parse_args()
    if not args.directed and not args.undirected:
        args.directed = True
    if args.attribute_bins < 2:
        parser.error("--attribute-bins must be at least 2")
    if args.max_nodes is not None and args.max_nodes < 1:
        parser.error("--max-nodes must be positive")
    if args.max_graphs is not None and args.max_graphs < 1:
        parser.error("--max-graphs must be positive")
    return args


def _split_csv(line: str) -> list[str]:
    return [value.strip() for value in line.strip().split(",")]


def _read_scalar_ints(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_float_rows(path: Path) -> list[tuple[float, ...]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(tuple(float(value) for value in _split_csv(line)))
        except ValueError as exc:
            raise ValueError(f"Invalid numeric row at {path}:{line_number}: {line}") from exc
    return rows


def _read_edge_rows(path: Path) -> list[tuple[int, int]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = _split_csv(line)
        if len(values) != 2:
            raise ValueError(f"Expected two node IDs at {path}:{line_number}: {line}")
        rows.append((int(values[0]), int(values[1])))
    return rows


def _required_path(dataset_dir: Path, dataset_name: str, suffix: str) -> Path:
    path = dataset_dir / f"{dataset_name}_{suffix}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Required TU dataset file not found: {path}")
    return path


def candidate_dataset_dirs(spec: DatasetSpec, requested: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if requested is not None:
        requested = requested.expanduser()
        candidates.extend([requested, requested / spec.name])
    env_root = os.environ.get("TUDATASET_ROOT")
    if env_root:
        root = Path(env_root).expanduser()
        candidates.extend([root / spec.name, root])
    candidates.extend(
        [
            DATA_ROOT / "Kernel_dataset" / spec.name,
            DATA_ROOT / "tu" / spec.name,
            Path.home() / ".cache" / "tu_datasets" / spec.name,
        ]
    )
    unique: list[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _is_dataset_dir(path: Path, spec: DatasetSpec) -> bool:
    return (path / f"{spec.name}_A.txt").is_file()


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as infile:
        for member in infile.infolist():
            output = (destination / member.filename).resolve()
            if output != destination_resolved and destination_resolved not in output.parents:
                raise RuntimeError(f"Unsafe ZIP member path: {member.filename}")
        infile.extractall(destination)


def download_dataset(spec: DatasetSpec) -> Path:
    destination_root = DATA_ROOT / "Kernel_dataset"
    destination_root.mkdir(parents=True, exist_ok=True)
    url = f"{TU_DOWNLOAD_ROOT}/{spec.name}.zip"
    print(f"Downloading official TU dataset: {url}")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        archive = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, archive)
        _safe_extract_zip(archive, destination_root)
    finally:
        archive.unlink(missing_ok=True)

    expected = destination_root / spec.name
    if _is_dataset_dir(expected, spec):
        return expected
    if _is_dataset_dir(destination_root, spec):
        expected.mkdir(parents=True, exist_ok=True)
        for source in destination_root.glob(f"{spec.name}_*.txt"):
            shutil.move(str(source), expected / source.name)
        return expected
    raise RuntimeError(f"Downloaded {url}, but could not find extracted {spec.name} files")


def find_dataset_dir(
    spec: DatasetSpec,
    requested: Path | None = None,
    allow_download: bool = True,
) -> Path:
    """Find a TU dataset for both training and database-import call sites."""

    searched = candidate_dataset_dirs(spec, requested)
    for candidate in searched:
        if _is_dataset_dir(candidate, spec):
            return candidate
    if allow_download:
        return download_dataset(spec)
    locations = "\n  - ".join(str(path) for path in searched)
    raise FileNotFoundError(f"Could not find {spec.name}. Searched:\n  - {locations}")


def resolve_dataset_dir(spec: DatasetSpec, args: argparse.Namespace) -> Path:
    return find_dataset_dir(
        spec,
        requested=args.data_dir,
        allow_download=not args.no_download,
    )


def load_tu_graphs(
    spec: DatasetSpec,
    dataset_dir: Path,
    max_nodes: int | None,
    max_graphs: int | None,
) -> tuple[list[TUGraph], dict]:
    indicator = _read_scalar_ints(_required_path(dataset_dir, spec.name, "graph_indicator"))
    graph_labels = _read_scalar_ints(_required_path(dataset_dir, spec.name, "graph_labels"))
    edge_pairs = _read_edge_rows(_required_path(dataset_dir, spec.name, "A"))

    if spec.has_node_labels:
        node_labels = _read_scalar_ints(_required_path(dataset_dir, spec.name, "node_labels"))
    else:
        node_labels = [0] * len(indicator)

    attribute_path = dataset_dir / f"{spec.name}_node_attributes.txt"
    if spec.node_attribute_count:
        if not attribute_path.exists():
            raise FileNotFoundError(
                f"{spec.name} is expected to have {spec.node_attribute_count} node attributes: "
                f"{attribute_path}"
            )
        node_attributes = _read_float_rows(attribute_path)
    else:
        node_attributes = [tuple() for _ in indicator]

    edge_label_path = dataset_dir / f"{spec.name}_edge_labels.txt"
    if spec.has_edge_labels:
        edge_labels = _read_scalar_ints(_required_path(dataset_dir, spec.name, "edge_labels"))
    else:
        edge_labels = [None] * len(edge_pairs)

    node_count = len(indicator)
    if len(node_labels) != node_count or len(node_attributes) != node_count:
        raise ValueError(
            f"Node row mismatch: indicators={node_count}, labels={len(node_labels)}, "
            f"attributes={len(node_attributes)}"
        )
    if len(edge_labels) != len(edge_pairs):
        raise ValueError(
            f"Edge row mismatch: adjacency={len(edge_pairs)}, labels={len(edge_labels)}"
        )
    if node_attributes and any(len(row) != spec.node_attribute_count for row in node_attributes):
        widths = sorted({len(row) for row in node_attributes})
        raise ValueError(
            f"Expected {spec.node_attribute_count} node attributes for {spec.name}; found widths {widths}"
        )

    graph_ids = sorted(set(indicator))
    if graph_ids != list(range(1, len(graph_labels) + 1)):
        raise ValueError(
            "TU graph indicators must be contiguous and 1-based; "
            f"found {graph_ids[:3]}...{graph_ids[-3:]}"
        )

    global_nodes_by_graph: dict[int, list[int]] = {graph_id: [] for graph_id in graph_ids}
    for global_index, graph_id in enumerate(indicator):
        global_nodes_by_graph[graph_id].append(global_index)

    local_node_id: dict[int, tuple[int, int]] = {}
    graphs = []
    for graph_id in graph_ids:
        global_indices = global_nodes_by_graph[graph_id]
        for local_index, global_index in enumerate(global_indices):
            local_node_id[global_index] = (graph_id, local_index)
        graphs.append(
            TUGraph(
                node_labels=[node_labels[index] for index in global_indices],
                node_attributes=[node_attributes[index] for index in global_indices],
                edges=[],
                graph_label=graph_labels[graph_id - 1],
            )
        )

    for row_index, ((src_one_based, dst_one_based), edge_label) in enumerate(
        zip(edge_pairs, edge_labels), 1
    ):
        src_global = src_one_based - 1
        dst_global = dst_one_based - 1
        if not 0 <= src_global < node_count or not 0 <= dst_global < node_count:
            raise ValueError(f"Edge row {row_index} references an invalid node")
        src_graph, src_local = local_node_id[src_global]
        dst_graph, dst_local = local_node_id[dst_global]
        if src_graph != dst_graph:
            raise ValueError(f"Cross-graph edge at adjacency row {row_index}")
        if src_local == dst_local:
            continue
        graphs[src_graph - 1].edges.append((src_local, dst_local, edge_label))

    original_graph_count = len(graphs)
    skipped_large = 0
    if max_nodes is not None:
        filtered = []
        for graph in graphs:
            if len(graph.node_labels) > max_nodes:
                skipped_large += 1
            else:
                filtered.append(graph)
        graphs = filtered
    if max_graphs is not None:
        graphs = graphs[:max_graphs]

    stats = {
        "source_graphs": original_graph_count,
        "loaded_graphs": len(graphs),
        "skipped_max_nodes": skipped_large,
        "source_nodes": node_count,
        "source_edge_rows": len(edge_pairs),
    }
    return graphs, stats


def quantile_thresholds(values: Sequence[float], bins: int) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    thresholds = []
    for bin_index in range(1, bins):
        position = max(0, min(len(ordered) - 1, math.ceil(bin_index * len(ordered) / bins) - 1))
        threshold = ordered[position]
        if not thresholds or threshold > thresholds[-1]:
            thresholds.append(threshold)
    return thresholds


def prepare_attributes(
    graphs: list[TUGraph],
    mode: str,
    bins: int,
    width: int,
) -> tuple[list[list[tuple[int | float, ...]]], str | None, list[list[float]]]:
    if mode == "omit" or width == 0:
        return [[tuple() for _ in graph.node_labels] for graph in graphs], None, []
    if mode == "raw":
        return [[tuple(row) for row in graph.node_attributes] for graph in graphs], "DOUBLE", []

    dimension_values = [list() for _ in range(width)]
    for graph in graphs:
        for row in graph.node_attributes:
            for dimension, value in enumerate(row):
                dimension_values[dimension].append(value)
    thresholds = [quantile_thresholds(values, bins) for values in dimension_values]

    prepared: list[list[tuple[int, ...]]] = []
    for graph in graphs:
        graph_rows = []
        for row in graph.node_attributes:
            graph_rows.append(
                tuple(bisect.bisect_right(thresholds[dimension], value) for dimension, value in enumerate(row))
            )
        prepared.append(graph_rows)
    return prepared, "INT", thresholds


def analyze_edge_direction(graphs: Iterable[TUGraph]) -> dict[str, int]:
    stats = {
        "graphs": 0,
        "graphs_with_edges": 0,
        "source_edge_rows": 0,
        "undirected_edge_pairs": 0,
        "missing_reverse_rows": 0,
        "typed_reverse_mismatches": 0,
    }
    for graph in graphs:
        stats["graphs"] += 1
        rows = {(src, dst): label for src, dst, label in graph.edges}
        if not rows:
            continue
        stats["graphs_with_edges"] += 1
        stats["source_edge_rows"] += len(rows)
        stats["undirected_edge_pairs"] += len({tuple(sorted(pair)) for pair in rows})
        for (src, dst), label in rows.items():
            if (dst, src) not in rows:
                stats["missing_reverse_rows"] += 1
            elif rows[(dst, src)] != label:
                stats["typed_reverse_mismatches"] += 1
    return stats


def node_attribute_columns(width: int, mode: str) -> list[str]:
    if mode == "omit":
        return []
    return [f"node_attr_{index:02d}" for index in range(width)]


def nodes_schema_sql(spec: DatasetSpec, attribute_mode: str) -> str:
    columns = ["node_id INT PRIMARY KEY"]
    indexes = []
    if spec.has_node_labels:
        columns.append("node_label INT NOT NULL")
        indexes.append("INDEX idx_node_label (node_label)")
    attr_type = "DOUBLE" if attribute_mode == "raw" else "INT"
    for name in node_attribute_columns(spec.node_attribute_count, attribute_mode):
        columns.append(f"{quote_identifier(name)} {attr_type} NOT NULL")
        indexes.append(f"INDEX idx_{name} ({quote_identifier(name)})")
    definitions = ",\n    ".join([*columns, *indexes])
    return f"CREATE TABLE nodes (\n    {definitions}\n)"


def edges_schema_sql(spec: DatasetSpec) -> str:
    columns = [
        "source_node_id INT NOT NULL",
        "target_node_id INT NOT NULL",
    ]
    indexes = []
    if spec.has_edge_labels:
        columns.append("edge_label INT NOT NULL")
        indexes.append("INDEX idx_edge_label (edge_label)")
    columns.extend(
        [
            "PRIMARY KEY (source_node_id, target_node_id)",
            "FOREIGN KEY (source_node_id) REFERENCES nodes(node_id)",
            "FOREIGN KEY (target_node_id) REFERENCES nodes(node_id)",
        ]
    )
    definitions = ",\n    ".join([*columns, *indexes])
    return f"CREATE TABLE edges (\n    {definitions}\n)"


def print_schema(spec: DatasetSpec, attribute_mode: str) -> None:
    print("NODES SQL:")
    print(nodes_schema_sql(spec, attribute_mode))
    print("\nEDGES SQL:")
    print(edges_schema_sql(spec))


def prepare_dataset(spec: DatasetSpec, args: argparse.Namespace) -> tuple[PreparedDataset, dict]:
    dataset_dir = resolve_dataset_dir(spec, args)
    graphs, load_stats = load_tu_graphs(
        spec,
        dataset_dir,
        max_nodes=args.max_nodes,
        max_graphs=args.max_graphs,
    )
    attribute_rows, attribute_sql_type, thresholds = prepare_attributes(
        graphs,
        mode=args.node_attribute_mode,
        bins=args.attribute_bins,
        width=spec.node_attribute_count,
    )
    prepared = PreparedDataset(
        graphs=graphs,
        attribute_rows=attribute_rows,
        attribute_sql_type=attribute_sql_type,
        quantile_thresholds=thresholds,
        source=str(dataset_dir),
    )
    return prepared, load_stats


def print_dataset_summary(
    spec: DatasetSpec,
    prepared: PreparedDataset,
    load_stats: dict,
    edge_mode: str,
    attribute_mode: str,
) -> dict:
    graphs = prepared.graphs
    edge_stats = analyze_edge_direction(graphs)
    node_count = sum(len(graph.node_labels) for graph in graphs)
    edge_count = sum(len(graph.edges) for graph in graphs)
    node_label_counts = Counter(label for graph in graphs for label in graph.node_labels)
    edge_label_counts = Counter(
        label for graph in graphs for _src, _dst, label in graph.edges if label is not None
    )
    graph_label_counts = Counter(graph.graph_label for graph in graphs)
    graph_sizes = [len(graph.node_labels) for graph in graphs]
    observed_attribute_values = [
        sorted({
            int(graph_rows[node_index][dimension])
            for graph_rows in prepared.attribute_rows
            for node_index in range(len(graph_rows))
        })
        for dimension in range(spec.node_attribute_count)
    ] if attribute_mode == "quantile" else []

    print("=" * 60)
    print("SOURCE EDGE DIRECTION ANALYSIS")
    print("=" * 60)
    print(f"Dataset: {spec.name}")
    print(f"Graphs analyzed: {edge_stats['graphs']:,}")
    print(f"Graphs with edges: {edge_stats['graphs_with_edges']:,}")
    print(f"Source edge rows: {edge_stats['source_edge_rows']:,}")
    print(f"Unique undirected edge pairs: {edge_stats['undirected_edge_pairs']:,}")
    print(f"Rows missing reverse edge: {edge_stats['missing_reverse_rows']:,}")
    print(f"Typed reverse mismatches: {edge_stats['typed_reverse_mismatches']:,}")
    print(f"Selected DB edge mode: {edge_mode.upper()}")

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Source: {prepared.source}")
    print(f"Graphs: {len(graphs):,} / {load_stats['source_graphs']:,}")
    print(f"Skipped by --max-nodes: {load_stats['skipped_max_nodes']:,}")
    print(f"Nodes: {node_count:,}")
    print(f"Source edge rows in retained graphs: {edge_count:,}")
    if graph_sizes:
        print(
            f"Nodes per graph: min={min(graph_sizes)}, "
            f"mean={sum(graph_sizes) / len(graph_sizes):.2f}, max={max(graph_sizes)}"
        )
    print(f"Node attribute mode: {attribute_mode}")
    if attribute_mode == "quantile":
        realized = [len(values) for values in observed_attribute_values]
        print(f"Realized bins per attribute: {realized}")
    print(f"Node labels: {dict(sorted(node_label_counts.items()))}")
    if spec.has_edge_labels:
        print(f"Edge labels: {dict(sorted(edge_label_counts.items()))}")
    print(f"Graph labels (not imported): {dict(sorted(graph_label_counts.items()))}")

    return {
        "dataset": spec.name,
        "source": prepared.source,
        "graphs": len(graphs),
        "nodes": node_count,
        "source_edge_rows": edge_count,
        "nodes_per_graph_min": min(graph_sizes) if graph_sizes else None,
        "nodes_per_graph_mean": sum(graph_sizes) / len(graph_sizes) if graph_sizes else None,
        "nodes_per_graph_max": max(graph_sizes) if graph_sizes else None,
        "edge_mode": edge_mode,
        "node_attribute_mode": attribute_mode,
        "attribute_bins": [len(values) for values in observed_attribute_values],
        "attribute_values": observed_attribute_values,
        "quantile_thresholds": prepared.quantile_thresholds,
        "node_label_counts": dict(sorted(node_label_counts.items())),
        "edge_label_counts": dict(sorted(edge_label_counts.items())),
        "graph_label_counts": dict(sorted(graph_label_counts.items())),
        "load_stats": load_stats,
        "edge_direction_stats": edge_stats,
    }


def deduplicated_edges(
    graph: TUGraph,
    edge_mode: str,
) -> list[tuple[int, int, int | None]]:
    rows: dict[tuple[int, int], int | None] = {}
    for src, dst, label in graph.edges:
        candidates = [(src, dst)] if edge_mode == "directed" else [(src, dst), (dst, src)]
        for pair in candidates:
            previous = rows.get(pair)
            if pair in rows and previous != label:
                raise ValueError(f"Conflicting edge labels for edge {pair}: {previous} vs {label}")
            rows[pair] = label
    return [(src, dst, rows[(src, dst)]) for src, dst in sorted(rows)]


def populate_mysql(
    spec: DatasetSpec,
    args: argparse.Namespace,
    prepared: PreparedDataset,
    edge_mode: str,
) -> tuple[int, int]:
    try:
        from pymysql import connect
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError("PyMySQL is required for database creation: pip install pymysql") from exc

    db_name = args.db_name or spec.default_db_name
    if not db_name.strip():
        raise ValueError("Database name cannot be empty")

    connection = connect(
        host=args.mysql_host,
        port=args.mysql_port,
        user=args.mysql_user,
        password=args.mysql_password,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"DROP DATABASE IF EXISTS {quote_identifier(db_name)}")
            cursor.execute(
                f"CREATE DATABASE {quote_identifier(db_name)} "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            cursor.execute(f"USE {quote_identifier(db_name)}")
            cursor.execute("SET FOREIGN_KEY_CHECKS=1")
            cursor.execute("SET sql_mode='STRICT_TRANS_TABLES'")
            cursor.execute(nodes_schema_sql(spec, args.node_attribute_mode))
            cursor.execute(edges_schema_sql(spec))

            attr_columns = node_attribute_columns(
                spec.node_attribute_count, args.node_attribute_mode
            )
            node_columns = ["node_id"]
            if spec.has_node_labels:
                node_columns.append("node_label")
            node_columns.extend(attr_columns)
            node_sql = (
                "INSERT INTO nodes ("
                + ", ".join(quote_identifier(name) for name in node_columns)
                + ") VALUES ("
                + ", ".join(["%s"] * len(node_columns))
                + ")"
            )

            edge_columns = ["source_node_id", "target_node_id"]
            if spec.has_edge_labels:
                edge_columns.append("edge_label")
            edge_sql = (
                "INSERT INTO edges ("
                + ", ".join(quote_identifier(name) for name in edge_columns)
                + ") VALUES ("
                + ", ".join(["%s"] * len(edge_columns))
                + ")"
            )

            global_node_id = 0
            inserted_edges = 0
            for graph_index, (graph, graph_attributes) in enumerate(
                zip(prepared.graphs, prepared.attribute_rows)
            ):
                if graph_index % 100 == 0:
                    print(f"Import progress: {graph_index}/{len(prepared.graphs)} graphs")
                graph_offset = global_node_id
                node_rows = []
                for local_node_id, (label, attributes) in enumerate(
                    zip(graph.node_labels, graph_attributes)
                ):
                    row: list[int | float] = [global_node_id]
                    if spec.has_node_labels:
                        row.append(label)
                    row.extend(attributes)
                    node_rows.append(tuple(row))
                    global_node_id += 1
                if node_rows:
                    cursor.executemany(node_sql, node_rows)

                edge_rows = []
                for src, dst, edge_label in deduplicated_edges(graph, edge_mode):
                    row = [graph_offset + src, graph_offset + dst]
                    if spec.has_edge_labels:
                        if edge_label is None:
                            raise ValueError("Missing required edge label")
                        row.append(edge_label)
                    edge_rows.append(tuple(row))
                if edge_rows:
                    cursor.executemany(edge_sql, edge_rows)
                    inserted_edges += len(edge_rows)
                if (graph_index + 1) % 100 == 0:
                    connection.commit()

            connection.commit()
            cursor.execute("SELECT COUNT(*) FROM nodes")
            node_count = int(cursor.fetchone()[0])
            cursor.execute("SELECT COUNT(*) FROM edges")
            edge_count = int(cursor.fetchone()[0])
            if node_count != global_node_id or edge_count != inserted_edges:
                raise RuntimeError(
                    f"Database verification failed: nodes={node_count}/{global_node_id}, "
                    f"edges={edge_count}/{inserted_edges}"
                )
    finally:
        connection.close()

    print("\n" + "=" * 60)
    print("DATABASE READY")
    print("=" * 60)
    print(f"Database: {db_name}")
    print(f"Dataset: {spec.name}")
    print(f"Graphs: {len(prepared.graphs):,}")
    print(f"Nodes: {node_count:,}")
    print(f"Edges: {edge_count:,}")
    return node_count, edge_count


def write_metadata(path: Path, metadata: dict) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Metadata written to: {path}")


def main(spec: DatasetSpec) -> None:
    args = parse_args(spec)
    if args.print_schema:
        print_schema(spec, args.node_attribute_mode)
        return

    edge_mode = "undirected" if args.undirected else "directed"
    prepared, load_stats = prepare_dataset(spec, args)
    metadata = print_dataset_summary(
        spec,
        prepared,
        load_stats,
        edge_mode=edge_mode,
        attribute_mode=args.node_attribute_mode,
    )
    metadata["database_name"] = args.db_name or spec.default_db_name
    if args.metadata_json is not None:
        write_metadata(args.metadata_json, metadata)
    if args.dry_run:
        print("\nDry run complete; MySQL was not modified.")
        return
    populate_mysql(spec, args, prepared, edge_mode)

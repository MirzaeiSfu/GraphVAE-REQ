#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QM9 Dataset to MySQL Database Converter - FINAL VERSION
"""

import argparse
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data_raw"
DEFAULT_EDGE_MODE = "directed"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load the QM9 dataset into a MySQL database for FactorBase."
    )
    parser.add_argument("--db-name", help="MySQL database name to create")

    edge_group = parser.add_mutually_exclusive_group()
    edge_group.add_argument(
        "--directed",
        action="store_true",
        help="Store exactly the edge directions exposed by the source graph",
    )
    edge_group.add_argument(
        "--undirected",
        action="store_true",
        help="Store both directions for each edge pair",
    )
    args = parser.parse_args()
    if not args.directed and not args.undirected:
        if DEFAULT_EDGE_MODE == "directed":
            args.directed = True
        elif DEFAULT_EDGE_MODE == "undirected":
            args.undirected = True
    return args


args = parse_args()

EDGE_MODE_LABELS = {
    "directed": "DIRECTED (source graph edge directions)",
    "undirected": "UNDIRECTED (A->B and B->A for each edge pair)",
}

from torch_geometric.datasets import QM9
import torch


def analyze_source_edge_direction(dataset):
    stats = {
        "graphs": len(dataset),
        "graphs_with_edges": 0,
        "source_edge_rows": 0,
        "source_typed_edge_rows": 0,
        "undirected_edge_pairs": 0,
        "missing_reverse_rows": 0,
        "missing_typed_reverse_rows": 0,
        "graphs_with_missing_reverse": 0,
    }

    for mol in dataset:
        if mol.edge_attr is None or mol.edge_attr.shape[0] == 0:
            continue

        edge_index = mol.edge_index
        edge_attr = mol.edge_attr
        edge_rows = set()
        typed_edge_rows = set()

        for edge_id in range(edge_index.shape[1]):
            src = int(edge_index[0, edge_id].item())
            dst = int(edge_index[1, edge_id].item())
            if src == dst:
                continue
            bond_type = int(torch.argmax(edge_attr[edge_id]).item())
            edge_rows.add((src, dst))
            typed_edge_rows.add((src, dst, bond_type))

        if not edge_rows:
            continue

        missing_reverse_rows = sum(
            1 for src, dst in edge_rows if (dst, src) not in edge_rows
        )
        missing_typed_reverse_rows = sum(
            1
            for src, dst, bond_type in typed_edge_rows
            if (dst, src, bond_type) not in typed_edge_rows
        )

        stats["graphs_with_edges"] += 1
        stats["source_edge_rows"] += len(edge_rows)
        stats["source_typed_edge_rows"] += len(typed_edge_rows)
        stats["undirected_edge_pairs"] += len({tuple(sorted(edge)) for edge in edge_rows})
        stats["missing_reverse_rows"] += missing_reverse_rows
        stats["missing_typed_reverse_rows"] += missing_typed_reverse_rows
        if missing_reverse_rows or missing_typed_reverse_rows:
            stats["graphs_with_missing_reverse"] += 1

    return stats


def print_source_edge_direction_analysis(dataset_name, stats, edge_mode):
    print("=" * 60)
    print("SOURCE EDGE DIRECTION ANALYSIS")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Graphs analyzed: {stats['graphs']:,}")
    print(f"Graphs with edges: {stats['graphs_with_edges']:,}")
    print(f"Source edge rows: {stats['source_edge_rows']:,}")
    print(f"Unique undirected edge pairs: {stats['undirected_edge_pairs']:,}")
    print(f"Rows missing reverse edge: {stats['missing_reverse_rows']:,}")
    print(f"Typed rows missing reverse edge with same bond_type: {stats['missing_typed_reverse_rows']:,}")

    has_edges = stats["source_edge_rows"] > 0
    source_is_bidirectional = (
        has_edges
        and stats["missing_reverse_rows"] == 0
        and stats["missing_typed_reverse_rows"] == 0
    )

    if source_is_bidirectional:
        print("Source appears bidirectional/undirected: every edge row has a reverse row.")
        if edge_mode == "directed":
            print(
                "WARNING: --directed preserves source rows, but this source is already "
                "bidirectional; the edge table should match --undirected."
            )
        else:
            print(
                "NOTE: --undirected will ensure reverse rows, but the source already has "
                "them; no extra edge rows are expected."
            )
    elif has_edges:
        print("Source contains one-way edge rows.")
        if edge_mode == "directed":
            print("NOTE: --directed will preserve those one-way source rows.")
        else:
            print("NOTE: --undirected will add reverse rows for one-way source edges.")
    else:
        print("Source has no edge rows to analyze.")
    print()

# ============================================
# LOAD QM9 DATASET
# ============================================
print("=" * 60)
print("LOADING QM9 DATASET")
print("=" * 60)
dataset_root = DATA_ROOT / "QM9"
dataset = QM9(root=str(dataset_root))
print(f"Loaded {len(dataset):,} molecules\n")

# ============================================
# ASK USER FOR GRAPH TYPE
# ============================================
print("=" * 60)
print("GRAPH DIRECTION CONFIGURATION")
print("=" * 60)
if args.directed:
    edge_mode = "directed"
    print("Selected: DIRECTED\n")
elif args.undirected:
    edge_mode = "undirected"
    print("Selected: UNDIRECTED\n")
else:
    while True:
        choice = input(
            "Edge storage mode?\n"
            "  1 - DIRECTED (store exactly the source edge directions)\n"
            "  2 - UNDIRECTED (store A->B and B->A for each edge pair)\n"
            "Choice: "
        ).strip()
        if choice == "1":
            edge_mode = "directed"
            print("Selected: DIRECTED\n")
            break
        elif choice == "2":
            edge_mode = "undirected"
            print("Selected: UNDIRECTED\n")
            break
        else:
            print("Please enter 1 or 2.")

source_edge_stats = analyze_source_edge_direction(dataset)
print_source_edge_direction_analysis(
    "QM9",
    source_edge_stats,
    edge_mode,
)

# ============================================
# ANALYZE DATA DISTRIBUTIONS
# ============================================
print("=" * 60)
print("ANALYZING DATA DISTRIBUTIONS")
print("=" * 60)

atom_type_dist = defaultdict(int)
bond_type_dist = defaultdict(int)

sample_size = min(10000, len(dataset))
for mol_id in range(sample_size):
    mol = dataset[mol_id]
    for node_id in range(mol.num_nodes):
        features = mol.x[node_id].tolist()
        atom_type = features[0:5].index(max(features[0:5])) + 1
        atom_type_dist[atom_type] += 1
    if mol.edge_attr is not None and mol.edge_attr.shape[0] > 0:
        for edge_id in range(mol.edge_attr.shape[0]):
            bond_type = torch.argmax(mol.edge_attr[edge_id]).item()
            bond_type_dist[bond_type] += 1

print(f"Analyzed {sample_size:,} molecules")
atom_names = {
    1: "H (Hydrogen)",
    2: "C (Carbon)",
    3: "N (Nitrogen)",
    4: "O (Oxygen)",
    5: "F (Fluorine)",
}
print("\nAtom type distribution (sample):")
for at in sorted(atom_type_dist.keys()):
    print(f"  Type {at} - {atom_names[at]:20s}: {atom_type_dist[at]:,}")

bond_names = {
    0: "Single bonds",
    1: "Double bonds",
    2: "Triple bonds",
    3: "Aromatic bonds",
}
print("\nBond type distribution (sample):")
for bt in sorted(bond_type_dist.keys()):
    print(f"  Type {bt} - {bond_names[bt]:20s}: {bond_type_dist[bt]:,}")

# ============================================
# DATABASE CONFIGURATION
# ============================================
print("\n" + "=" * 60)
print("DATABASE CONFIGURATION")
print("=" * 60)
db_name = args.db_name if args.db_name else input("Enter the database name: ").strip()

from pymysql import connect

db_params = {
    "host": "localhost",
    "user": "fbuser",
    "password": "",
}

print("\n" + "=" * 60)
print("CONNECTING TO DATABASE")
print("=" * 60)

# Connect WITHOUT specifying a database first
connection = connect(**db_params)
cursor = connection.cursor()

# Create the database immediately after getting the name
cursor.execute(f"DROP DATABASE IF EXISTS `{db_name}`")
cursor.execute(f"CREATE DATABASE `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
cursor.execute(f"USE `{db_name}`")
cursor.execute("SET FOREIGN_KEY_CHECKS=1;")
cursor.execute("SET sql_mode='STRICT_TRANS_TABLES';")
print(f"Connected to MySQL | Database: {db_name} created and selected\n")

# ============================================
# CREATE SCHEMA
# ============================================
print("=" * 60)
print("CREATING DATABASE SCHEMA")
print("=" * 60)

cursor.execute(
    """
CREATE TABLE IF NOT EXISTS nodes (
    node_id INT PRIMARY KEY,
    atom_type INT NOT NULL,
    num_hydrogens INT NOT NULL,
    INDEX idx_atom_type (atom_type)
)
"""
)
print("NODES TABLE created")
print("   - atom_type: INT (1=H, 2=C, 3=N, 4=O, 5=F)")

cursor.execute(
    """
CREATE TABLE IF NOT EXISTS edges (
    source_node_id INT NOT NULL,
    target_node_id INT NOT NULL,
    bond_type INT NOT NULL,
    PRIMARY KEY (source_node_id, target_node_id),
    FOREIGN KEY (source_node_id) REFERENCES nodes(node_id),
    FOREIGN KEY (target_node_id) REFERENCES nodes(node_id),
    INDEX idx_bond_type (bond_type)
)
"""
)
print("EDGES TABLE created")
print(f"Edge mode: {EDGE_MODE_LABELS[edge_mode]}")

# ============================================
# POPULATE DATABASE
# ============================================
print("\n" + "=" * 60)
print("POPULATING DATABASE WITH QM9 DATA")
print("=" * 60)
print("This will take several minutes...\n")

global_node_id = 0

atom_type_counts = defaultdict(int)
num_hydrogens_counts = defaultdict(int)
bond_type_counts = defaultdict(int)

for mol_id, mol in enumerate(dataset):
    if mol_id % 1000 == 0:
        print(f"Progress: {mol_id}/{len(dataset)} molecules ({mol_id/len(dataset)*100:.1f}%)")

    molecule_node_offset = global_node_id

    # INSERT NODES
    node_rows = []
    for local_node_id in range(mol.num_nodes):
        features = mol.x[local_node_id].tolist()
        one_hot_atom = features[0:5]
        atom_type = one_hot_atom.index(max(one_hot_atom)) + 1
        num_hydrogens = int(min(features[10], 3)) + 1

        atom_type_counts[atom_type] += 1
        num_hydrogens_counts[num_hydrogens] += 1

        node_rows.append((global_node_id, atom_type, num_hydrogens))
        global_node_id += 1

    cursor.executemany(
        """
        INSERT INTO nodes (node_id, atom_type, num_hydrogens)
        VALUES (%s, %s, %s)
        """,
        node_rows,
    )

    # INSERT EDGES
    edge_rows = []
    if mol.edge_attr is not None and mol.edge_attr.shape[0] > 0:
        edge_index = mol.edge_index
        edge_attr = mol.edge_attr
        seen_edges = set()

        for edge_id in range(edge_index.shape[1]):
            src_local = edge_index[0, edge_id].item()
            dst_local = edge_index[1, edge_id].item()
            src_global = molecule_node_offset + src_local
            dst_global = molecule_node_offset + dst_local
            bond_type = torch.argmax(edge_attr[edge_id]).item()

            if edge_mode == "directed":
                edge_key = (src_global, dst_global)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edge_rows.append((src_global, dst_global, bond_type))
                bond_type_counts[bond_type] += 1
                continue

            inserted_rows = 0
            for edge_key in ((src_global, dst_global), (dst_global, src_global)):
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edge_rows.append((edge_key[0], edge_key[1], bond_type))
                inserted_rows += 1
            bond_type_counts[bond_type] += inserted_rows

    if edge_rows:
        cursor.executemany(
            """
            INSERT INTO edges (source_node_id, target_node_id, bond_type)
            VALUES (%s, %s, %s)
            """,
            edge_rows,
        )

    if (mol_id + 1) % 1000 == 0:
        connection.commit()

connection.commit()

# ============================================
# PRINT STATISTICS
# ============================================
print("\n" + "=" * 60)
print("DATABASE POPULATION COMPLETE!")
print("=" * 60)

cursor.execute("SELECT COUNT(*) FROM nodes")
node_count = cursor.fetchone()[0]
print(f"\nNODES: {node_count:,} total")

atom_names_map = {
    1: "H (Hydrogen)",
    2: "C (Carbon)",
    3: "N (Nitrogen)",
    4: "O (Oxygen)",
    5: "F (Fluorine)",
}
print("\nAtom type distribution:")
for atom_type in sorted(atom_type_counts.keys()):
    count = atom_type_counts[atom_type]
    pct = count / node_count * 100
    print(f"  Type {atom_type} - {atom_names_map[atom_type]:20s}: {count:,} ({pct:.1f}%)")

print("\nNum hydrogens distribution (capped at 3):")
for num_h in sorted(num_hydrogens_counts.keys()):
    count = num_hydrogens_counts[num_h]
    pct = count / node_count * 100
    print(f"  {num_h} hydrogens: {count:,} ({pct:.1f}%)")

cursor.execute("SELECT COUNT(*) FROM edges")
edge_count = cursor.fetchone()[0]
print(f"\nEDGES: {edge_count:,} total")
if edge_mode == "directed":
    print("  (DIRECTED: exact source edge directions)")
else:
    print(f"  (UNDIRECTED: 2x edge pairs | Edge pairs: {edge_count//2:,})")

bond_names_map = {
    0: "Single bonds",
    1: "Double bonds",
    2: "Triple bonds",
    3: "Aromatic bonds",
}
total_bond_rows = sum(bond_type_counts.values())
print("\nBond type distribution:")
for bond_type in sorted(bond_type_counts.keys()):
    count = bond_type_counts[bond_type]
    pct = count / total_bond_rows * 100 if total_bond_rows else 0
    print(f"  Type {bond_type} - {bond_names_map[bond_type]:20s}: {count:,} ({pct:.1f}%)")

# Sample nodes
print("\nSAMPLE DATA (First 10 nodes):")
cursor.execute("SELECT * FROM nodes LIMIT 10")
rows = cursor.fetchall()
print("  node_id | atom_type | num_hydrogens")
print("  " + "-" * 38)
for row in rows:
    atom_symbol = [None, "H", "C", "N", "O", "F"][row[1]]
    print(f"  {row[0]:7d} | {row[1]:9d} ({atom_symbol}) | {row[2]:13d}")

# Sample edges
print("\nSAMPLE DATA (First 10 edges):")
cursor.execute("SELECT * FROM edges LIMIT 10")
rows = cursor.fetchall()
print("  source_node_id | target_node_id | bond_type")
print("  " + "-" * 46)
for row in rows:
    bond_name = ["Single", "Double", "Triple", "Aromatic"][row[2]]
    print(f"  {row[0]:14d} | {row[1]:14d} | {row[2]:9d} ({bond_name})")

# Schema verification
print("\nSCHEMA VERIFICATION:")
cursor.execute("SHOW CREATE TABLE edges")
result = cursor.fetchone()
checks = [
    ("PRIMARY KEY (`source_node_id`,`target_node_id`)", "Composite primary key"),
    ("FOREIGN KEY (`source_node_id`) REFERENCES `nodes`", "source_node_id foreign key"),
    ("FOREIGN KEY (`target_node_id`) REFERENCES `nodes`", "target_node_id foreign key"),
]
for pattern, label in checks:
    status = "OK" if pattern in result[1] else "MISSING"
    print(f"  [{status}] {label}")

# Bidirectional verification
if edge_mode == "undirected":
    print("\nBIDIRECTIONAL EDGE VERIFICATION:")
    cursor.execute(
        """
        SELECT COUNT(*) FROM edges e1
        WHERE EXISTS (
            SELECT 1 FROM edges e2
            WHERE e2.source_node_id = e1.target_node_id
            AND e2.target_node_id = e1.source_node_id
            AND e2.bond_type = e1.bond_type
        )
    """
    )
    bidirectional_count = cursor.fetchone()[0]
    print(f"  Edges with reverse direction: {bidirectional_count:,} / {edge_count:,}")
    if bidirectional_count == edge_count:
        print("  ALL edges are bidirectional")
    else:
        print("  WARNING: Some edges missing reverse direction")

cursor.close()
connection.close()

print("\n" + "=" * 60)
print("DATABASE READY!")
print("=" * 60)
print(f"  Database : {db_name}")
print(f"  Mode     : {edge_mode.upper()}")
print(f"  Nodes    : {node_count:,}")
print(f"  Edges    : {edge_count:,}")

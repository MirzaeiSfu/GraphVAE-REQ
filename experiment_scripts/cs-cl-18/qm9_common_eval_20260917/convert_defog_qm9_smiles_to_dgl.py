#!/usr/bin/env python3
"""Convert DeFoG QM9 SMILES to GraphVAE-compatible attributed DGL graphs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import dgl
import torch
from rdkit import Chem


ATOM_INDEX = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}  # H, C, N, O, F
BOND_INDEX = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def convert(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("RDKit rejected SMILES")
    node_count = molecule.GetNumAtoms()
    node_attr = torch.zeros((node_count, 9), dtype=torch.float32)
    for index, atom in enumerate(molecule.GetAtoms()):
        atomic_number = atom.GetAtomicNum()
        if atomic_number not in ATOM_INDEX:
            raise ValueError(f"unsupported atomic number {atomic_number}")
        node_attr[index, ATOM_INDEX[atomic_number]] = 1.0
        hydrogen_count = min(int(atom.GetTotalNumHs(includeNeighbors=True)), 3)
        node_attr[index, 5 + hydrogen_count] = 1.0

    sources, targets, edge_rows = [], [], []
    for bond in molecule.GetBonds():
        if bond.GetBondType() not in BOND_INDEX:
            raise ValueError(f"unsupported bond type {bond.GetBondType()}")
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        onehot = torch.zeros(3, dtype=torch.float32)
        onehot[BOND_INDEX[bond.GetBondType()]] = 1.0
        sources.extend((u, v))
        targets.extend((v, u))
        edge_rows.extend((onehot, onehot.clone()))

    graph = dgl.graph((sources, targets), num_nodes=node_count)
    graph.ndata["attr"] = node_attr
    graph.edata["attr"] = (
        torch.stack(edge_rows) if edge_rows else torch.zeros((0, 3), dtype=torch.float32)
    )
    return graph


def main():
    args = parse_args()
    rows = list(csv.DictReader(args.smiles.open()))
    graphs, failures = [], []
    for index, row in enumerate(rows):
        try:
            graphs.append(convert(row["SMILES"]))
        except Exception as exc:
            failures.append({"index": index, "smiles": row.get("SMILES"), "error": str(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dgl.save_graphs(str(args.output), graphs)
    metadata = {
        "generator": "DeFoG",
        "training_seed": args.seed,
        "input_smiles": str(args.smiles.resolve()),
        "input_rows": len(rows),
        "accepted_graphs": len(graphs),
        "rejected_graphs": len(failures),
        "rejections": failures,
        "node_schema": "atom_type onehot[H,C,N,O,F] + num_h onehot[0,1,2,3]",
        "edge_schema": "bond_type onehot[single,double,triple]",
    }
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

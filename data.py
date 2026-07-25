

from __future__ import annotations

import random
from pathlib import Path

from typing import Dict, List, Optional
import networkx as nx
import numpy as np
import torch
from scipy.sparse import *
from  Synthatic_graph_generator import *
# from util import *
import os
import pickle as pkl
import scipy.sparse as sp
import warnings

import dgl as dgl
from dataset_feature_utils.grid_features import (
    EDGE_AXIS_TO_ID,
    EDGE_SQUARE_COUNT_LABELS,
    compute_boundary_depth,
    compute_edge_axis,
    compute_edge_boundary_band,
    compute_edge_square_count,
    get_grid_dimensions,
)
import dataset_feature_utils.lobster_features as lobster_features
import dataset_feature_utils.triangular_grid_features as triangular_grid_features
from factorbase_motif_pipeline.tu_dataset_to_db import (
    TU_DATASET_SPECS,
    deduplicated_edges,
    find_dataset_dir,
    load_tu_graphs,
    prepare_attributes,
)

# import ogb


OGB_MOL_NODE_FEATURE_NAMES = [
    "atomic_num",
    "chirality",
    "degree",
    "formal_charge",
    "num_h",
    "num_radical_e",
    "hybridization",
    "is_aromatic",
    "is_in_ring",
]

OGB_MOL_EDGE_FEATURE_NAMES = [
    "bond_type",
    "bond_stereo",
    "is_conjugated",
]


def get_data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "data_raw")).expanduser()


def data_path(*parts: str) -> Path:
    return get_data_dir().joinpath(*parts)


def load_gin_dataset(name: str):
    try:
        return dgl.data.GINDataset(
            name=name,
            self_loop=False,
            raw_dir=str(data_path("dgl")),
        )
    except TypeError:  # pragma: no cover - depends on installed DGL version
        return dgl.data.GINDataset(name=name, self_loop=False)


def _candidate_ogb_roots() -> List[Path]:
    """Return OGB roots, preferring explicit/local data before download paths."""

    candidates: List[Path] = []
    env_root = os.environ.get("OGB_DATA_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.extend([
        data_path("ogb"),
        Path("dataset"),
        # Helpful for this workspace: GraphVAE-MM already ships a local
        # ogbg_molbbbp copy. This is only a fallback; a copied/explicit
        # REQ data root remains preferred.
        Path("../GraphVAE-MM/dataset"),
    ])

    seen = set()
    unique = []
    for candidate in candidates:
        resolved_key = str(candidate)
        if resolved_key in seen:
            continue
        seen.add(resolved_key)
        unique.append(candidate)
    return unique


def load_ogbg_molbbbp_dataset():
    from ogb.graphproppred import DglGraphPropPredDataset

    last_error = None
    for root in _candidate_ogb_roots():
        dataset_dir = root / "ogbg_molbbbp"
        try:
            if dataset_dir.exists() or root == data_path("ogb"):
                print(f"Loading ogbg-molbbbp from OGB root: {root}")
                return DglGraphPropPredDataset(
                    name="ogbg-molbbbp",
                    root=str(root),
                )
        except Exception as exc:
            last_error = exc
            print(f"Warning: failed to load ogbg-molbbbp from {root}: {exc}")

    message = (
        "Could not load ogbg-molbbbp. Put the OGB dataset at one of these "
        "roots, or set OGB_DATA_ROOT: "
        + ", ".join(str(p) for p in _candidate_ogb_roots())
    )
    if last_error is not None:
        message += f". Last error: {last_error}"
    raise RuntimeError(message)


def _dgl_graph_to_csr(graph):
    try:
        return csr_matrix(graph.adjacency_matrix().to_dense().cpu().numpy())
    except Exception:
        return csr_matrix(graph.adj().to_dense().cpu().numpy())


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# load cora, citeseer and pubmed dataset
def Graph_load(dataset = 'cora'):
    '''
    Load a single graph dataset
    :param dataset: dataset name
    :return:
    '''
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    kernel_dataset_dir = data_path("Kernel_dataset")
    for i in range(len(names)):
        load = pkl.load(open(kernel_dataset_dir / f"ind.{dataset}.{names[i]}", 'rb'), encoding='latin1')
        # print('loaded')
        objects.append(load)
        # print(load)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(kernel_dataset_dir / f"ind.{dataset}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    return adj, features, G

def graph_load_batch(data_dir,
                     min_num_nodes=20,
                     max_num_nodes=1000,
                     name='ENZYMES',
                     node_attributes=True,
                     graph_labels=True):
  '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
  print('Loading graph dataset: ' + str(name))
  G = nx.Graph()
  # load data
  path = os.path.join(data_dir, name)
  data_adj = np.loadtxt(
      os.path.join(path, '{}_A.txt'.format(name)), delimiter=',').astype(int)
  if node_attributes:
    data_node_att = np.loadtxt(
        os.path.join(path, '{}_node_attributes.txt'.format(name)),
        delimiter=',')
  data_node_label = np.loadtxt(
      os.path.join(path, '{}_node_labels.txt'.format(name)),
      delimiter=',').astype(int)
  data_graph_indicator = np.loadtxt(
      os.path.join(path, '{}_graph_indicator.txt'.format(name)),
      delimiter=',').astype(int)
  if graph_labels:
    data_graph_labels = np.loadtxt(
        os.path.join(path, '{}_graph_labels.txt'.format(name)),
        delimiter=',').astype(int)

  data_tuple = list(map(tuple, data_adj))
  # print(len(data_tuple))
  # print(data_tuple[0])

  # add edges
  G.add_edges_from(data_tuple)
  # add node attributes
  for i in range(data_node_label.shape[0]):
    if node_attributes:
      G.add_node(i + 1, feature=data_node_att[i])
    G.add_node(i + 1, label=data_node_label[i])
  G.remove_nodes_from(list(nx.isolates(G)))

  # remove self-loop
  G.remove_edges_from(nx.selfloop_edges(G))

  # print(G.number_of_nodes())
  # print(G.number_of_edges())

  # split into graphs
  graph_num = data_graph_indicator.max()
  node_list = np.arange(data_graph_indicator.shape[0]) + 1
  graphs = []
  max_nodes = 0
  for i in range(graph_num):
    # find the nodes for each graph
    nodes = node_list[data_graph_indicator == i + 1]
    G_sub = G.subgraph(nodes)
    G_sub = nx.Graph((G_sub))
    if graph_labels:
      G_sub.graph['label'] = data_graph_labels[i]
    # print('nodes', G_sub.number_of_nodes())
    # print('edges', G_sub.number_of_edges())
    # print('label', G_sub.graph)
    if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes(
    ) <= max_num_nodes:
      graphs.append(G_sub)
      if G_sub.number_of_nodes() > max_nodes:
        max_nodes = G_sub.number_of_nodes()
      # print(G_sub.number_of_nodes(), 'i', i)
      # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
      # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
  print('Loaded')
  list_adj = []
  list_x= []
  list_label = []
  for G in graphs:
      list_adj.append(nx.adjacency_matrix(G))
      list_x.append(None)
      list_label.append(G.graph['label']-1)
  return list_adj, list_x, list_label


class Datasets():
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_adjs, self_for_none, list_Xs, graphlabels=None, padding=True,
                 Max_num=None, set_diag_of_isol_Zer=True,
                 list_node_onehot=None, list_edge_onehot=None):

        if Max_num != 0 and Max_num is not None:
            list_adjs, graphlabels, list_Xs = self.remove_largergraphs(
                list_adjs, graphlabels, list_Xs, Max_num)

        self.set_diag_of_isol_Zer = set_diag_of_isol_Zer
        self.paading               = padding
        self.list_Xs               = list_Xs
        self.labels                = graphlabels
        self.list_adjs             = list_adjs
        self.list_node_onehot      = list_node_onehot
        self.list_edge_onehot      = list_edge_onehot
        self.motif_counts          = None
        self.motif_matrices        = None
        self.motif_matrix_mask     = None
        self.motif_statistics      = None
        self.motif_statistic_mask  = None
        self.motif_histogram_spec  = None
        self.motif_full_matrices   = None
        self.motif_full_matrix_mask = None
        self.toatl_num_of_edges    = 0
        self.max_num_nodes         = 0

        for i, adj in enumerate(list_adjs):
            list_adjs[i] = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            list_adjs[i] += sp.eye(list_adjs[i].shape[0])
            if self.max_num_nodes < adj.shape[0]:
                self.max_num_nodes = adj.shape[0]
            self.toatl_num_of_edges += adj.sum().item()

        if Max_num is not None:
            self.max_num_nodes = Max_num

        self.processed_Xs          = []
        self.processed_adjs        = []
        self.processed_node_onehot = []
        self.processed_edge_onehot = []
        self.num_of_edges          = []

        for i in range(self.__len__()):
            a, x, n, _, node_oh, edge_oh = self.process(i, self_for_none)
            self.processed_Xs.append(x)
            self.processed_adjs.append(a)
            self.processed_node_onehot.append(node_oh)
            self.processed_edge_onehot.append(edge_oh)
            self.num_of_edges.append(n)

        self.feature_size     = self.processed_Xs[0].shape[-1]
        self.adj_s            = []
        self.x_s              = []
        self.node_onehot_s    = []
        self.edge_onehot_s    = []
        self.num_nodes        = []
        self.subgraph_indexes = []
        self.featureList      = None

    
    def remove_largergraphs(self, adjs, labels, Xs, max_size):
        processed_adjs   = []
        processed_labels = []
        processed_Xs     = []
        for i in range(len(adjs)):
            if adjs[i].shape[0] <= max_size:
                processed_adjs.append(adjs[i])
                if labels is not None:
                    processed_labels.append(labels[i])
                if Xs is not None:
                    processed_Xs.append(Xs[i])
        return processed_adjs, processed_labels, processed_Xs

    def get(self):
        indexces = list(range(self.__len__()))
        return ([self.processed_adjs[i] for i in indexces],
                [self.processed_Xs[i]   for i in indexces])

    def set_features(self, some_feature):
        self.featureList = some_feature

    def get_adj_list(self):
        return self.adj_s

    def get__(self, from_, to_, self_for_none, bfs=None, ignore_isolate_nodes=False):
        adj_s            = []
        x_s              = []
        num_nodes        = []
        subgraph_indexes = []

        if bfs is None:
            graphfeatures = []
            for element in self.featureList:
                graphfeatures.append(element[from_:to_])
            return (self.adj_s[from_:to_], self.x_s[from_:to_],
                    self.num_nodes[from_:to_], self.subgraph_indexes[from_:to_],
                    graphfeatures)

        for i in range(from_, to_):
            adj, x, num_node, indexes, _, _ = self.process(
                i, self_for_none, None, bfs, ignore_isolate_nodes)
            adj_s.append(adj)
            x_s.append(x)
            num_nodes.append(num_node)
            subgraph_indexes.append(indexes)

        return adj_s, x_s, num_nodes, subgraph_indexes

    def get_feature_targets(self, from_, to_):
        """
        Return the padded node/edge one-hot targets for the same graph slice
        used by get__(). These targets are the structure expected by the
        feature decoders: node (B, N_max, D), edge (B, C, N_max, N_max).
        """
        actual_to = min(to_, len(self.list_adjs))

        node_targets = None
        if self.node_onehot_s:
            node_targets = self.node_onehot_s[from_:actual_to]

        edge_targets = None
        if self.edge_onehot_s:
            edge_targets = self.edge_onehot_s[from_:actual_to]

        return node_targets, edge_targets, actual_to

    def get_max_degree(self):
        return np.max([adj.sum(-1) for adj in self.processed_adjs])

    def processALL(self, self_for_none, bfs=None, ignore_isolate_nodes=False):
        self.adj_s            = []
        self.x_s              = []
        self.node_onehot_s    = []
        self.edge_onehot_s    = []
        self.num_nodes        = []
        self.subgraph_indexes = []

        for i in range(len(self.list_adjs)):
            adj, x, num_node, indexes, node_oh, edge_oh = self.process(
                i, self_for_none, None, bfs, ignore_isolate_nodes)
            self.adj_s.append(adj)
            self.x_s.append(x)
            self.node_onehot_s.append(node_oh)
            self.edge_onehot_s.append(edge_oh)
            self.num_nodes.append(num_node)
            self.subgraph_indexes.append(indexes)

    def __len__(self):
        return len(self.list_adjs)

    def process(self, index, self_for_none, padded_to=None,
                bfs_max_length=None, ignore_isolate_nodes=True):

        if bfs_max_length is not None:
            bfs_max_length = min(bfs_max_length, self.max_num_nodes)

        num_nodes = self.list_adjs[index].shape[0]
        if self.paading:
            max_num_nodes = self.max_num_nodes if padded_to is None else padded_to
        else:
            max_num_nodes = num_nodes

        adj_padded = lil_matrix((max_num_nodes, max_num_nodes))
        if max_num_nodes == num_nodes:
            adj_padded = lil_matrix(self.list_adjs[index], dtype=np.int8)
        else:
            adj_padded[:num_nodes, :num_nodes] = self.list_adjs[index][:, :]
        adj_padded.setdiag(0)
        nodeDegree = adj_padded.sum(-1)
        if not ignore_isolate_nodes:
            nodeDegree += 1
        if self_for_none:
            adj_padded.setdiag(1)
        else:
            if max_num_nodes != num_nodes:
                adj_padded[:num_nodes, :num_nodes] += sp.eye(num_nodes)
            else:
                adj_padded += sp.eye(num_nodes)

        if type(self.list_Xs[index]) != np.ndarray:
            diag = np.ones(max_num_nodes)
            if self.set_diag_of_isol_Zer:
                diag[num_nodes:] = 0
            X = np.identity(max_num_nodes)
            np.fill_diagonal(X, diag)
            featureVec = np.array(adj_padded.sum(1)) / max_num_nodes
            X = numpy.concatenate([X, featureVec], 1)
        else:
            X = self.list_Xs[index]
        X = torch.tensor(X).float()

        # ── node onehot: (N, D) → (max_num_nodes, D) ─────────────────────
        node_oh_padded = None
        if self.list_node_onehot is not None and self.list_node_onehot[index] is not None:
            noh = self.list_node_onehot[index]          # (N, D)
            D   = noh.shape[1]
            node_oh_padded = np.zeros((max_num_nodes, D), dtype=np.float32)
            node_oh_padded[:num_nodes, :] = noh

        # ── edge onehot: (C, N, N) → (C, max_num_nodes, max_num_nodes) ───
        edge_oh_padded = None
        if self.list_edge_onehot is not None and self.list_edge_onehot[index] is not None:
            eoh = self.list_edge_onehot[index]          # (C, N, N)
            C   = eoh.shape[0]
            edge_oh_padded = np.zeros((C, max_num_nodes, max_num_nodes), dtype=np.float32)
            edge_oh_padded[:, :num_nodes, :num_nodes] = eoh

        bfs_indexes = set()
        if bfs_max_length is not None:
            while len(bfs_indexes) < bfs_max_length:
                indexes     = set(range(adj_padded.shape[0])).difference(
                                  bfs_indexes).difference(np.where(nodeDegree == 0)[0])
                source_indx = list(indexes)[np.random.randint(len(indexes))]
                bfs_index   = scipy.sparse.csgraph.breadth_first_order(adj_padded, source_indx)
                portionSize = min(len(bfs_index[0]), int(bfs_max_length / 5))
                if portionSize + len(bfs_indexes) >= bfs_max_length:
                    bfs_indexes = bfs_indexes.union(
                        bfs_index[0][:(bfs_max_length - len(bfs_indexes))])
                else:
                    bfs_indexes = bfs_indexes.union(bfs_index[0][:portionSize])
            bfs_indexes = list(bfs_indexes)

        if len(bfs_indexes) == 0:
            bfs_indexes = list(range(max_num_nodes))

        return adj_padded, X, num_nodes, bfs_indexes, node_oh_padded, edge_oh_padded

    def shuffle(self):
        indx = list(range(len(self.list_adjs)))
        np.random.shuffle(indx)

        if self.list_Xs is not None:
            self.list_Xs = [self.list_Xs[i] for i in indx]
        else:
            warnings.warn("X is empty")

        self.list_adjs = [self.list_adjs[i] for i in indx]

        if self.list_node_onehot is not None:
            self.list_node_onehot = [self.list_node_onehot[i] for i in indx]
        if self.list_edge_onehot is not None:
            self.list_edge_onehot = [self.list_edge_onehot[i] for i in indx]

        if self.featureList is not None:
            for el_i, element in enumerate(self.featureList):
                self.featureList[el_i] = element[indx]
        else:
            warnings.warn("Graph structural feature is an empty Set")

        if self.labels is not None:
            self.labels = [self.labels[i] for i in indx]
        else:
            warnings.warn("Label is an empty Set")

        if self.motif_counts is not None:
            if torch.is_tensor(self.motif_counts) or isinstance(self.motif_counts, np.ndarray):
                self.motif_counts = self.motif_counts[indx]
            else:
                self.motif_counts = [self.motif_counts[i] for i in indx]
        elif (
            self.motif_matrices is None
            and self.motif_statistics is None
            and self.motif_full_matrices is None
        ):
            warnings.warn("Motif counts is an empty Set")

        if self.motif_matrices is not None:
            if torch.is_tensor(self.motif_matrices) or isinstance(self.motif_matrices, np.ndarray):
                self.motif_matrices = self.motif_matrices[indx]
            else:
                self.motif_matrices = [self.motif_matrices[i] for i in indx]

        if self.motif_statistics is not None:
            if torch.is_tensor(self.motif_statistics) or isinstance(
                self.motif_statistics,
                np.ndarray,
            ):
                self.motif_statistics = self.motif_statistics[indx]
            else:
                self.motif_statistics = [self.motif_statistics[i] for i in indx]

        if self.motif_full_matrices is not None:
            if torch.is_tensor(self.motif_full_matrices) or isinstance(
                self.motif_full_matrices,
                np.ndarray,
            ):
                self.motif_full_matrices = self.motif_full_matrices[indx]
            else:
                self.motif_full_matrices = [
                    self.motif_full_matrices[i] for i in indx
                ]

        # Motif masks and histogram specifications are shared across graphs
        # and therefore must not be shuffled.
        if len(self.subgraph_indexes) > 0:
            self.adj_s            = [self.adj_s[i]            for i in indx]
            self.x_s              = [self.x_s[i]              for i in indx]
            self.node_onehot_s    = [self.node_onehot_s[i]    for i in indx]
            self.edge_onehot_s    = [self.edge_onehot_s[i]    for i in indx]
            self.num_nodes        = [self.num_nodes[i]        for i in indx]
            self.subgraph_indexes = [self.subgraph_indexes[i] for i in indx]

    def __getitem__(self, index):
        return self.processed_adjs[index], self.processed_Xs[index]


# generate a list of graph
def list_graph_loader(
        graph_type,
        _max_list_size=None,
        return_labels=False,
        limited_to=None,
        lobster_feature_schema="optimal_v2",
        tu_attribute_bins=8,
        tu_max_nodes=None,
        shuffle_seed=None):
  list_adj = []
  list_x =[]
  list_labels = []
  list_node_feature = []    
  list_edge_feature = []    
  node_feature_info = None
  edge_feature_info = None

  def _extract_gin_node_feature(
          graph,
          dataset_name,
          feature_keys=("feat", "attr", "label")):
      for key in feature_keys:
          if key not in graph.ndata:
              continue

          raw = graph.ndata[key]
          if raw.dim() == 1:
              feat = raw.long()
          elif raw.dim() == 2 and raw.shape[1] == 1:
              feat = raw[:, 0].long()
          else:
              feat = torch.argmax(raw, dim=1).long()

          # Match the FactorBase GIN importers, which store categorical node
          # labels as 1-based values.
          return feat + 1

      raise KeyError(
          f"{dataset_name} graph has no supported node feature key. "
          f"Available keys: {list(graph.ndata.keys())}"
      )

  def _extract_proteins_node_feature(graph):
      return _extract_gin_node_feature(graph, "PROTEINS")

  def _build_grid_graph_features(graph):
      """
      Matches factorbase_motif_pipeline/best_grid.py's "optimal" feature
      schema: node distance_to_boundary (computed per-axis via
      compute_boundary_depth, not the old single-grid_size formula, which
      over-estimated boundary distance on the shorter axis of a non-square
      grid); edge edge_axis/edge_square_count/edge_boundary_band. struct_type
      is intentionally dropped -- it's a pure relabeling of node degree,
      already implicit in the adjacency reconstruction + degree-histogram
      kernel loss.
      """
      nodes = list(graph.nodes())
      node_to_idx = {node: idx for idx, node in enumerate(nodes)}
      adj = csr_matrix(nx.adjacency_matrix(graph, nodelist=nodes))

      width, height = get_grid_dimensions(graph)

      node_rows = []
      for node in nodes:
          distance_to_boundary = compute_boundary_depth(node, width, height)
          node_rows.append([
              distance_to_boundary,
          ])

      edge_rows = []
      for node_u, node_v in graph.edges():
          edge_axis = compute_edge_axis(node_u, node_v)
          edge_square_count = compute_edge_square_count(node_u, node_v, width, height)
          edge_boundary_band = compute_edge_boundary_band(node_u, node_v, width, height)
          src = node_to_idx[node_u]
          dst = node_to_idx[node_v]

          # Store both directions so edge features align with the symmetric
          # adjacency used locally, matching the existing counting pipeline.
          edge_rows.append([src, dst, edge_axis, edge_square_count, edge_boundary_band])
          edge_rows.append([dst, src, edge_axis, edge_square_count, edge_boundary_band])

      edge_feature = None
      if edge_rows:
          edge_feature = np.asarray(edge_rows, dtype=np.int64)

      return (
          adj,
          np.asarray(node_rows, dtype=np.int64),
          edge_feature,
      )

  def _build_triangular_grid_graph_features(graph):
      """
      Matches factorbase_motif_pipeline/best_triangular_grid.py's "optimal"
      feature schema: node distance_to_boundary/num_3cycles/num_hexagons;
      edge edge_direction/edge_hexagons/edge_triangle_count. struct_type and
      the old num_6cycles are dropped -- verified via direct SQL on a learned
      FactorBase BN that struct_type/num_3cycles/num_6cycles in the old
      schema are a 100% deterministic relabeling of node degree, and the old
      num_6cycles was a degree>=4 proxy, not a real hexagon count.
      num_hexagons here is a REAL induced-6-cycle participation count.
      """
      nodes = list(graph.nodes())
      node_to_idx = {node: idx for idx, node in enumerate(nodes)}
      adj = csr_matrix(nx.adjacency_matrix(graph, nodelist=nodes))

      bounds = triangular_grid_features.get_lattice_bounds(graph)
      # Compute ONCE per graph -- O(graph) cycle search, then indexed per
      # node/edge below.
      node_hexagons_raw, edge_hexagons_raw = (
          triangular_grid_features.compute_induced_hexagon_participation(graph)
      )

      node_rows = []
      for node in nodes:
          distance_to_boundary = triangular_grid_features.compute_distance_to_boundary(
              node, bounds
          )
          num_3cycles = triangular_grid_features.compute_num_3cycles(graph, node)
          num_hexagons = node_hexagons_raw[node] + 1
          node_rows.append([
              distance_to_boundary,
              num_3cycles,
              num_hexagons,
          ])

      edge_rows = []
      for node_u, node_v in graph.edges():
          edge_direction = triangular_grid_features.compute_edge_direction(
              graph, node_u, node_v
          )
          edge_key = tuple(sorted((node_u, node_v)))
          edge_hexagons = edge_hexagons_raw[edge_key] + 1
          edge_triangle_count = triangular_grid_features.compute_edge_triangle_count(
              graph, node_u, node_v
          )
          src = node_to_idx[node_u]
          dst = node_to_idx[node_v]

          # Keep local edge-feature rows symmetric with the adjacency relation.
          edge_rows.append([src, dst, edge_direction, edge_hexagons, edge_triangle_count])
          edge_rows.append([dst, src, edge_direction, edge_hexagons, edge_triangle_count])

      edge_feature = None
      if edge_rows:
          edge_feature = np.asarray(edge_rows, dtype=np.int64)

      return (
          adj,
          np.asarray(node_rows, dtype=np.int64),
          edge_feature,
      )

  def _build_triangular_grid_legacy_graph_features(graph):
      """
      Matches factorbase_motif_pipeline/to_db_triangular_grid.py's legacy
      feature schema used by triangular_grid_undir_feat_snap_ce92ed:
      node struct_type/distance_to_boundary/num_3cycles/num_6cycles and
      edge edge_orbit.
      """
      nodes = list(graph.nodes())
      node_to_idx = {node: idx for idx, node in enumerate(nodes)}
      adj = csr_matrix(nx.adjacency_matrix(graph, nodelist=nodes))

      bounds = triangular_grid_features.get_lattice_bounds(graph)

      node_rows = []
      for node in nodes:
          struct_type = triangular_grid_features.compute_struct_type(graph, node)
          distance_to_boundary = triangular_grid_features.compute_distance_to_boundary(
              node, bounds
          )
          num_3cycles = triangular_grid_features.compute_num_3cycles(graph, node)
          num_6cycles = triangular_grid_features.compute_num_6cycles(graph, node)
          node_rows.append([
              struct_type,
              distance_to_boundary,
              num_3cycles,
              num_6cycles,
          ])

      edge_rows = []
      for node_u, node_v in graph.edges():
          edge_orbit = triangular_grid_features.compute_edge_orbit(
              node_u, node_v, bounds
          )
          src = node_to_idx[node_u]
          dst = node_to_idx[node_v]
          edge_rows.append([src, dst, edge_orbit])
          edge_rows.append([dst, src, edge_orbit])

      edge_feature = None
      if edge_rows:
          edge_feature = np.asarray(edge_rows, dtype=np.int64)

      return (
          adj,
          np.asarray(node_rows, dtype=np.int64),
          edge_feature,
      )

  def _build_lobster_optimal_graph_features(graph):
      """
      Matches factorbase_motif_pipeline/best_lobster.py's "optimal" feature
      schema: node node_degree/spine_role/subtree_size/eccentricity; edge
      edge_type/depth_pair/terminal_edge. spine_role replaces the old
      distance_to_spine -- that scheme's "Far-Spine" bucket is empirically
      EMPTY at this dataset's p1=p2=0.7 generation scale (a dead category),
      and the freed slot is used to distinguish the two structurally-special
      spine endpoints instead. subtree_size/eccentricity are merged to 3
      buckets each (their old 4th bucket was a persistently thin tail).
      depth_pair/terminal_edge are new relational edge features replacing
      the old single edge_type-only scheme.
      """
      nodes = list(graph.nodes())
      node_to_idx = {node: idx for idx, node in enumerate(nodes)}
      adj = csr_matrix(nx.adjacency_matrix(graph, nodelist=nodes))

      spine_path = lobster_features.find_spine_path(graph)
      spine_nodes = set(spine_path)
      distance_to_spine = lobster_features.compute_distance_to_spine_raw(
          graph,
          spine_path,
      )
      subtree_sizes = lobster_features.compute_branch_component_sizes_v2(
          graph,
          spine_path,
      )

      node_rows = []
      for node in nodes:
          node_degree = lobster_features.compute_node_degree(graph, node)
          spine_role = lobster_features.compute_spine_role(
              node, spine_path, spine_nodes, distance_to_spine
          )
          subtree_size = subtree_sizes.get(node, 1)
          eccentricity = lobster_features.compute_eccentricity_v2(graph, node)
          node_rows.append([
              node_degree,
              spine_role,
              subtree_size,
              eccentricity,
          ])

      edge_rows = []
      for source_node, target_node in graph.edges():
          edge_type = lobster_features.compute_edge_type(
              source_node,
              target_node,
              spine_nodes,
          )
          depth_pair = lobster_features.compute_depth_pair(
              source_node, target_node, distance_to_spine
          )
          terminal_edge = lobster_features.compute_terminal_edge(
              graph, source_node, target_node
          )
          src = node_to_idx[source_node]
          dst = node_to_idx[target_node]

          # Keep local edge-feature rows symmetric with the adjacency relation.
          edge_rows.append([src, dst, edge_type, depth_pair, terminal_edge])
          edge_rows.append([dst, src, edge_type, depth_pair, terminal_edge])

      edge_feature = None
      if edge_rows:
          edge_feature = np.asarray(edge_rows, dtype=np.int64)

      return (
          adj,
          np.asarray(node_rows, dtype=np.int64),
          edge_feature,
      )

  def _build_lobster_old_graph_features(graph):
      """Build the v1.2 features used by the old-feature Lobster results.

      This schema matches ``factorbase_motif_pipeline/to_db_lobster.py`` and
      the ``lobster_undir_feat_snap_85093d`` FactorBase database:

      * nodes: node_degree, distance_to_spine, subtree_size, eccentricity
      * edges: edge_type
      """
      nodes = list(graph.nodes())
      node_to_idx = {node: idx for idx, node in enumerate(nodes)}
      adj = csr_matrix(nx.adjacency_matrix(graph, nodelist=nodes))

      spine_path = lobster_features.find_spine_path(graph)
      spine_nodes = set(spine_path)
      distance_to_spine = lobster_features.compute_distance_to_spine_labels(
          graph,
          spine_path,
      )
      subtree_sizes = lobster_features.compute_branch_component_sizes(
          graph,
          spine_path,
      )

      node_rows = []
      for node in nodes:
          node_rows.append([
              lobster_features.compute_node_degree(graph, node),
              distance_to_spine[node],
              subtree_sizes.get(node, 1),
              lobster_features.compute_eccentricity(graph, node),
          ])

      edge_rows = []
      for source_node, target_node in graph.edges():
          edge_type = lobster_features.compute_edge_type(
              source_node,
              target_node,
              spine_nodes,
          )
          src = node_to_idx[source_node]
          dst = node_to_idx[target_node]
          edge_rows.append([src, dst, edge_type])
          edge_rows.append([dst, src, edge_type])

      edge_feature = None
      if edge_rows:
          edge_feature = np.asarray(edge_rows, dtype=np.int64)

      return (
          adj,
          np.asarray(node_rows, dtype=np.int64),
          edge_feature,
      )

  if graph_type=="IMDBBINARY":
      data = load_gin_dataset('IMDBBINARY')
      graphs, labels = data.graphs, data.labels
      for i, graph in enumerate(graphs):
          list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
          # list_x.append(graph.ndata['feat'])
          list_x.append(None)
          list_labels.append(labels[i].cpu().item())
      graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
      np.save('IMDBBINARY_lattice_graph.npy', graphs_to_writeOnDisk, allow_pickle=True)

  elif graph_type=="NCI1":
      data = load_gin_dataset('NCI1')
      graphs, labels = data.graphs, data.labels
      for i, graph in enumerate(graphs):
          list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
          # list_x.append(graph.ndata['feat'])
          list_x.append(None)
          list_labels.append(labels[i].cpu().item())
      graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
      np.save('NCI1_lattice_graph.npy', graphs_to_writeOnDisk, allow_pickle=True)
  elif graph_type=="MUTAG":
      data = load_gin_dataset('MUTAG')
      graphs, labels = data.graphs, data.labels
      node_feature_info = {
          0: {'feature_name': 'node_feature'}
      }
      edge_feature_info = None
      for i, graph in enumerate(graphs):
          list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
          list_x.append(None)
          list_labels.append(labels[i].cpu().item())
          node_feature = _extract_gin_node_feature(
              graph,
              "MUTAG",
              feature_keys=("label", "attr", "feat"),
          )
          list_node_feature.append(
              node_feature.view(-1, 1).cpu().numpy().astype(np.int64)
          )
          list_edge_feature.append(None)
  elif graph_type=="COLLAB":
      data = load_gin_dataset('COLLAB')
      graphs, labels = data.graphs, data.labels
      for i, graph in enumerate(graphs):
          list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
          # list_x.append(graph.ndata['feat'])
          list_x.append(None)
          list_labels.append(labels[i].cpu().item())
      graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
      # np.save('COLLAB_lattice_graph.npy', graphs_to_writeOnDisk, allow_pickle=True)
  elif graph_type=="PTC":
      data = load_gin_dataset('PTC')
      graphs, labels = data.graphs, data.labels
      node_feature_info = {
          0: {'feature_name': 'node_feature'}
      }
      edge_feature_info = None
      for i, graph in enumerate(graphs):
          list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
          list_x.append(None)
          list_labels.append(labels[i].cpu().item())
          node_feature = _extract_gin_node_feature(
              graph,
              "PTC",
              feature_keys=("label", "attr", "feat"),
          )
          list_node_feature.append(
              node_feature.view(-1, 1).cpu().numpy().astype(np.int64)
          )
          list_edge_feature.append(None)
  elif graph_type == "PROTEINS":
#===================================start kiarash code

      # data = dgl.data.GINDataset(name='PROTEINS', self_loop=False)
      # graphs, labels = data.graphs, data.labels
      # for i, graph in enumerate(graphs):
      #     if graph.adjacency_matrix().shape[0] < 100:
      #         list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
      #         # list_x.append(graph.ndata['feat'])
      #         list_x.append(None)
      #         list_labels.append(labels[i].cpu().item())
      # # graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
      # # np.save('PROTEINS.npy', graphs_to_writeOnDisk, allow_pickle=True)
#==================================end kiarash code
      data = dgl.data.GINDataset(name='PROTEINS', self_loop=False)
      graphs, labels = data.graphs, data.labels

      node_feature_info = {
          0: {'feature_name': 'node_feature'}
      }
      edge_feature_info = None
      include_large_proteins = os.environ.get("INCLUDE_LARGE_PROTEINS", "0") == "1"

      for i, graph in enumerate(graphs):
          if (not include_large_proteins) and graph.adjacency_matrix().shape[0] >= 100:
              continue

          if i % 100 == 0:
              print(f"PROTEINS loading: {i}/{len(graphs)}")

          adj = csr_matrix(graph.adjacency_matrix().to_dense().numpy())
          list_adj.append(adj)
          list_x.append(None)
          list_labels.append(labels[i].cpu().item())

          node_feature = _extract_proteins_node_feature(graph)
          list_node_feature.append(
              node_feature.view(-1, 1).cpu().numpy().astype(np.int64)
          )
          list_edge_feature.append(None)


  elif graph_type.upper() in {"AIDS", "ENZYMES", "ENZYMEZ"}:
      dataset_name = "ENZYMES" if graph_type.upper() == "ENZYMEZ" else graph_type.upper()
      spec = TU_DATASET_SPECS[dataset_name]
      dataset_dir = find_dataset_dir(
          spec,
          requested=data_path("Kernel_dataset"),
          allow_download=True,
      )
      graphs, load_stats = load_tu_graphs(
          spec,
          dataset_dir,
          max_nodes=tu_max_nodes,
          max_graphs=_max_list_size,
      )
      prepared_attributes, _attribute_sql_type, quantile_thresholds = prepare_attributes(
          graphs,
          mode="quantile",
          bins=int(tu_attribute_bins),
          width=spec.node_attribute_count,
      )

      node_label_values = sorted({
          int(label)
          for graph in graphs
          for label in graph.node_labels
      })
      node_feature_info = {
          0: {
              "feature_name": "node_label",
              "unique_values": node_label_values,
          }
      }
      for attribute_index, _thresholds in enumerate(quantile_thresholds):
          observed_values = sorted({
              int(row[attribute_index])
              for graph_rows in prepared_attributes
              for row in graph_rows
          })
          node_feature_info[attribute_index + 1] = {
              "feature_name": f"node_attr_{attribute_index:02d}",
              # FactorBase learns only states actually present in the SQL
              # table. Do not seed empty quantile bins into the decoder.
              "unique_values": observed_values,
          }

      edge_feature_info = None
      if spec.has_edge_labels:
          edge_label_values = sorted({
              int(label)
              for graph in graphs
              for _src, _dst, label in graph.edges
              if label is not None
          })
          edge_feature_info = {
              0: {
                  "feature_name": "edge_label",
                  "unique_values": edge_label_values,
              }
          }

      for graph, graph_attributes in zip(graphs, prepared_attributes):
          edge_rows = deduplicated_edges(graph, edge_mode="undirected")
          num_nodes = len(graph.node_labels)
          if edge_rows:
              sources = np.asarray([row[0] for row in edge_rows], dtype=np.int64)
              targets = np.asarray([row[1] for row in edge_rows], dtype=np.int64)
              adjacency = csr_matrix(
                  (np.ones(len(edge_rows), dtype=np.int8), (sources, targets)),
                  shape=(num_nodes, num_nodes),
              )
          else:
              adjacency = csr_matrix((num_nodes, num_nodes), dtype=np.int8)

          attribute_array = np.asarray(graph_attributes, dtype=np.int64)
          label_array = np.asarray(graph.node_labels, dtype=np.int64).reshape(-1, 1)
          node_features = np.concatenate([label_array, attribute_array], axis=1)

          list_adj.append(adjacency)
          list_x.append(None)
          list_labels.append(graph.graph_label)
          list_node_feature.append(node_features)

          if spec.has_edge_labels and edge_rows:
              list_edge_feature.append(np.asarray([
                  [src, dst, int(edge_label)]
                  for src, dst, edge_label in edge_rows
              ], dtype=np.int64))
          else:
              list_edge_feature.append(None)

      print(
          f"{dataset_name} loading: kept {load_stats['loaded_graphs']}/"
          f"{load_stats['source_graphs']} graphs, "
          f"skipped {load_stats['skipped_max_nodes']} above tu_max_nodes, "
          f"quantile_bins={tu_attribute_bins}"
      )


  elif graph_type == "QM9":
#===================================start kiarash code
  
    #   data = dgl.data.QM9Dataset(label_keys=['mu'])
    #   for i, graph in enumerate(data):
    #       # if i==1000:
    #       #     break
    #       adj = graph[0].adj().to_dense().cpu().numpy()
    #       list_adj.append(scipy.sparse.csr_matrix(adj))
    #       list_x.append(None)
    #       list_labels.append(None)
    #       print(i)
#==================================end kiarash code
        from torch_geometric.datasets import QM9
        data = QM9(root=str(data_path("QM9")))

        # ── node feature metadata ─────────────────────────────────
        # Two columns: col-0 = atom_type (1-5), col-1 = num_h (1-4)
        # This matches the qm9_experiment database schema.
        node_feature_info = {
            0: {'feature_name': 'atom_type'},
            1: {'feature_name': 'num_h'},
        }

        # ── edge feature metadata ─────────────────────────────────
        # One edge feature: bond_type encoded in col-2 of list_edge_feature[i]
        # Scan dataset once to find all unique bond-type values globally
        all_bond_vals = set()
        for mol in data:
            if mol.edge_attr is not None and mol.edge_attr.size(0) > 0:
                all_bond_vals.update(
                    torch.argmax(mol.edge_attr, dim=1).tolist()
                )
        edge_feature_info = {
            0: {
                'feature_name':  'bond_type',
                'unique_values': sorted(int(v) for v in all_bond_vals),
            }
        }

        for i, mol in enumerate(data):
            if i % 10000 == 0:
                print(f"QM9 loading: {i}/{len(data)}")

            N          = mol.num_nodes
            edge_index = mol.edge_index

            # adjacency
            adj = scipy.sparse.csr_matrix(
                (np.ones(edge_index.size(1)),
                 (edge_index[0].numpy(), edge_index[1].numpy())),
                shape=(N, N)
            )
            list_adj.append(adj)
            list_x.append(None)
            list_labels.append(None)

            # node features  →  (N, 2)  int array
            X         = mol.x
            atom_type = torch.argmax(X[:, 0:5], dim=1) + 1
            num_h     = torch.clamp(X[:, 10].long(), max=3) + 1
            list_node_feature.append(
                torch.stack([atom_type, num_h], dim=1).numpy().astype(np.int64)
            )

            # edge features  →  (E, 3)  int array  [src, dst, bond_type]
            if mol.edge_attr is not None and mol.edge_attr.size(0) > 0:
                bond_type = torch.argmax(mol.edge_attr, dim=1)
                list_edge_feature.append(
                    torch.stack([edge_index[0], edge_index[1], bond_type], dim=1)
                    .numpy().astype(np.int64)
                )
            else:
                list_edge_feature.append(None)


    #   print("done")
  elif graph_type == "ogbg-molbbbp":
      # OGB molecular graph property dataset used by GraphVAE-MM.
      #
      # Unlike the legacy GraphVAE-MM loader, keep OGB's categorical atom and
      # bond features so the REQ node/edge feature decoders and motif counting
      # can use them. Feature arrays follow the repo-wide convention:
      #   list_node_feature[i] = (N, F_node) int array
      #   list_edge_feature[i] = (E, 2 + F_edge) int array [src, dst, ...]
      dataset = load_ogbg_molbbbp_dataset()
      ogbg_molbbbp_max_nodes = 60

      first_graph, _first_label = dataset[0]
      if "feat" not in first_graph.ndata:
          raise KeyError(
              "ogbg-molbbbp DGL graphs do not expose node feature key 'feat'. "
              f"Available node keys: {list(first_graph.ndata.keys())}"
          )

      node_feature_dim = int(first_graph.ndata["feat"].shape[1])

      # Pre-scan node/edge feature values globally so one-hot dimensions are
      # stable across train/validation/test splits, subsets, and future motif
      # caches.
      all_node_vals: Dict[int, set] = {col: set() for col in range(node_feature_dim)}
      edge_feature_dim = 0
      all_edge_vals: Dict[int, set] = {}
      for graph, _label in dataset:
          if graph.num_nodes() > ogbg_molbbbp_max_nodes:
              continue

          node_feat = graph.ndata["feat"]
          for col in range(node_feature_dim):
              all_node_vals[col].update(
                  int(v) for v in torch.unique(node_feat[:, col]).cpu().tolist()
              )

          edge_feat = graph.edata["feat"] if "feat" in graph.edata else None
          if edge_feat is None:
              continue
          if edge_feat.dim() == 1:
              edge_feat = edge_feat.view(-1, 1)
          edge_feature_dim = max(edge_feature_dim, int(edge_feat.shape[1]))
          for col in range(int(edge_feat.shape[1])):
              all_edge_vals.setdefault(col, set()).update(
                  int(v) for v in torch.unique(edge_feat[:, col]).cpu().tolist()
              )

      node_feature_info = {
          col: {
              "feature_name": (
                  OGB_MOL_NODE_FEATURE_NAMES[col]
                  if col < len(OGB_MOL_NODE_FEATURE_NAMES)
                  else f"atom_feature_{col}"
              ),
              "unique_values": sorted(all_node_vals.get(col, set())),
          }
          for col in range(node_feature_dim)
      }

      edge_feature_info = {
          col: {
              "feature_name": (
                  OGB_MOL_EDGE_FEATURE_NAMES[col]
                  if col < len(OGB_MOL_EDGE_FEATURE_NAMES)
                  else f"bond_feature_{col}"
              ),
              "unique_values": sorted(all_edge_vals.get(col, set())),
          }
          for col in range(edge_feature_dim)
      } if edge_feature_dim > 0 else None

      kept_ogbg_molbbbp_graphs = 0
      skipped_ogbg_molbbbp_graphs = 0
      for i, (graph, label) in enumerate(dataset):
          if i % 250 == 0:
              print(f"ogbg-molbbbp loading: {i}/{len(dataset)}")

          if graph.num_nodes() > ogbg_molbbbp_max_nodes:
              skipped_ogbg_molbbbp_graphs += 1
              continue

          adj = _dgl_graph_to_csr(graph)
          list_adj.append(adj)
          list_x.append(None)
          label_values = label.cpu().view(-1)
          list_labels.append(
              None if label_values.numel() == 0 else label_values[0].item()
          )

          node_feature = graph.ndata["feat"].cpu().numpy().astype(np.int64)
          list_node_feature.append(node_feature)

          edge_feat = graph.edata["feat"] if "feat" in graph.edata else None
          if edge_feat is None or edge_feat.numel() == 0:
              list_edge_feature.append(None)
          else:
              if edge_feat.dim() == 1:
                  edge_feat = edge_feat.view(-1, 1)
              src, dst = graph.edges()
              edge_feature = torch.cat(
                  [
                      src.view(-1, 1).cpu(),
                      dst.view(-1, 1).cpu(),
                      edge_feat.cpu().long(),
                  ],
                  dim=1,
              ).numpy().astype(np.int64)
              list_edge_feature.append(edge_feature)

          kept_ogbg_molbbbp_graphs += 1

      print(
          "ogbg-molbbbp max-node filter: "
          f"kept {kept_ogbg_molbbbp_graphs}/{len(dataset)} graphs, "
          f"skipped {skipped_ogbg_molbbbp_graphs} graphs with "
          f"num_nodes > {ogbg_molbbbp_max_nodes}"
      )

      # list_labels are kept for completeness; graph generation does not use
      # labels directly.
  elif graph_type=="large_grid":
      for i in range(10):
            list_adj.append(nx.adjacency_matrix(grid(30, 100)))
            list_x.append(None)
  elif graph_type=="GRID":
#===================================start kirash code
      # for i in range(10, 20):
      #   for j in range(10, 20):
      #       list_adj.append(nx.adjacency_matrix(grid(i, j)))
      #       list_x.append(None)
#==================================end kirash code
      node_feature_info = {
          0: {'feature_name': 'distance_to_boundary'},
      }
      edge_feature_info = {
          0: {
              'feature_name': 'edge_axis',
              'unique_values': sorted(EDGE_AXIS_TO_ID.values()),
          },
          1: {
              'feature_name': 'edge_square_count',
              'unique_values': sorted(EDGE_SQUARE_COUNT_LABELS.keys()),
          },
          2: {
              'feature_name': 'edge_boundary_band',
              'unique_values': [1, 2, 3, 4, 5],
          },
      }

      for i in range(10, 20):
        for j in range(10, 20):
            graph = grid(i, j)
            adj, node_feature, edge_feature = _build_grid_graph_features(graph)
            list_adj.append(adj)
            list_x.append(None)
            list_node_feature.append(node_feature)
            list_edge_feature.append(edge_feature)

  elif graph_type=="TRIANGULAR_GRID":
#===================================start kirash code
      # for i in range(10, 20):
      #   for j in range(10, 20):
      #       list_adj.append(nx.adjacency_matrix(nx.triangular_lattice_graph(i, j)))
      #       list_x.append(None)
      # # graphs_to_writeOnDisk = [gr.toarray() for  gr in list_adj]
      # # np.save('triangular_lattice_graph.npy', graphs_to_writeOnDisk, allow_pickle=True)
#==================================end kirash code
      triangular_feature_schema = os.environ.get(
          "TRIANGULAR_GRID_FEATURE_SCHEMA",
          "optimal",
      ).strip().lower()
      if triangular_feature_schema == "legacy":
          node_feature_info = {
              0: {'feature_name': 'struct_type'},
              1: {'feature_name': 'distance_to_boundary'},
              2: {'feature_name': 'num_3cycles'},
              3: {'feature_name': 'num_6cycles'},
          }
          edge_feature_info = {
              0: {
                  'feature_name': 'edge_orbit',
                  'unique_values': sorted(triangular_grid_features.EDGE_ORBIT_TO_ID.values()),
              },
          }
      else:
          node_feature_info = {
              0: {'feature_name': 'distance_to_boundary'},
              1: {'feature_name': 'num_3cycles'},
              2: {'feature_name': 'num_hexagons'},
          }
          edge_feature_info = {
              0: {
                  'feature_name': 'edge_direction',
                  'unique_values': sorted(triangular_grid_features.EDGE_DIRECTION_TO_ID.values()),
              },
              1: {
                  'feature_name': 'edge_hexagons',
                  'unique_values': [1, 2, 3],
              },
              2: {
                  'feature_name': 'edge_triangle_count',
                  'unique_values': [0, 1, 2],
              },
          }

      for i in range(10, 20):
        for j in range(10, 20):
            graph = nx.triangular_lattice_graph(i, j)
            if triangular_feature_schema == "legacy":
                adj, node_feature, edge_feature = _build_triangular_grid_legacy_graph_features(graph)
            else:
                adj, node_feature, edge_feature = _build_triangular_grid_graph_features(graph)
            list_adj.append(adj)
            list_x.append(None)
            list_node_feature.append(node_feature)
            list_edge_feature.append(edge_feature)
  elif graph_type=="small_triangular_grid":
      for i in range(6, 12):
        for j in range(6, 12):
            list_adj.append(nx.adjacency_matrix(nx.triangular_lattice_graph(i, j)))
            list_x.append(None)
      # graphs_to_writeOnDisk = [gr.toarray() for  gr in list_adj]
      # np.save('triangular_lattice_graph.npy', graphs_to_writeOnDisk, allow_pickle=True)
  elif graph_type=="fancy_grid":
      for i in range(4, 8):
        for j in range(4, 8):
            list_adj.append(nx.adjacency_matrix(grid(i, j)))
      list_adj = padd_adj_to(list_adj, np.max(np.array([adj.shape[0] for adj in list_adj])))
      for adj in list_adj:
        list_x.append(node_festure_creator(adj, 3,10))
  elif graph_type == "tree":
      for graph_size in range(3, 83):
          list_x.append(None)
          list_adj.append(nx.adjacency_matrix(nx.random_tree(graph_size)))

  elif graph_type == "star":
      for graph_size in range(3,83):
          list_x.append(None)
          list_adj.append(nx.adjacency_matrix(nx.star_graph(graph_size)))

  elif graph_type == "wheel_graph":
      for graph_size in range(3,83):
          list_x.append(None)
          list_adj.append(nx.adjacency_matrix(nx.wheel_graph(graph_size)))
  elif graph_type=="IMDbMulti":
      list_adj = pkl.load(open(data_path("IMDbMulti", "IMDBMulti.p"),'rb'))
      list_x= [None for x in list_adj]
  elif graph_type=="one_grid":
        list_adj.append(nx.adjacency_matrix(grid(350, 10)))
        list_x.append(None)
  elif graph_type=="small_grid":
      for i in range(2, 3):
        for j in range(2, 5):
            list_adj.append(nx.adjacency_matrix(grid(i, j)))
            list_x.append(None)
  elif graph_type=="huge_grids":
      for i in range(4, 10):
          for j in range(4, 10):
              list_adj.append(nx.adjacency_matrix(grid(i, j)))
              list_x.append(None)
  elif graph_type=="community":
      for i in range(30, 81):
        for j in range(30,81):
            list_adj.append(nx.adjacency_matrix(n_community([i, j], p_inter=0.3, p_intera=0.05)))
            list_x.append(None)

  elif graph_type=="multi_community":
      for g_i in range(400):
            communities = [random.randint(30, 81) for i in range(random.randint(2, 5))]
            list_adj.append(nx.adjacency_matrix(n_community(communities, p_inter=0.3, p_intera=0.05)))
            list_x.append(None)
            list_labels.append(len(communities)-2)

  elif graph_type == "PVGAErandomGraphs":
      for i in range(1000):
          import randomGraphGen
          # n = np.random.randint(low=20, high=40)
          n = 20
          graphGen = randomGraphGen.GraphGenerator()
          list_x.append(None)
          g, g_type = graphGen(n)
          list_adj.append(nx.adjacency_matrix(g))
          list_labels.append(g_type)
      # graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
      # np.save('PVGAErandomGraphs.npy', graphs_to_writeOnDisk, allow_pickle=True)

  # elif graph_type == "PVGAErandomGraphs_10000":
  #     for i in range(10000):
  #         import randomGraphGen
  #         # n = np.random.randint(low=20, high=40)
  #         n = 20
  #         graphGen = randomGraphGen.GraphGenerator()
  #         list_x.append(None)
  #         list_adj.append(nx.adjacency_matrix(graphGen(n)))
  #     graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
  #     np.save('PVGAErandomGraphs_10000.npy', graphs_to_writeOnDisk, allow_pickle=True)
  # elif graph_type == "PVGAErandomGraphs_100000":
  #     for i in range(100000):
  #         import randomGraphGen
  #         # n = np.random.randint(low=20, high=40)
  #         n = 20
  #         graphGen = randomGraphGen.GraphGenerator()
  #         list_x.append(None)
  #         list_adj.append(nx.adjacency_matrix(graphGen(n)))
  #     graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
  #     np.save('PVGAErandomGraphs_100000.npy', graphs_to_writeOnDisk, allow_pickle=True)
  elif graph_type == 'small_lobster':
      graphs = []
      p1 = 0.7
      p2 = 0.7
      count = 0
      min_node = 8
      max_node = 12
      max_edge = 0
      mean_node = 15
      num_graphs = 8
      seed=1234
      seed_tmp = seed
      while count < num_graphs:
          G = nx.random_lobster(mean_node, p1, p2, seed=seed_tmp)
          if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
              graphs.append(G)
              list_adj.append(nx.adjacency_matrix(G))
              list_x.append(None)
              count += 1
          seed_tmp += 1
  elif graph_type == 'small_lobster':
      graphs = []
      p1 = 0.7
      p2 = 0.7
      count = 0
      min_node = 1000
      max_node = 10000
      max_edge = 0
      mean_node = 5000
      num_graphs = 100
      seed=1234
      seed_tmp = seed
      while count < num_graphs:
          G = nx.random_lobster(mean_node, p1, p2, seed=seed_tmp)
          if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
              graphs.append(G)
              list_adj.append(nx.adjacency_matrix(G))
              list_x.append(None)
              count += 1
          seed_tmp += 1
  elif graph_type == 'LOBSTER':
#===================================start kirash code
      # graphs = []
      # p1 = 0.7
      # p2 = 0.7
      # count = 0
      # min_node = 10
      # max_node = 100
      # max_edge = 0
      # mean_node = 80
      # num_graphs = 100
      # seed=1234
      # seed_tmp = seed
      # while count < num_graphs:
      #     G = nx.random_lobster(mean_node, p1, p2, seed=seed_tmp)
      #     if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
      #         graphs.append(G)
      #         list_adj.append(nx.adjacency_matrix(G))
      #         list_x.append(None)
      #         count += 1
      #     seed_tmp += 1
      # # writing the generated graph for benchmarking
      # # graphs_to_writeOnDisk = [gr.toarray() for  gr in list_adj]
      # # np.save('Lobster_adj.npy', graphs_to_writeOnDisk, allow_pickle=True)
#==================================end kirash code
      if lobster_feature_schema == "old_v1":
          node_feature_info = {
              0: {'feature_name': 'node_degree'},
              1: {'feature_name': 'distance_to_spine'},
              2: {'feature_name': 'subtree_size'},
              3: {'feature_name': 'eccentricity'},
          }
          edge_feature_info = {
              0: {
                  'feature_name': 'edge_type',
                  'unique_values': sorted(lobster_features.EDGE_TYPE_TO_ID.values()),
              },
          }
          lobster_feature_builder = _build_lobster_old_graph_features
      elif lobster_feature_schema == "optimal_v2":
          node_feature_info = {
              0: {'feature_name': 'node_degree'},
              1: {'feature_name': 'spine_role'},
              2: {'feature_name': 'subtree_size'},
              3: {'feature_name': 'eccentricity'},
          }
          edge_feature_info = {
              0: {
                  'feature_name': 'edge_type',
                  'unique_values': sorted(lobster_features.EDGE_TYPE_TO_ID.values()),
              },
              1: {
                  'feature_name': 'depth_pair',
                  'unique_values': sorted(lobster_features.DEPTH_PAIR_LABELS.keys()),
              },
              2: {
                  'feature_name': 'terminal_edge',
                  'unique_values': sorted(lobster_features.TERMINAL_EDGE_LABELS.keys()),
              },
          }
          lobster_feature_builder = _build_lobster_optimal_graph_features
      else:
          raise ValueError(
              "lobster_feature_schema must be 'old_v1' or 'optimal_v2'; "
              f"received {lobster_feature_schema!r}"
          )

      graphs = []
      p1 = 0.7
      p2 = 0.7
      count = 0
      min_node = 10
      max_node = 100
      max_edge = 0
      mean_node = 80
      num_graphs = 100
      seed=1234
      seed_tmp = seed
      while count < num_graphs:
          G = nx.random_lobster(mean_node, p1, p2, seed=seed_tmp)
          if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
              graphs.append(G)
              adj, node_feature, edge_feature = lobster_feature_builder(G)
              list_adj.append(adj)
              list_x.append(None)
              list_node_feature.append(node_feature)
              list_edge_feature.append(edge_feature)
              count += 1
          seed_tmp += 1
  elif graph_type=="mnist":
      list_adj = []
      list_x = []
      import torch_geometric
      dataset_b = torch_geometric.datasets.MNISTSuperpixels(root=str(data_path("geometric")))
      for i in range(len(dataset_b.data.y)):  # len(dataset_b.data.y)
          in_1 = dataset_b[i].edge_index[0].detach().numpy()
          in_2 = dataset_b[i].edge_index[1].detach().numpy()
          valu = numpy.ones(len(in_2))
          adj = scipy.sparse.csr_matrix((valu, (in_1, in_2)), shape=(dataset_b[i].num_nodes, dataset_b[i].num_nodes))
          list_adj.append(adj)
          list_x.append(None)
  elif graph_type == "zinc":
      import torch_geometric
      dataset_b = torch_geometric.datasets.ZINC(root=str(data_path("geometric", "MoleculeNet", "zinc")), subset=False)
      list_adj = []
      for i in range(len(dataset_b.data.y)):
          in_1 = dataset_b[i].edge_index[0].detach().numpy()
          in_2 = dataset_b[i].edge_index[1].detach().numpy()
          valu = numpy.ones(len(in_2))
          adj = scipy.sparse.csr_matrix((valu, (in_1, in_2)), shape=(dataset_b[i].num_nodes, dataset_b[i].num_nodes))
          list_adj.append(adj)
          list_x.append(None)
  elif graph_type == "cora":
      import input_data
      list_adj, list_x, _,_,_ = input_data.load_data(graph_type)
      list_adj = [list_adj]
      list_x = [list_x]
  elif graph_type == "ACM":
      import input_data
      list_adj, list_x, _,_,_ = input_data.load_data(graph_type)
      list_adj = [list_adj]
      list_x = [list_x]
  elif graph_type == 'ego':
      _, _, G = Graph_load(dataset='citeseer')
      # G = max(nx.connected_component_subgraphs(G), key=len)
      G = max((G.subgraph(c) for c in nx.connected_components(G)), key=len)
      G = nx.convert_node_labels_to_integers(G)
      graphs = []
      for i in range(G.number_of_nodes()):
          G_ego = nx.ego_graph(G, i, radius=3)
          if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
              graphs.append(G_ego)
              list_adj.append(nx.adjacency_matrix(G_ego))
              list_x.append(None)


  elif graph_type == 'FIRSTMM_DB':
    list_adj, list_x, list_labels  = graph_load_batch(
        str(data_path("Kernel_dataset")),
        min_num_nodes=0,
        max_num_nodes=2000,
        name='FIRSTMM_DB',
        node_attributes=False,
        graph_labels=True)

  elif graph_type == 'DD':
    list_adj, list_x, list_labels  = graph_load_batch(
        str(data_path("Kernel_dataset")),
        min_num_nodes=100,
        max_num_nodes=500,
        name='DD',
        node_attributes=False,
        graph_labels=True)
    # args.max_prev_node = 230



  def return_subset(A, X, Y, NF, EF, limited_to):
        indx = list(range(len(A)))
        shuffle_rng = (
            random
            if shuffle_seed is None
            else random.Random(int(shuffle_seed))
        )
        shuffle_rng.shuffle(indx)
        A  = [A[i]  for i in indx]
        X  = [X[i]  for i in indx]
        NF = [NF[i] for i in indx]
        EF = [EF[i] for i in indx]
        if Y is not None and len(Y) != 0:
            Y = [Y[i] for i in indx]
        if limited_to is not None:
            A, X, NF, EF = A[:limited_to], X[:limited_to], NF[:limited_to], EF[:limited_to]
            if Y is not None and len(Y) != 0:
                Y = Y[:limited_to]
        return A, X, Y, NF, EF


  if return_labels ==True:
      if len(list_labels)==0:
          list_labels = None

  if len(list_node_feature) == 0:
      list_node_feature = [None for _ in list_adj]
  elif len(list_node_feature) != len(list_adj):
      raise ValueError("list_node_feature must align with list_adj")

  if len(list_edge_feature) == 0:
      list_edge_feature = [None for _ in list_adj]
  elif len(list_edge_feature) != len(list_adj):
      raise ValueError("list_edge_feature must align with list_adj")

  list_adj, list_x, list_labels, list_node_feature, list_edge_feature = \
      return_subset(list_adj, list_x, list_labels, list_node_feature, list_edge_feature, limited_to)

  return (list_adj, list_x, list_labels,
          list_node_feature, list_edge_feature,
          node_feature_info, edge_feature_info)

def data_split(
        graph_lis,
        list_x=None,
        list_label=None,
        list_node_onehot=None,
        list_edge_onehot=None,
        train_fraction=0.8,
        seed=123):

    rng = random.Random(seed)
    index = list(range(len(graph_lis)))
    rng.shuffle(index)

    graph_lis = [graph_lis[i] for i in index]

    if list_x is not None:
        list_x = [list_x[i] for i in index]
    if list_label is not None:
        list_label = [list_label[i] for i in index]
    if list_node_onehot is not None:
        list_node_onehot = [list_node_onehot[i] for i in index]
    if list_edge_onehot is not None:
        list_edge_onehot = [list_edge_onehot[i] for i in index]

    # ── split ────────────────────────────────────────────────────
    n       = len(graph_lis)
    n_train = int(train_fraction * n)

    def split(lst):
        if lst is None:
            return None, None
        return lst[:n_train], lst[n_train:]

    graph_train,      graph_test      = split(graph_lis)
    list_x_train,     list_x_test     = split(list_x)
    list_label_train, list_label_test = split(list_label)
    list_noh_train,   list_noh_test   = split(list_node_onehot)
    list_eoh_train,   list_eoh_test   = split(list_edge_onehot)

    return (graph_train,      graph_test,
            list_x_train,     list_x_test,
            list_label_train, list_label_test,
            list_noh_train,   list_noh_test,
            list_eoh_train,   list_eoh_test)


def data_split_three_way(
    graph_lis,
    list_x=None,
    list_label=None,
    list_node_onehot=None,
    list_edge_onehot=None,
    train_fraction=0.7,
    val_fraction=0.1,
    seed=123,
):
    """Deterministic train/validation/test split for paper reproduction runs."""

    rng = random.Random(seed)
    index = list(range(len(graph_lis)))
    rng.shuffle(index)

    graph_lis = [graph_lis[i] for i in index]

    if list_x is not None:
        list_x = [list_x[i] for i in index]
    if list_label is not None:
        list_label = [list_label[i] for i in index]
    if list_node_onehot is not None:
        list_node_onehot = [list_node_onehot[i] for i in index]
    if list_edge_onehot is not None:
        list_edge_onehot = [list_edge_onehot[i] for i in index]

    n = len(graph_lis)
    n_train = int(train_fraction * n)
    n_val = int(val_fraction * n)
    train_slice = slice(0, n_train)
    val_slice = slice(n_train, n_train + n_val)
    test_slice = slice(n_train + n_val, n)

    def split(lst):
        if lst is None:
            return None, None, None
        return lst[train_slice], lst[val_slice], lst[test_slice]

    graph_train, graph_val, graph_test = split(graph_lis)
    list_x_train, list_x_val, list_x_test = split(list_x)
    list_label_train, list_label_val, list_label_test = split(list_label)
    list_noh_train, list_noh_val, list_noh_test = split(list_node_onehot)
    list_eoh_train, list_eoh_val, list_eoh_test = split(list_edge_onehot)

    return (
        graph_train, graph_val, graph_test,
        list_x_train, list_x_val, list_x_test,
        list_label_train, list_label_val, list_label_test,
        list_noh_train, list_noh_val, list_noh_test,
        list_eoh_train, list_eoh_val, list_eoh_test,
    )

# list_adj, list_x = list_graph_loader("GRID")
# list_graph = Datasets(list_adj,self_for_none, None)

def _apply_bfs_order(list_adj, list_node_feature, list_edge_feature, graph_idx, order):
    # adjacency
    list_adj[graph_idx] = list_adj[graph_idx][order, :][:, order]

    # node features: rows are nodes -> reorder rows
    if list_node_feature is not None and list_node_feature[graph_idx] is not None:
        list_node_feature[graph_idx] = list_node_feature[graph_idx][order, :]

    # edge features: remap src/dst node indices
    if list_edge_feature is not None and list_edge_feature[graph_idx] is not None:
        ef = list_edge_feature[graph_idx].copy()
        # Legacy BFS may keep only the component reachable from node 0. Drop
        # edge-feature rows whose endpoints were dropped from the adjacency.
        keep = np.isin(ef[:, 0], order) & np.isin(ef[:, 1], order)
        ef = ef[keep]
        original_num_nodes = int(max(
            int(np.max(order)) + 1 if len(order) else 0,
            int(np.max(ef[:, :2])) + 1 if ef.size else 0,
        ))
        inv_order = np.full(original_num_nodes, -1, dtype=np.int64)
        inv_order[order] = np.arange(len(order))
        ef[:, 0] = inv_order[ef[:, 0]]   # remap src
        ef[:, 1] = inv_order[ef[:, 1]]   # remap dst
        list_edge_feature[graph_idx] = ef


def BFS(list_adj, list_node_feature=None, list_edge_feature=None):
    """
    Legacy BFS ordering.

    Important: on disconnected graphs this keeps only the component reachable
    from node 0, because scipy.sparse.csgraph.breadth_first_order() returns
    only the visited nodes.
    """
    for i, _ in enumerate(list_adj):
        order = sp.csgraph.breadth_first_order(list_adj[i], 0)[0]
        _apply_bfs_order(list_adj, list_node_feature, list_edge_feature, i, order)

    return list_adj, list_node_feature, list_edge_feature


def BFS_all_components(list_adj, list_node_feature=None, list_edge_feature=None):
    """
    BFS ordering that preserves all nodes in disconnected graphs.

    Each connected component gets its own BFS traversal, and the final order is
    the concatenation of those traversals.
    """
    for i, _ in enumerate(list_adj):
        num_nodes = list_adj[i].shape[0]
        if num_nodes == 0:
            continue

        visited = np.zeros(num_nodes, dtype=bool)
        order_parts = []

        for start_node in range(num_nodes):
            if visited[start_node]:
                continue

            component_order = sp.csgraph.breadth_first_order(
                list_adj[i], start_node
            )[0]
            visited[component_order] = True
            order_parts.append(component_order)

        order = np.concatenate(order_parts) if order_parts else np.arange(num_nodes)
        _apply_bfs_order(list_adj, list_node_feature, list_edge_feature, i, order)

    return list_adj, list_node_feature, list_edge_feature

def BFSWithAug(list_adj,X_s, label_s, number_of_per = 1):
    list_adj_ = []
    X_s_ = []
    label_s_ = []
    for _ in range(number_of_per):
        for i, adj in enumerate(list_adj):
            mone_is_nodes = list(np.array(adj.sum(0)).reshape(-1))
            mone_is_nodes = [x for x in range(len(mone_is_nodes)) if mone_is_nodes[x] >= 1]
            node_i = random.choice(mone_is_nodes)
            bfs_index = scipy.sparse.csgraph.breadth_first_order(list_adj[i],node_i)
            list_adj_.append(list_adj[i][bfs_index[0],:][:,bfs_index[0]])


            X_s_.append(X_s[i])
            if label_s!=None:
                label_s_.append(label_s[i])
    if len(label_s_)==0:
        label_s_ = label_s
    return list_adj_, X_s_, label_s_

def permute(list_adj, X):
    for i, _ in enumerate(list_adj):
        p = list(range(list_adj[i].shape[0]))
        np.random.shuffle(p)

        list_adj[i] = list_adj[i][p, :]
        list_adj[i]= list_adj[i][:, p]
        # list_adj[i].eliminate_zeros()
        if X != None:
            X[i] = X[i][p, :]
            X[i] = X[i][:, p]
    return list_adj, X

def node_festure_creator(adj_in,steps=3, rand_dim=0, Use_identity = False, norm=None, uniform_size=False):

    if norm==None:
        norm=adj_in.shape[0]

    if not uniform_size:
        adj = adj_in
    else:
        adj = csr_matrix((norm, norm))
        adj[:adj_in.shape[0],:adj_in.shape[0]] +=adj_in

    traverse_matrix = adj
    featureVec=[np.array(adj.sum(1))/norm]
    for i in range(steps):
        traverse_matrix = traverse_matrix.dot(adj.transpose())
        feature = traverse_matrix.diagonal().reshape(-1,1)
        # converting it to one hot
        # one_hot = np.zeros((feature.size, int(feature.max()+1)))
        # one_hot[np.arange(one_hot.shape[0]),np.squeeze(np.asarray((feature).astype(int)))] = 1
        # one_hot.astype(int)
        featureVec.append(feature/norm**(i+1))
    if rand_dim>0:
        np.random.seed(0)
        featureVec.append(np.random.rand(adj.shape[-1], rand_dim))

    if Use_identity:
        featureVec.append(np.identity(norm))

    return numpy.concatenate(featureVec, 1)

def padd_adj_to(adj_list, size):
    uniformed_list = []
    for adj in adj_list:
        adj_padded = lil_matrix((size, size))
        adj_padded[:adj.shape[-1], :adj.shape[0]] = adj[:, :]
        adj_padded.setdiag(1)
        uniformed_list.append(adj_padded)
    return uniformed_list

def BFS_Permute( adj_s, x_s, target_kelrnel_val):
  for i in range(len(adj_s)):
      degree = np.array(adj_s[0].sum(0)).reshape(-1)
      connected_node = np.where(degree > 1)
      unconnected_nodes = np.where(degree == 1)

      bfs_index = scipy.sparse.csgraph.breadth_first_order(adj_s[i], random.choice(connected_node[0]))
      bfs_index = list(np.unique(bfs_index[0]) )+ list(unconnected_nodes[0])
      adj_s[i] = adj_s[i][bfs_index, :][:, bfs_index]
      x_s[i] = x_s[i][bfs_index, :]
      for j in range(len(target_kelrnel_val)-2):
          target_kelrnel_val[j][i] = target_kelrnel_val[j][i][bfs_index, :][:, bfs_index]


  return adj_s, x_s, target_kelrnel_val



if __name__ == '__main__':
    import numpy as np
    from itertools import combinations
    import plotter

    result = list_graph_loader("PVGAErandomGraphs")
    graph = np.load(data_path("PVGAErandomGraphs.npy"), allow_pickle=True)


    result = list_graph_loader("PVGAErandomGraphs_100000")

    for G in result[0]:
        G = nx.from_numpy_array(G.toarray())
        plotter.plotG(G,"DD")
    # ----------------------------------------
    import plotter
    result = list_graph_loader("TRIANGULAR_GRID")
    for G in result[0]:


        G = nx.from_numpy_array(G.toarray())
        plotter.plotG(G,"DD")
    #----------------------------------------
    result_ = list_graph_loader("QM9")
    result=list_graph_loader("NCI1")
    import plotter

    for i, G in enumerate(result[0]):
        G = nx.from_numpy_array(G.toarray())
        plotter.plotG(G, "test_graph", plot_it=True)

    result=list_graph_loader("TRIANGULAR_GRID")
    import plotter

    for i, G in enumerate(result[0]):
        G = nx.from_numpy_array(G.toarray())
        plotter.plotG(G, "test_graph")

    import torch_sparse
    import torch; print(torch.version.cuda)


    for i, graph in  enumerate(result[0]):
        print(nx.number_connected_components(nx.from_scipy_sparse_matrix(graph)))

    BFS(result[0])
    result = list_graph_loader("multi_community")
    Datasets(result[0], True, None,Max_num=None)
    Datasets.get__(0,2, True, None, None)
    for G in result[0]:


        G = nx.from_numpy_array(G.toarray())
        plotter.plotG(G,"DD")

"""
DataWrapper
===========
Reads directly from GraphVAE Datasets objects (list_graphs / list_test_graphs)
and produces a preprocessor-compatible object for
RelationalMotifCounter.count_batch().

Usage
-----
    merged = merge_datasets(list_graphs, list_test_graphs)  # sanity check
    merged = merge_datasets(list_graphs)                    # training only

    wrapper = DataWrapper(
        merged,
        motif_counter.relation_keys,
        node_onehot_info = node_onehot_info,
        device           = 'cuda',
    )
    counts     = motif_counter.count_batch(wrapper, batch_size=5000)
    aggregated = counts.sum(0)
"""



# ════════════════════════════════════════════════════════════════════════
#  Merge helper
# ════════════════════════════════════════════════════════════════════════

def merge_datasets(train_dataset, test_dataset=None):
    """
    Merge one or two Datasets objects into a plain dict of padded lists.

    Parameters
    ----------
    train_dataset : Datasets
    test_dataset  : Datasets | None
        # ── SANITY CHECK MERGE BLOCK ─────────────────────────────────
        # Pass test_dataset to merge train + test (sanity check mode).
        # Set test_dataset=None or omit it for training-only mode.
        # ── END SANITY CHECK MERGE BLOCK ─────────────────────────────
    """
    sources = [train_dataset]
    if test_dataset is not None:
        sources.append(test_dataset)

    def _get_list(ds, attr, default_len):
        lst = getattr(ds, attr, None)
        if lst:
            return list(lst)
        return [None] * default_len

    merged = {
        'processed_adjs':        [],
        'processed_Xs':          [],
        'processed_node_onehot': [],
        'processed_edge_onehot': [],
    }

    for ds in sources:
        n = len(ds.processed_adjs)
        merged['processed_adjs']        += list(ds.processed_adjs)
        merged['processed_Xs']          += list(ds.processed_Xs)
        merged['processed_node_onehot'] += _get_list(ds, 'processed_node_onehot', n)
        merged['processed_edge_onehot'] += _get_list(ds, 'processed_edge_onehot', n)

    merged['max_num_nodes'] = max(ds.max_num_nodes for ds in sources)

    n_train = len(train_dataset.processed_adjs)
    n_test  = len(test_dataset.processed_adjs) if test_dataset is not None else 0
    print(f"  [merge_datasets] {n_train} train"
          + (f" + {n_test} test = {n_train + n_test} total" if n_test else " (training only)")
          + f"  |  N_max={merged['max_num_nodes']}")

    return merged


# ════════════════════════════════════════════════════════════════════════
#  feature_onehot_mapping  built directly from node_onehot_info
# ════════════════════════════════════════════════════════════════════════

def _build_fom(node_onehot_info: Dict) -> Dict:
    """
    Build feature_onehot_mapping = {col_idx: {val_int: oh_col_idx}}
    directly from node_onehot_info.

    node_onehot_info structure:
        {oh_col_idx: {'feature_name': str, 'value': int}}

    e.g.  {0: {'feature_name': 'atom_type', 'value': 0},
           1: {'feature_name': 'atom_type', 'value': 1},
           ...
           5: {'feature_name': 'num_h',     'value': 0},
           ...}

    col_idx is the ORDER of first appearance of each feature name
    (atom_type appears first → col 0, num_h appears second → col 1).
    This matches the column order in list_node_feature exactly because
    both are produced from the same loop in list_graph_loader.

    Result for QM9:
        {0: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},   # atom_type
         1: {0: 5, 1: 6, 2: 7, 3: 8}}           # num_h
    """
    name_to_col: Dict[str, int] = {}
    col_counter = 0
    mapping: Dict[int, Dict[int, int]] = {}

    for oh_col in sorted(node_onehot_info.keys()):
        meta = node_onehot_info[oh_col]
        name = meta['feature_name']
        val  = int(meta['value'])

        # assign col_idx on first encounter of this feature name
        if name not in name_to_col:
            name_to_col[name] = col_counter
            col_counter += 1

        col_idx = name_to_col[name]
        if col_idx not in mapping:
            mapping[col_idx] = {}
        mapping[col_idx][val] = int(oh_col)

    return mapping


def _edge_feature_channel_groups(edge_onehot_info: Optional[Dict]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    if not edge_onehot_info:
        return groups

    for channel_idx, meta in sorted(edge_onehot_info.items()):
        feature_name = meta['feature_name']
        groups.setdefault(feature_name, []).append(int(channel_idx))
    return groups


def _split_edge_tensor_by_feature(
    edge_tensor: torch.Tensor,
    edge_onehot_info: Optional[Dict] = None,
    edge_feature_info_mapping: Optional[Dict] = None,
) -> List[torch.Tensor]:
    """
    Convert a packed edge channel tensor into the list layout expected by
    RelationalMotifCounter: edge_b[feature_idx] is one categorical feature.
    """
    if not edge_onehot_info:
        return [edge_tensor]

    groups = _edge_feature_channel_groups(edge_onehot_info)
    channel_by_feature_value = {
        (meta['feature_name'], int(meta['value'])): int(channel_idx)
        for channel_idx, meta in edge_onehot_info.items()
    }

    split_tensors: List[torch.Tensor] = []
    if edge_feature_info_mapping:
        feature_items = [
            (info['feature_name'], [
                int(value)
                for _value_idx, value in sorted(info['value_index_mapping'].items())
            ])
            for _idx, info in sorted(edge_feature_info_mapping.items())
        ]
    else:
        feature_items = [
            (feature_name, None)
            for feature_name in groups.keys()
        ]

    for feature_name, expected_values in feature_items:
        if expected_values is None:
            channels = groups.get(feature_name)
        else:
            channels = [
                channel_by_feature_value.get((feature_name, value))
                for value in expected_values
            ]
            if any(channel is None for channel in channels):
                missing_values = [
                    value for value, channel in zip(expected_values, channels)
                    if channel is None
                ]
                raise RuntimeError(
                    f"Edge feature '{feature_name}' is missing loaded values "
                    f"required by the motif cache: {missing_values}"
                )
        if not channels:
            raise RuntimeError(
                f"Edge feature '{feature_name}' is present in the motif cache, "
                "but not in the loaded graph edge features."
            )
        split_tensors.append(edge_tensor[:, channels, :, :])

    return split_tensors


# ════════════════════════════════════════════════════════════════════════
#  DataWrapper
# ════════════════════════════════════════════════════════════════════════

class DataWrapper:
    """
    Stacks already-padded Datasets lists into pin-memory CPU tensors
    matching the DataPreprocessor interface expected by count_batch.

    Parameters
    ----------
    merged : dict           — output of merge_datasets()
    relation_keys : list    — motif_counter.relation_keys  e.g. ['edges']
    node_onehot_info : dict — from build_onehot_features()
                              {oh_col: {'feature_name': str, 'value': int}}
    edge_onehot_info : dict — from build_onehot_features(), used to split
                              packed edge channels into counter feature groups
    device : str
    """

    def __init__(
        self,
        merged:           dict,
        relation_keys:    List[str],
        node_onehot_info: Optional[Dict] = None,
        edge_onehot_info: Optional[Dict] = None,
        edge_feature_info_mapping: Optional[Dict] = None,
        device:           str = 'cuda',
    ):
        self.device        = device
        self.relation_keys = relation_keys

        # ── feature_onehot_mapping ────────────────────────────────────
        if node_onehot_info:
            self.feature_onehot_mapping = _build_fom(node_onehot_info)
            print(f"  [DataWrapper] feature_onehot_mapping:")
            for col, val_map in sorted(self.feature_onehot_mapping.items()):
                print(f"    col {col} → {val_map}")
        else:
            self.feature_onehot_mapping = {}
            print("  [DataWrapper] Warning: node_onehot_info not provided "
                  "— feature_onehot_mapping is empty.")

        adjs     = merged['processed_adjs']
        Xs       = merged['processed_Xs']
        node_ohs = merged['processed_node_onehot']
        edge_ohs = merged['processed_edge_onehot']
        N_max    = int(merged['max_num_nodes'])

        self.num_graphs = len(adjs)
        self.N_max      = N_max

        print(f"  [DataWrapper] Stacking {self.num_graphs} graphs  N_max={N_max} ...")

        # ── features (G, N_max, F) ────────────────────────────────────
        self.all_features = _stack_2d(Xs, N_max)

        # ── node one-hot (G, N_max, D) ────────────────────────────────
        has_noh = any(x is not None for x in node_ohs)
        if has_noh:
            D = next(x for x in node_ohs if x is not None).shape[-1]
            self.all_feat_onehot  = _stack_2d(node_ohs, N_max, D=D)
            self.total_onehot_dim = D
        else:
            self.all_feat_onehot  = torch.zeros(
                self.num_graphs, N_max, 1).pin_memory()
            self.total_onehot_dim = 1

        # ── adjacency {rel: (G, N_max, N_max)} ───────────────────────
        stacked_adj  = _stack_adj(adjs, N_max)
        self.all_adj = {rk: stacked_adj for rk in relation_keys}

        # ── edge one-hot list[(G, C, N_max, N_max)] ──────────────────
        has_eoh = any(x is not None for x in edge_ohs)
        if has_eoh:
            C           = next(x for x in edge_ohs if x is not None).shape[0]
            stacked_eoh = _stack_3d(edge_ohs, C, N_max)
            self.all_edge = _split_edge_tensor_by_feature(
                stacked_eoh,
                edge_onehot_info=edge_onehot_info,
                edge_feature_info_mapping=edge_feature_info_mapping,
            )
            self.has_edge_features = True
        else:
            self.all_edge          = None
            self.has_edge_features = False

        edge_shapes = (
            "[" + ", ".join(str(tuple(edge.shape)) for edge in self.all_edge) + "]"
            if self.all_edge else "None"
        )
        print(f"  [DataWrapper] Ready."
              f"  features={tuple(self.all_features.shape)}"
              f"  onehot={tuple(self.all_feat_onehot.shape)}"
              f"  adj={tuple(stacked_adj.shape)}"
              f"  edge={edge_shapes}")

    # ------------------------------------------------------------------
    #  DataPreprocessor-compatible interface (called by count_batch)
    # ------------------------------------------------------------------

    def get_batch(self, start: int, end: int):
        """
        Returns
        -------
        feat_b        (B, N_max, F)
        feat_onehot_b (B, N_max, D)
        adj_b         {rel: (B, N_max, N_max)}
        edge_b        list[(B, C, N_max, N_max)] | None
        """
        dev = self.device
        kw  = dict(non_blocking=True)
        feat_b        = self.all_features[start:end].to(dev, **kw)
        feat_onehot_b = self.all_feat_onehot[start:end].to(dev, **kw)
        adj_b         = {rk: self.all_adj[rk][start:end].to(dev, **kw)
                         for rk in self.relation_keys}
        edge_b        = ([e[start:end].to(dev, **kw) for e in self.all_edge]
                         if self.all_edge is not None else None)
        return feat_b, feat_onehot_b, adj_b, edge_b


# ════════════════════════════════════════════════════════════════════════
#  Stacking helpers
# ════════════════════════════════════════════════════════════════════════

def _t(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.float().cpu()
    return torch.tensor(np.asarray(x, dtype=np.float32))


def _stack_2d(lst, N_max: int, D: int = None) -> torch.Tensor:
    """
    Stack list of (N_i, D) tensors/arrays (or None) → (G, N_max, D).
    D is inferred from first non-None entry if not provided.
    """
    first = next(x for x in lst if x is not None)
    D = D or _t(first).shape[-1]
    out = torch.zeros(len(lst), N_max, D)
    for g, x in enumerate(lst):
        if x is None:
            continue
        t = _t(x)
        n = min(t.shape[0], N_max)
        out[g, :n] = t[:n]
    return out.pin_memory()


def _stack_adj(adjs, N_max: int) -> torch.Tensor:
    """Stack list of sparse/dense (N_i, N_i) → (G, N_max, N_max)."""
    out = torch.zeros(len(adjs), N_max, N_max)
    for g, a in enumerate(adjs):
        t = torch.tensor(a.toarray(), dtype=torch.float32) if sp.issparse(a) else _t(a)
        n = min(t.shape[0], N_max)
        out[g, :n, :n] = t[:n, :n]
    return out.pin_memory()


def _stack_3d(edge_ohs, C: int, N_max: int) -> torch.Tensor:
    """Stack list of (C, N_i, N_i) | None → (G, C, N_max, N_max)."""
    out = torch.zeros(len(edge_ohs), C, N_max, N_max)
    for g, x in enumerate(edge_ohs):
        if x is None:
            continue
        t = _t(x)
        n = min(t.shape[1], N_max)
        out[g, :, :n, :n] = t[:, :n, :n]
    return out.pin_memory()


import torch
import torch.nn.functional as F
from typing import Dict, List, Optional


# ════════════════════════════════════════════════════════════════════════
#  ReconstructedDataWrapper
# ════════════════════════════════════════════════════════════════════════  

class ReconstructedDataWrapper:


    def __init__(
        self,
        reconstructed_adj: torch.Tensor,
        node_feat_logits: torch.Tensor,
        edge_feat_logits: Optional[torch.Tensor],
        relation_keys: List[str],
        node_onehot_info: Optional[Dict],
        feature_onehot_mapping: Dict,
        edge_onehot_info: Optional[Dict] = None,
        edge_feature_info_mapping: Optional[Dict] = None,
        adj_threshold: float = 0.5,
        use_soft_adj: bool = True,
        prob_temperature: float = 1.0,
        device: str = 'cuda',
    ):
        self.device = device
        self.relation_keys = relation_keys
        self.feature_onehot_mapping = feature_onehot_mapping
        self.adj_threshold = adj_threshold
        self.use_soft_adj = use_soft_adj
        self.prob_temperature = max(float(prob_temperature), 1e-3)


        adj = reconstructed_adj
        if adj.dim() == 4:
            adj = adj.squeeze(-1)
        assert adj.dim() == 3, \
            f"reconstructed_adj must be (B, N, N), got {adj.shape}"

        B, N, _ = adj.shape
        self.num_graphs = B
        self.N_max = N


        adj_min = adj.detach().min().item()
        adj_max = adj.detach().max().item()
        is_logit = (adj_min < -0.01) or (adj_max > 1.01)

        if is_logit:
            # Apply a temperature-scaled sigmoid so motif counting stays soft
            # early in training and becomes sharper later on.
            adj_soft = torch.sigmoid(adj / self.prob_temperature)
        else:
            adj_soft = adj  # already probabilities


        # adj_soft = (adj_soft + adj_soft.transpose(1, 2)) / 2.0

        # ── Store adjacency ───────────────────────────────────────────────────

        if use_soft_adj:
            adj_view = adj_soft
        else:
            # Evaluation-only hard view: threshold the decoded adjacency so the
            # motif counter sees the same discrete graph a human would inspect.
            adj_view = (adj_soft >= adj_threshold).to(adj_soft.dtype)

        self.all_adj = {rk: adj_view for rk in relation_keys}

        # ── Build node one-hot features ───────────────────────────────────────

        # if node_feat_logits is not None:
        nf = node_feat_logits 
        if nf.dim() == 2:
            D = nf.shape[-1]
            nf = nf.view(B, N, D)


        node_onehot_soft = self._apply_node_softmax(
            nf,
            node_onehot_info,
            temperature=self.prob_temperature,
        )
        if use_soft_adj:
            self.all_feat_onehot = node_onehot_soft  # (B, N, D)
        else:
            # For hard motif metrics, convert each categorical feature group to
            # a one-hot argmax so motif counting uses discrete labels.
            self.all_feat_onehot = self._harden_node_assignments(
                node_onehot_soft,
                node_onehot_info,
            )
        self.total_onehot_dim = self.all_feat_onehot.shape[-1]


        # ── Build node features (raw, non-onehot) ────────────────────────────

        if node_feat_logits is not None:
            self.all_features = node_feat_logits.view(B, N, -1)
        else:
            self.all_features = torch.zeros(B, N, 1, device=device)

        # ── Build edge features ───────────────────────────────────────────────
        # edge_feat_logits: (B, C, N, N) or None
        if edge_feat_logits is not None:
            ef = edge_feat_logits  # (B, C, N, N)
            assert ef.dim() == 4, \
                f"edge_feat_logits must be (B, C, N, N), got {ef.shape}"

            edge_logits_by_feature = _split_edge_tensor_by_feature(
                ef,
                edge_onehot_info=edge_onehot_info,
                edge_feature_info_mapping=edge_feature_info_mapping,
            )

            edge_views = []
            for edge_logits in edge_logits_by_feature:
                # Apply softmax within each categorical edge feature group.
                edge_soft = F.softmax(edge_logits / self.prob_temperature, dim=1)

                if use_soft_adj:
                    edge_views.append(edge_soft)
                else:
                    # Match the hard adjacency view with discrete edge-type labels.
                    edge_views.append(
                        F.one_hot(
                            edge_soft.argmax(dim=1),
                            num_classes=edge_soft.shape[1],
                        ).permute(0, 3, 1, 2).to(edge_soft.dtype)
                    )

            # all_edge is a list of tensors, each (B, C_feature, N, N)
            self.all_edge = edge_views
            self.has_edge_features = True
        else:
            self.all_edge = None
            self.has_edge_features = False

        edge_shapes = (
            "[" + ", ".join(str(tuple(edge.shape)) for edge in self.all_edge) + "]"
            if self.all_edge else "None"
        )
        print(
            f"[ReconstructedDataWrapper] Ready."
            f" B={B}, N_max={N}"
            f" adj={'soft' if use_soft_adj else 'hard'}"
            f" temp={self.prob_temperature:.3f}"
            f" node_onehot={tuple(self.all_feat_onehot.shape)}"
            f" edge={edge_shapes}"
        )

    def _harden_node_assignments(
        self,
        node_probs: torch.Tensor,
        node_onehot_info: Optional[Dict],
    ) -> torch.Tensor:
        """
        Convert grouped categorical node probabilities to discrete one-hot labels.
        """
        if node_onehot_info is None or len(node_onehot_info) == 0:
            hard_idx = torch.argmax(node_probs, dim=-1)
            return F.one_hot(hard_idx, num_classes=node_probs.shape[-1]).to(node_probs.dtype)

        output = torch.zeros_like(node_probs)
        feature_groups = {}
        covered_cols = set()

        for col_idx, info in sorted(node_onehot_info.items()):
            fname = info['feature_name']
            feature_groups.setdefault(fname, []).append(col_idx)

        for cols in feature_groups.values():
            cols_sorted = sorted(cols)
            covered_cols.update(cols_sorted)
            group_probs = node_probs[:, :, cols_sorted]
            group_idx = torch.argmax(group_probs, dim=-1)
            group_hard = F.one_hot(group_idx, num_classes=len(cols_sorted)).to(node_probs.dtype)
            output[:, :, cols_sorted] = group_hard

        if len(covered_cols) != node_probs.shape[-1]:
            hard_idx = torch.argmax(node_probs, dim=-1)
            return F.one_hot(hard_idx, num_classes=node_probs.shape[-1]).to(node_probs.dtype)

        return output

    def _apply_node_softmax(
        self,
        node_feat_logits: torch.Tensor,
        node_onehot_info: Optional[Dict],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply softmax per feature group to node feature logits.

        If node_onehot_info is provided, we know which columns belong to
        which categorical feature and apply softmax per group.
        Otherwise, apply softmax over the full last dimension.

        Parameters
        ----------
        node_feat_logits : (B, N, D)
        node_onehot_info : dict {oh_col: {'feature_name': str, 'value': int}}

        Returns
        -------
        Tensor (B, N, D) with softmax applied per feature group.
        """
        if node_onehot_info is None or len(node_onehot_info) == 0:
            # No info about feature groups → softmax over all
            return F.softmax(node_feat_logits / max(float(temperature), 1e-3), dim=-1)

        B, N, D = node_feat_logits.shape

        # Group columns by feature_name
        # node_onehot_info: {oh_col_idx: {'feature_name': str, 'value': int}}
        feature_groups = {}  # feature_name → list of column indices
        for col_idx, info in sorted(node_onehot_info.items()):
            fname = info['feature_name']
            if fname not in feature_groups:
                feature_groups[fname] = []
            feature_groups[fname].append(col_idx)

        # Apply softmax per group and reconstruct
        # We build a list of softmax'd slices and concatenate
        result_parts = []
        covered_cols = set()

        for fname, cols in sorted(feature_groups.items()):
            cols_sorted = sorted(cols)
            covered_cols.update(cols_sorted)
            # Extract the logits for this feature group
            group_logits = node_feat_logits[:, :, cols_sorted]  # (B, N, len(cols))
            # Apply softmax over the group dimension (differentiable)
            group_soft = F.softmax(group_logits / max(float(temperature), 1e-3), dim=-1)
            result_parts.append((cols_sorted[0], group_soft))

        # Sort by starting column index and concatenate
        result_parts.sort(key=lambda x: x[0])

        # Handle any uncovered columns (shouldn't happen, but just in case)
        all_covered = sorted(covered_cols)
        if len(all_covered) == D:
            # All columns are covered by feature groups
            output = torch.cat([part for _, part in result_parts], dim=-1)
        else:
            # Some columns not in any group → softmax over full tensor
            output = F.softmax(node_feat_logits / max(float(temperature), 1e-3), dim=-1)

        return output  # (B, N, D)

    def to(self, device: str):
        """Move all tensors to the specified device."""
        self.device = device
        self.all_adj = {k: v.to(device) for k, v in self.all_adj.items()}
        self.all_feat_onehot = self.all_feat_onehot.to(device)
        self.all_features = self.all_features.to(device)
        if self.all_edge is not None:
            self.all_edge = [e.to(device) for e in self.all_edge]
        return self
    def get_batch(self, start: int, end: int):
        """
        Returns
        -------
        feat_b        (B, N_max, F)
        feat_onehot_b (B, N_max, D)
        adj_b         {rel: (B, N_max, N_max)}
        edge_b        list[(B, C, N_max, N_max)] | None
        """
        dev = self.device
        kw  = dict(non_blocking=True)
        feat_b        = self.all_features[start:end].to(dev, **kw)
        feat_onehot_b = self.all_feat_onehot[start:end].to(dev, **kw)
        adj_b         = {rk: self.all_adj[rk][start:end].to(dev, **kw)
                         for rk in self.relation_keys}
        edge_b        = ([e[start:end].to(dev, **kw) for e in self.all_edge]
                         if self.all_edge is not None else None)
        return feat_b, feat_onehot_b, adj_b, edge_b

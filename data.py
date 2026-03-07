

from __future__ import annotations

import random

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

# import ogb


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
    for i in range(len(names)):
        load = pkl.load(open("data/Kernel_dataset/ind.{}.{}".format(dataset, names[i]), 'rb'), encoding='latin1')
        # print('loaded')
        objects.append(load)
        # print(load)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/Kernel_dataset/ind.{}.test.index".format(dataset))
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
def list_graph_loader( graph_type, _max_list_size=None, return_labels=False, limited_to=None):
  list_adj = []
  list_x =[]
  list_labels = []
  list_node_feature = []    
  list_edge_feature = []    

  if graph_type=="IMDBBINARY":
      data = dgl.data.GINDataset(name='IMDBBINARY', self_loop=False)
      graphs, labels = data.graphs, data.labels
      for i, graph in enumerate(graphs):
          list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
          # list_x.append(graph.ndata['feat'])
          list_x.append(None)
          list_labels.append(labels[i].cpu().item())
      graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
      np.save('IMDBBINARY_lattice_graph.npy', graphs_to_writeOnDisk, allow_pickle=True)

  elif graph_type=="NCI1":
      data = dgl.data.GINDataset(name='NCI1', self_loop=False)
      graphs, labels = data.graphs, data.labels
      for i, graph in enumerate(graphs):
          list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
          # list_x.append(graph.ndata['feat'])
          list_x.append(None)
          list_labels.append(labels[i].cpu().item())
      graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
      np.save('NCI1_lattice_graph.npy', graphs_to_writeOnDisk, allow_pickle=True)
  elif graph_type=="MUTAG":
      data = dgl.data.GINDataset(name='MUTAG', self_loop=False)
      graphs, labels = data.graphs, data.labels
      for i, graph in enumerate(graphs):
          list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
          # list_x.append(graph.ndata['feat'])
          list_x.append(None)
          list_labels.append(labels[i].cpu().item())
      graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
      np.save('MUTAG_lattice_graph.npy', graphs_to_writeOnDisk, allow_pickle=True)
  elif graph_type=="COLLAB":
      data = dgl.data.GINDataset(name='COLLAB', self_loop=False)
      graphs, labels = data.graphs, data.labels
      for i, graph in enumerate(graphs):
          list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
          # list_x.append(graph.ndata['feat'])
          list_x.append(None)
          list_labels.append(labels[i].cpu().item())
      graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
      # np.save('COLLAB_lattice_graph.npy', graphs_to_writeOnDisk, allow_pickle=True)
  elif graph_type=="PTC":
      data = dgl.data.GINDataset(name='PTC', self_loop=False)
      graphs, labels = data.graphs, data.labels
      for i, graph in enumerate(graphs):
          list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
          # list_x.append(graph.ndata['feat'])
          list_x.append(None)
          list_labels.append(labels[i].cpu().item())
      graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
      np.save('PTC_lattice_graph.npy', graphs_to_writeOnDisk, allow_pickle=True)
  elif graph_type == "PROTEINS":
      data = dgl.data.GINDataset(name='PROTEINS', self_loop=False)
      graphs, labels = data.graphs, data.labels
      for i, graph in enumerate(graphs):
          if graph.adjacency_matrix().shape[0]<100:
              list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
              # list_x.append(graph.ndata['feat'])
              list_x.append(None)
              list_labels.append(labels[i].cpu().item())
      # graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
      # np.save('PROTEINS.npy', graphs_to_writeOnDisk, allow_pickle=True)
  elif graph_type == "QM9":
    #   data = dgl.data.QM9Dataset(label_keys=['mu'])
    #   for i, graph in enumerate(data):
    #       # if i==1000:
    #       #     break
    #       adj = graph[0].adj().to_dense().cpu().numpy()
    #       list_adj.append(scipy.sparse.csr_matrix(adj))
    #       list_x.append(None)
    #       list_labels.append(None)
    #       print(i)

        from torch_geometric.datasets import QM9
        data = QM9(root="./data/QM9")

        # ── node feature metadata ─────────────────────────────────
        # Two columns: col-0 = atom_type (0-4), col-1 = num_h (0-3)
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
            atom_type = torch.argmax(X[:, 0:5], dim=1)
            num_h     = torch.clamp(X[:, 10].long(), max=3)
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
#   elif graph_type=="ogbg-molbbbp":
#       # https://ogb.stanford.edu/docs/graphprop/
#       from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
#       d_name = "ogbg-molbbbp"  # ogbg-molhiv   'ogbg-code2' ogbg-ppa
#       dataset = DglGraphPropPredDataset(name=d_name)


#       list_adj = []
#       for graph, label in dataset:
#           list_adj.append(csr_matrix(graph.adjacency_matrix().to_dense().numpy()))
#           # list_x.append(graph.ndata['feat'])
#           list_x.append(None)
#           list_labels.append(label.cpu().item())

#       # graphs_to_writeOnDisk = [gr.toarray() for gr in list_adj]
#       # np.save('ogbg-molbbbp.npy', graphs_to_writeOnDisk, allow_pickle=True)


      # list_labels = [adj.sum() for adj in list_adj]
  elif graph_type=="large_grid":
      for i in range(10):
            list_adj.append(nx.adjacency_matrix(grid(30, 100)))
            list_x.append(None)
  elif graph_type=="grid":
      for i in range(10, 20):
        for j in range(10, 20):
            list_adj.append(nx.adjacency_matrix(grid(i, j)))
            list_x.append(None)

  elif graph_type=="triangular_grid":
      for i in range(10, 20):
        for j in range(10, 20):
            list_adj.append(nx.adjacency_matrix(nx.triangular_lattice_graph(i, j)))
            list_x.append(None)
      # graphs_to_writeOnDisk = [gr.toarray() for  gr in list_adj]
      # np.save('triangular_lattice_graph.npy', graphs_to_writeOnDisk, allow_pickle=True)
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
      list_adj = pkl.load(open("data/IMDbMulti/IMDBMulti.p",'rb'))
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
  elif graph_type == 'lobster':
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
              list_adj.append(nx.adjacency_matrix(G))
              list_x.append(None)
              count += 1
          seed_tmp += 1
      # writing the generated graph for benchmarking
      # graphs_to_writeOnDisk = [gr.toarray() for  gr in list_adj]
      # np.save('Lobster_adj.npy', graphs_to_writeOnDisk, allow_pickle=True)
  elif graph_type=="mnist":
      list_adj = []
      list_x = []
      import torch_geometric
      dataset_b = torch_geometric.datasets.MNISTSuperpixels(root="data/geometric")
      for i in range(len(dataset_b.data.y)):  # len(dataset_b.data.y)
          in_1 = dataset_b[i].edge_index[0].detach().numpy()
          in_2 = dataset_b[i].edge_index[1].detach().numpy()
          valu = numpy.ones(len(in_2))
          adj = scipy.sparse.csr_matrix((valu, (in_1, in_2)), shape=(dataset_b[i].num_nodes, dataset_b[i].num_nodes))
          list_adj.append(adj)
          list_x.append(None)
  elif graph_type == "zinc":
      import torch_geometric
      dataset_b = torch_geometric.datasets.ZINC(root="data/geometric/MoleculeNet/zinc", subset=False)
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
        "data/Kernel_dataset/",
        min_num_nodes=0,
        max_num_nodes=2000,
        name='FIRSTMM_DB',
        node_attributes=False,
        graph_labels=True)

  elif graph_type == 'DD':
    list_adj, list_x, list_labels  = graph_load_batch(
        "data/Kernel_dataset/",
        min_num_nodes=100,
        max_num_nodes=500,
        name='DD',
        node_attributes=False,
        graph_labels=True)
    # args.max_prev_node = 230



  def return_subset(A, X, Y, NF, EF, limited_to):
        indx = list(range(len(A)))
        random.shuffle(indx)
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

  list_adj, list_x, list_labels, list_node_feature, list_edge_feature = \
      return_subset(list_adj, list_x, list_labels, list_node_feature, list_edge_feature, limited_to)

  return (list_adj, list_x, list_labels,
          list_node_feature, list_edge_feature,
          node_feature_info, edge_feature_info)

def data_split(graph_lis, list_x=None, list_label=None,
               list_node_onehot=None, list_edge_onehot=None):

    random.seed(123)
    index = list(range(len(graph_lis)))
    random.shuffle(index)

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
    n_train = int(0.8 * n)

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

# list_adj, list_x = list_graph_loader("grid")
# list_graph = Datasets(list_adj,self_for_none, None)

def BFS(list_adj, list_node_feature=None, list_edge_feature=None):
    for i, _ in enumerate(list_adj):
        order = scipy.sparse.csgraph.breadth_first_order(list_adj[i], 0)[0]

        # adjacency
        list_adj[i] = list_adj[i][order, :][:, order]

        # node features: rows are nodes → reorder rows
        if list_node_feature is not None and list_node_feature[i] is not None:
            list_node_feature[i] = list_node_feature[i][order, :]

        # edge features: remap src/dst node indices
        if list_edge_feature is not None and list_edge_feature[i] is not None:
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(len(order))

            ef = list_edge_feature[i].copy()
            ef[:, 0] = inv_order[ef[:, 0]]   # remap src
            ef[:, 1] = inv_order[ef[:, 1]]   # remap dst
            list_edge_feature[i] = ef

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
    graph = np.load('C:\git\GRANon13\data/PVGAErandomGraphs.npy', allow_pickle=True)


    result = list_graph_loader("PVGAErandomGraphs_100000")

    for G in result[0]:
        G = nx.from_numpy_array(G.toarray())
        plotter.plotG(G,"DD")
    # ----------------------------------------
    import plotter
    result = list_graph_loader("triangular_grid")
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

    result=list_graph_loader("triangular_grid")
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
    device : str
    """

    def __init__(
        self,
        merged:           dict,
        relation_keys:    List[str],
        node_onehot_info: Optional[Dict] = None,
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
            self.all_edge          = [stacked_eoh]
            self.has_edge_features = True
        else:
            self.all_edge          = None
            self.has_edge_features = False

        print(f"  [DataWrapper] Ready."
              f"  features={tuple(self.all_features.shape)}"
              f"  onehot={tuple(self.all_feat_onehot.shape)}"
              f"  adj={tuple(stacked_adj.shape)}"
              + (f"  edge={tuple(self.all_edge[0].shape)}"
                 if self.all_edge else "  edge=None"))

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

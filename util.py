import torch as torch

# import dgl
# from dgl.nn.pytorch import GraphConv as GraphConv
import scipy.sparse as sp
import numpy as np
from torch.nn import init


# torch.seed()
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# torch._set_deterministic(True)
#************************************************************
"""some utiliees including some mudules to apply neural net blocks 
on the matrixes(relationl data)"""

#************************************************************

class node_mlp(torch.nn.Module):
    """
    This layer apply a chain of mlp on each node of tthe graph.
    thr input is a matric matrrix with n rows whixh n is the nide number.
    """
    def __init__(self, input, layers= [16, 16], normalize = False, dropout_rate = 0):
        """

        :param input: the feture size of input matrix; Number of the columns
        :param normalize: either use the normalizer layer or not
        :param layers: a list which shows the ouyput feature size of each layer; Note the number of layer is len(layers)
        """
        super(node_mlp, self).__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(input, layers[0])])

        for i in range(len(layers)-1):
            self.layers.append(torch.nn.Linear(layers[i],layers[i+1]))

        self.norm_layers = None
        if normalize:
            self.norm_layers =  torch.nn.ModuleList([torch.nn.BatchNorm1d(c) for c in [input]+layers])
        self.dropout = torch.nn.Dropout(dropout_rate)
        # self.reset_parameters()

    def forward(self, in_tensor, activation = torch.tanh, applyActOnTheLastLyr=True):
        h = in_tensor
        for i in range(len(self.layers)):
            if self.norm_layers!=None:
                if len(h.shape)==2:
                    h = self.norm_layers[i](h)
                else:
                    shape = h.shape
                    h= h.reshape(-1, h.shape[-1])
                    h = self.norm_layers[i](h)
                    h=h.reshape(shape)
            h = self.dropout(h)
            h = self.layers[i](h)
            if i != (len(self.layers)-1) or applyActOnTheLastLyr:
                h = activation(h)
        return h

class Graph_mlp(torch.nn.Module):
        """
        This layer apply a chain of mlp on each node of tthe graph.
        thr input is a matric matrrix with n rows whixh n is the nide number.
        """

        def __init__(self, input,  layers=[1024], normalize=False, dropout_rate=0):
            """

            :param input: the feture size of input matrix; Number of the columns
            :param normalize: either use the normalizer layer or not
            :param layers: a list which shows the ouyput feature size of each layer; Note the number of layer is len(layers)
            """
            super(Graph_mlp, self).__init__()

            layers = [input] + layers
            self.Each_neuron = torch.nn.ModuleList([torch.nn.Linear(layers[i],layers[i+1]) for i in range(len(layers)-1)])

        def forward(self, in_tensor, activation=torch.tanh):
            z = in_tensor
            for i in range(len(self.Each_neuron)):
                z = self.Each_neuron[i](z)
                if i !=(len(self.Each_neuron)-1):
                    z = activation(z)
            z = torch.mean(z,1)
            z = activation(z)
            return z

class poolingLayer_average(torch.nn.Module):
    """
    This layer apply a chain of mlp on each node of tthe graph.
    thr input is a matric matrrix with n rows whixh n is the nide number.
    """

    def __init__(self, input,):
        super(Graph_mlp, self).__init__()

    def forward(self, in_tensor, activation=torch.tanh):
        in_tensor = torch.mean(in_tensor,1)
        in_tensor = activation(in_tensor)
        return in_tensor


# class node_mlp(torch.nn.Module):
#     """
#     This laye applt a chain of mlp on each node of tthe graph.
#     This layer apply a chain of mlp on each node of tthe graph.
#     thr input is a matric matrrix with n rows whixh n is the nide number.
#     """
#     def __init__(self, input, layers= [16, 16]):
#         """
#         :param input: the feture size of input matrix; Number of the columns
#         :param layers: a list which shows the ouyput feature size of each layer; Note the number of layer is len(layers)
#         """
#         super(node_mlp, self).__init__()
#         self.layers = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor(input, layers[0]))])
#         for i in range(len(layers)-1):
#             self.layers.append(torch.nn.Parameter(torch.Tensor(layers[i],layers[i+1])))
#         self.reset_parameters()
#     def forward(self, in_tensor,activation = torch.tanh ):
#         h = in_tensor
#
#         for layer in self.layers:
#             torch.matmul(h, layer)
#             h = activation(h)
#         return h
#     def reset_parameters(self):
#         for i, weight in enumerate(self.layers):
#             self.layers[i] = init.xavier_uniform_(weight)



class edge_mlp(torch.nn.Module):
    """
    this layer applies Multi layer perceptron on each edge of the graph.
    the input of the layer is a 3 dimentional tensor in which
    the third dimention is feature vector of each mode.
    """
    def __init__(self, input, layers = [8, 4], activation = torch.tanh, last_layer_activation= torch.sigmoid):
        super(edge_mlp, self).__init__()
        """
        Construct the graph mlp
        Args:
            layer: a list whcih determine the number of layers and
            the number of neurons in each layer.
            input: The size of the third dimention of Tensor
        """
        self.activation = activation
        self.last_layer_activation = last_layer_activation
        self.mlp_layers = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor(input, layers[0]))])
        for i in range(len(layers)-1):
            self.mlp_layers.append(torch.nn.Parameter(torch.Tensor(layers[i],layers[i+1])))

        self.reset_parameters()

    def forward(self, in_tensor) :
        h = in_tensor
        for index,layer in enumerate(self.mlp_layers):
            h = torch.matmul(h, layer)
            if index<(len(self.mlp_layers)-1): h= self.activation(h)
            else: h = self.last_layer_activation(h)
        return torch.squeeze(h)

    def reset_parameters(self):
        for i, weight in enumerate(self.mlp_layers):
            self.mlp_layers[i] = init.xavier_uniform_(weight)

#=======================================================================================
# Added for edge feature decoding

class NodeFeatureDecoder(torch.nn.Module):
    def __init__(self, graphEmDim, max_nodes, node_D, hidden=512):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_D    = node_D
        self.net = torch.nn.Sequential(
            torch.nn.Linear(graphEmDim, hidden),
            torch.nn.LeakyReLU(0.01),
            torch.nn.LayerNorm(hidden, elementwise_affine=False),
            torch.nn.Linear(hidden, max_nodes * node_D),
        )

    def forward(self, z):
        out = self.net(z)                                   # (B, max_nodes*node_D)
        return out.reshape(z.shape[0], self.max_nodes, self.node_D)


class EdgeFeatureDecoder(torch.nn.Module):
    def __init__(self, graphEmDim, max_nodes, edge_C, hidden=512):
        super().__init__()
        self.max_nodes = max_nodes
        self.edge_C    = edge_C
        self.net = torch.nn.Sequential(
            torch.nn.Linear(graphEmDim, hidden),
            torch.nn.LeakyReLU(0.01),
            torch.nn.LayerNorm(hidden, elementwise_affine=False),
            torch.nn.Linear(hidden, edge_C * max_nodes * max_nodes),
        )

    def forward(self, z):
        out = self.net(z)                                   # (B, C*max_nodes*max_nodes)
        return out.reshape(z.shape[0], self.edge_C, self.max_nodes, self.max_nodes)
#=======================================================================================


# GCN basic operation
# class GraphConvNN(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(GraphConvNN, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.weight = torch.nn.Parameter(torch.FloatTensor(input_dim, output_dim))
#         self.reset_parameters()
#
#     def forward(self,adj, x, sparse= False):
#         """
#         :param adj: normalized adjacency matrix of graph
#         :param x: normalized node feature matrix
#         :param sparse: either the adj is a sparse matrix or not
#         :return:
#         """
#         y = torch.matmul( adj, x)
#         y = torch.spmm(y,self.weight) if sparse else torch.matmul(y,self.weight)
#         return y
#
#     def reset_parameters(self):
#         self.weight = init.xavier_uniform_(self.weight)

class GraphConvNN(torch.nn.Module):
    r"""Apply graph convolution over an input signal.

    Graph convolution is introduced in `GCN <https://arxiv.org/abs/1609.02907>`__
    and can be described as below:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the neighbor set of node :math:`i`. :math:`c_{ij}` is equal
    to the product of the square root of node degrees:
    :math:`\sqrt{|\mathcal{N}(i)|}\sqrt{|\mathcal{N}(j)|}`. :math:`\sigma` is an activation
    function.

    The model parameters are initialized as in the
    `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__ where
    the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
    and the bias is initialized to be zero.

    Notes
    -----
    Zero in degree nodes could lead to invalid normalizer. A common practice
    to avoid this is to add a self-loop for each node in the graph, which
    can be achieved by:

    >>> g = ... # some DGLGraph
    >>> g.add_edges(g.nodes(), g.nodes())


    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    norm : str, optional
        How to apply the normalizer. If is `'right'`, divide the aggregated messages
        by each node's in-degrees, which is equivalent to averaging the received messages.
        If is `'none'`, no normalization is applied. Default is `'both'`,
        where the :math:`c_{ij}` in the paper is applied.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=False,
                 activation=None):
        super(GraphConvNN, self).__init__()
        if norm not in ('none', 'both', 'right'):
            raise ('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        if weight:
            self.weight = torch.nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, feat, weight=None):
        r"""Compute graph convolution.

        Notes
        -----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: "math:`(\text{in_feats}, \text{out_feats})`.

        Parameters
        ----------
        graph : DGLGraph
            The adg of graph. It should include self loop
        feat : torch.Tensor
            The input feature
        weight : torch.Tensor, optional
            Optional external weight tensor.

        Returns
        -------
        torch.Tensor
            The output feature
        """


        if self._norm == 'both':
            degs = graph.sum(-2).float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,)
            norm = torch.reshape(norm, shp)
            feat = feat * norm

        if weight is not None:
            if self.weight is not None:
                raise ('External weight is provided while at the same time the'
                               ' module has defined its own weight parameter. Please'
                               ' create the module with flag weight=False.')
        else:
            weight = self.weight

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            if weight is not None:
                feat = torch.matmul(feat, weight)
            # graph.srcdata['h'] = feat
            # graph.update_all(fn.copy_src(src='h', out='m'),
            #                  fn.sum(msg='m', out='h'))
            rst = torch.matmul(graph, feat)
        else:
            # aggregate first then mult W
            # graph.srcdata['h'] = feat
            # graph.update_all(fn.copy_src(src='h', out='m'),
            #                  fn.sum(msg='m', out='h'))
            # rst = graph.dstdata['h']
            rst = torch.matmul(graph, feat)
            if weight is not None:
                rst = torch.matmul(rst, weight)

        if self._norm != 'none':
            degs = graph.sum(-1).float().clamp(min=1)
            if self._norm == 'both':
                norm = torch.pow(degs, -0.5)
            else:
                norm = 1.0 / degs
            shp = norm.shape + (1,)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def preprocess_graph(adj):
    rowsum = np.array(adj.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return adj_normalized

class Learnable_Histogram(torch.nn.Module):
    def __init__(self, bin_num):
        super(Learnable_Histogram, self).__init__()
        self.bin_num = bin_num
        self.bin_width = torch.nn.Parameter(torch.Tensor(bin_num,1))
        self.bin_center = torch.nn.Parameter(torch.Tensor(bin_num,1))
        self.reset_parameters()

    def forward(self, vec):
        score_vec = vec-self.bin_center
        score_vec = 1-torch.abs(score_vec)*torch.abs(self.bin_width)
        score_vec = torch.relu(score_vec)
        score_vec
        return score_vec

    def reset_parameters(self):
        self.bin_width = torch.nn.init.xavier_uniform_(self.bin_width)
        self.bin_center = torch.nn.init.xavier_uniform_(self.bin_center)






def build_onehot_features(list_node_feature, list_edge_feature, list_adj,
                          node_feature_info, edge_feature_info):
    """
    Generic one-hot encoder for node and edge features.
    Works for ANY dataset that follows the standard format — not QM9-specific.

    Input format contract
    ---------------------
    list_node_feature[i] : (N, F) int numpy array   — one column per feature
                           None if the dataset has no node features
    list_edge_feature[i] : (E, 2+F) int numpy array — col0=src, col1=dst,
                           col2=feat0, col3=feat1, ...
                           None if the dataset has no edge features
    node_feature_info    : { col_idx: { 'feature_name': str } }
                           None if no node features
    edge_feature_info    : { feat_idx: { 'feature_name': str,
                                         'unique_values': [int,...] } }
                           None if no edge features

    Returns
    -------
    list_node_onehot : list of (N, D) float32 numpy arrays  (or None per graph)
    list_edge_onehot : list of (C, N, N) float32 numpy arrays (or None per graph)

    node_onehot_info : dict — maps one-hot columns back to source feature/value
        {
          onehot_col_idx: {
            'feature_name': str,
            'source_col':   int,   # which column of list_node_feature[i]
            'value':        int,   # the raw value this column encodes
          }
        }
        Example (QM9):
        {
          0: {'feature_name': 'atom_type', 'source_col': 0, 'value': 0},
          1: {'feature_name': 'atom_type', 'source_col': 0, 'value': 1},
          ...
          5: {'feature_name': 'num_h',     'source_col': 1, 'value': 0},
          ...
        }

    edge_onehot_info : dict — maps (C, N, N) slices back to source feature/value
        {
          slice_idx: {
            'feature_name': str,
            'source_col':   int,   # which column of list_edge_feature[i] (2, 3, ...)
            'value':        int,   # the raw value this slice encodes
          }
        }
        Example (QM9):
        {
          0: {'feature_name': 'bond_type', 'source_col': 2, 'value': 0},
          1: {'feature_name': 'bond_type', 'source_col': 2, 'value': 1},
          2: {'feature_name': 'bond_type', 'source_col': 2, 'value': 2},
          3: {'feature_name': 'bond_type', 'source_col': 2, 'value': 3},
        }
    """

    # ── PART 1: node one-hot ──────────────────────────────────────────────
    node_onehot_info = None
    list_node_onehot = [None] * len(list_node_feature)

    has_node_features = (
        node_feature_info is not None
        and any(nf is not None for nf in list_node_feature)
    )

    if has_node_features:
        num_cols = len(node_feature_info)

        # Step 1a: discover unique values per column across ALL graphs
        col_unique = {col: set() for col in range(num_cols)}
        for nf in list_node_feature:
            if nf is None:
                continue
            for col in range(num_cols):
                col_unique[col].update(int(v) for v in np.unique(nf[:, col]))

        # Step 1b: assign one-hot column indices and build info dict
        node_onehot_info = {}   # onehot_col -> metadata
        # internal lookup: col -> { raw_val -> onehot_col_idx }
        col_val_to_oh = {}
        oh_col = 0

        for col in range(num_cols):
            feature_name = node_feature_info[col]['feature_name']
            col_val_to_oh[col] = {}
            for val in sorted(col_unique[col]):
                node_onehot_info[oh_col] = {
                    'feature_name': feature_name,
                    'value':        val,
                }
                col_val_to_oh[col][val] = oh_col
                oh_col += 1

        D = oh_col  # total one-hot dimension

        # Step 1c: encode every graph
        for g_idx, nf in enumerate(list_node_feature):
            if nf is None:
                continue
            N      = nf.shape[0]
            onehot = np.zeros((N, D), dtype=np.float32)
            for col in range(num_cols):
                val_map = col_val_to_oh[col]
                for node_i in range(N):
                    raw_val = int(nf[node_i, col])
                    if raw_val in val_map:
                        onehot[node_i, val_map[raw_val]] = 1.0
            list_node_onehot[g_idx] = onehot   # (N, D)

    # ── PART 2: edge one-hot adjacency ───────────────────────────────────
    edge_onehot_info = None
    list_edge_onehot = [None] * len(list_edge_feature)

    has_edge_features = (
        edge_feature_info is not None
        and any(ef is not None for ef in list_edge_feature)
    )

    if has_edge_features:
        # Step 2a: assign adjacency-slice indices and build info dict
        # edge_feature_info already contains unique_values per feature
        edge_onehot_info = {}   # slice_idx -> metadata
        # internal lookup: feat_idx -> { raw_val -> slice_idx }
        feat_val_to_slice = {}
        slice_idx = 0

        for feat_idx in sorted(edge_feature_info.keys()):
            info         = edge_feature_info[feat_idx]
            feature_name = info['feature_name']
            source_col   = 2 + feat_idx   # col0=src, col1=dst, col2=feat0, ...
            feat_val_to_slice[feat_idx] = {}

            for val in sorted(info['unique_values']):
                edge_onehot_info[slice_idx] = {
                    'feature_name': feature_name,
                    'value':        val,
                }
                feat_val_to_slice[feat_idx][val] = slice_idx
                slice_idx += 1

        C = slice_idx   # total number of adjacency slices

        # Step 2b: encode every graph
        for g_idx, (ef, adj) in enumerate(zip(list_edge_feature, list_adj)):
            N          = adj.shape[0]
            onehot_adj = np.zeros((C, N, N), dtype=np.float32)

            if ef is None:
                list_edge_onehot[g_idx] = onehot_adj
                continue

            srcs = ef[:, 0].astype(int)
            dsts = ef[:, 1].astype(int)

            for feat_idx in sorted(edge_feature_info.keys()):
                source_col = 2 + feat_idx
                val_map    = feat_val_to_slice[feat_idx]
                vals       = ef[:, source_col].astype(int)

                for s, d, v in zip(srcs, dsts, vals):
                    if v in val_map:
                        onehot_adj[val_map[v], s, d] = 1.0

            list_edge_onehot[g_idx] = onehot_adj   # (C, N, N)

    return list_node_onehot, list_edge_onehot, node_onehot_info, edge_onehot_info


# ===== SAVE CHECKPOINT =====

# data_to_save = {
#     "list_adj": list_adj,
#     "list_x": list_x,
#     "list_label": list_label,
#     "list_node_feature": list_node_feature,
#     "list_edge_feature": list_edge_feature,
#     "node_feature_info": node_feature_info,
#     "edge_feature_info": edge_feature_info,
#     "list_node_onehot": list_node_onehot,
#     "list_edge_onehot": list_edge_onehot,
#     "node_onehot_info": node_onehot_info,
#     "edge_onehot_info": edge_onehot_info,
# }

# save_path = os.path.join(save_dir, "graph_data.pkl")

# with open(save_path, "wb") as f:
#     pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

# print(f"✅ Saved to {save_path}")


# import pickle
# import os

# save_dir = "saved_state"
# os.makedirs(save_dir, exist_ok=True)

# save_path = os.path.join(save_dir, "dataset_checkpoint.pkl")

# with open(save_path, "wb") as f:
#     pickle.dump({
#         "self_for_none": self_for_none,
#         "list_adj": list_adj,
#         "test_list_adj": test_list_adj,
#         "list_x_train": list_x_train,
#         "list_x_test": list_x_test,
#         "list_label_train": list_label_train,
#         "list_label_test": list_label_test,
#         "list_noh_train": list_noh_train,
#         "list_noh_test": list_noh_test,
#         "list_eoh_train": list_eoh_train,
#         "list_eoh_test": list_eoh_test,
#         "list_graphs": list_graphs,
#         "list_test_graphs": list_test_graphs,
#     }, f, protocol=pickle.HIGHEST_PROTOCOL)

# print(f"✅ Checkpoint saved to {save_path}")




def remove_self_loops(dataset):
    """
    Zero out diagonal (self-loop) entries in all processed adjacency matrices
    and edge one-hot tensors of a Datasets object. Operates in-place.
    """
    for g in range(len(dataset.processed_adjs)):
        # ── adjacency ──────────────────────────────────────────────────
        adj = dataset.processed_adjs[g]
        if sp.issparse(adj):
            adj = adj.tolil()
            adj.setdiag(0)
            adj = adj.tocsr()
            adj.eliminate_zeros()
            dataset.processed_adjs[g] = adj
        else:
            # lil_matrix or dense numpy/tensor
            if torch.is_tensor(adj):
                adj.fill_diagonal_(0)
            else:
                np.fill_diagonal(
                    adj.toarray() if sp.issparse(adj) else np.asarray(adj), 0
                )
                dataset.processed_adjs[g] = adj

        # ── edge one-hot: (C, N, N) — zero diagonal slices ────────────
        if (hasattr(dataset, 'processed_edge_onehot')
                and dataset.processed_edge_onehot
                and dataset.processed_edge_onehot[g] is not None):
            eoh = dataset.processed_edge_onehot[g]   # (C, N, N)
            for c in range(eoh.shape[0]):
                np.fill_diagonal(eoh[c], 0)
                
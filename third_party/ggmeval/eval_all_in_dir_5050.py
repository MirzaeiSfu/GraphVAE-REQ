# Generate graphs for demonstration purposes
import utils.graph_generators as gen
import pickle
import torch
import dgl
import tensorflow as tf
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt

def preprocess_graphs(list_of_NXGraph):
    # remove self loops and isolated graphs
    for G in list_of_NXGraph:
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))

    list_of_NXGraph = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in
                       list_of_NXGraph if not nx.is_empty(G)]
    list_of_NXGraph = selfLoop(list_of_NXGraph)
    return list_of_NXGraph

def selfLoop(graph_list):
    # graph_list = [graph.add_edges_from([(i,i) for i in graph.number_of_nodes]) for graph in graph_list]
    for graph in graph_list:
        graph.add_edges_from([(i,i) for i in range(graph.number_of_nodes())])
    return  graph_list



def load_graphs(graph_pkl):
    import pickle5 as cp
    graphs = []
    with open(graph_pkl, 'rb') as f:
        while True:
            try:
                g = cp.load(f)
            except:
                break
            graphs.append(g)

    return graphs

def substructure_features(NXGraph):
    return nx.clustering(NXGraph)

def toDGL(NXgraph, Structuralinf=True):
    Graph = dgl.DGLGraph(NXgraph)
    if Structuralinf:
        cluster_attri =nx.clustering(NXgraph)
        degree_attri =dict(NXgraph.degree())
        orbits = nx.square_clustering(NXgraph)
        attri = {}
        for key, value in degree_attri.items():
            attri[key] = np.array([value+0.0,cluster_attri[key]+0.0, orbits[key]+0.0])

        nx.set_node_attributes(NXgraph, attri, "attr")
        Graph = dgl.from_networkx(NXgraph, node_attrs=["attr"])
    return Graph

def load_graph_list(fname,remove_self=True):

    if fname[-3:]=="pkl":
        glist = load_graphs(fname)
    else:
        with open(fname, "rb") as f:
            glist = np.load(f, allow_pickle=True)

    graph_list =[]
    for G in glist:
        try:
            if type(G)==list:
                if len(G[0])==0:
                    continue
                graph = nx.Graph()
                graph.add_nodes_from(G[0])
                graph.add_edges_from(G[1])
            elif type(G)==nx.classes.graph.Graph:
                graph = G
            else:
                graph = nx.from_numpy_matrix(G)
            if remove_self:
                graph.remove_edges_from(nx.selfloop_edges(graph))
            graph.remove_nodes_from(list(nx.isolates(graph)))
            Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
            graph = graph.subgraph(Gcc[0])
            graph = nx.Graph(graph)
            graph_list.append(graph)
        except:
            print("cpould not read a graph")
    return graph_list

def read_MMD(dir, line=-1):
    MMD_on_test = "Could not read"
    try:

        with open(dir) as f:
            lines = f.readlines()
        MMD_on_test =   lines[line]
    except Exception:
        pass
    return MMD_on_test


# bigg
dir = "/local-scratch/kiarash/AAAI/Arxived_BaseLines/bigg_result/data/"
Test_dir = "/test-graphs.pkl"
Train_dir= "/train-graphs.pkl"
Val_dir = "/val-graphs.pkl"


Structural_Feature_attri=True
import glob
import random
sub_dirs =glob.glob(dir + '*',recursive = True)
print(sub_dirs)
reprot = [["path", "f1_mean",  "f1_std" , "mmd_rbf_mean","mmd_rbf_std", "precision_mean","precision_std","recall_mean","recall_std","Statistics Report"]]
for path in sub_dirs:
    try:
        if  not os.path.isdir(path):
            continue
        Train = load_graph_list(path+Train_dir)
        Val = load_graph_list(path + Val_dir) # max 1000 graph
        Test = load_graph_list(path+Test_dir)

        data =Test+Val+Train

        random.shuffle(data)
        data = preprocess_graphs(data)


        device =  torch.device('cpu')
        # toDGL(generated[0])
        generated = data[:int(len(data)/2)]
        refrence = data[int(len(data)/2):]
        if Structural_Feature_attri:
            generated = [toDGL(g) for g in generated]
            refrence = [toDGL(g) for g in refrence]
        else:
            generated = [dgl.DGLGraph(g).to(device) for g in generated] # Convert graphs to DGL from NetworkX
            refrence = [dgl.DGLGraph(g).to(device) for g in refrence] # Convert graphs to DGL from NetworkX

        # Compute all GNN-based metrics at once
        from evaluation.evaluator import Evaluator

        f1 = []
        mmd_rbf = []
        recall =[]
        precision = []
        for i in range(10):
            evaluator = Evaluator(input_dim=3, device=device)
            result = evaluator.evaluate_all(generated, refrence)
            f1.append(result["f1_pr"])
            recall.append(result["recall"])
            precision.append(result["precision"])
            mmd_rbf.append(result["mmd_rbf"])


        f1_std= np.array(f1).std()
        f1_mean= np.array(f1).mean()

        mmd_rbf_std= np.array(mmd_rbf).std()
        mmd_rbf_mean= np.array(mmd_rbf).mean()

        recall_mean = np.array(recall).mean()
        recall_std = np.array(recall).std()
        precision_mean = np.array(precision).mean()
        precision_std = np.array(precision).std()
        res = [path, f1_mean,  f1_std , mmd_rbf_mean,mmd_rbf_std,precision_mean,precision_std,recall_mean,recall_std]
        print(res)
        reprot.append(res)
    except  Exception as e:
        reprot.append([path,"Error"])
        print(e)
import csv

if Structural_Feature_attri:
    dir = dir+"RandomGNN_"+"WithStructuralPorperites_Ideal"+"_"
else:
    dir ="RandomGNN_Ideal_"+dir

with open(dir+"report_of_dir_withPrecision.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(reprot)

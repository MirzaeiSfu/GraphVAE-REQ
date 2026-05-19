# Generate graphs for demonstration purposes
import random

import utils.graph_generators as gen
import pickle
import torch
import dgl
# import tensorflow as tf
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import scipy

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
        # orbits = nx.square_clustering(NXgraph)

        attri = {}
        for key, value in degree_attri.items():
            attri[key] = np.array([value+0.0,cluster_attri[key]+0.0,1.0])

        nx.set_node_attributes(NXgraph, attri, "attr")
        Graph = dgl.from_networkx(NXgraph, node_attrs=["attr"])
    return Graph

def load_attributedGraph_list(fname,remove_self=True):

    # def Hemogenizer(adj_matrix):
    #     return adj_matrix.sum(0)

    # with open(fname, "rb") as f:
    #         glist = np.load(f, allow_pickle=True)
    with open(fname, 'rb') as file:
        glist = pickle.load(file)

    graph_list =[]
    X_list = []
    for G,X in glist:
        try:
            # G = Hemogenizer(G)
            graph = nx.from_numpy_matrix(G)

            if remove_self:
                graph.remove_edges_from(nx.selfloop_edges(graph))
            graph.remove_nodes_from(list(nx.isolates(graph)))
            Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
            graph = graph.subgraph(Gcc[0])
            graph = nx.Graph(graph)
            graph_list.append(graph)
            X_list.append(X)
        except Exception as e:
            print("cpould not read a graph")
            print(e)
    return graph_list, X_list


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
            elif type(G)==scipy.sparse.csr.csr_matrix:
                graph = nx.from_scipy_sparse_matrix(G)
            else:
                graph = nx.from_numpy_matrix(G)
            if remove_self:
                graph.remove_edges_from(nx.selfloop_edges(graph))
            graph.remove_nodes_from(list(nx.isolates(graph)))
            Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
            graph = graph.subgraph(Gcc[0])
            graph = nx.Graph(graph)
            graph_list.append(graph)
        except Exception as e:
            print("cpould not read a graph")
            print(e)
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

def graphPloter(listOfNXgraphs, dir):
    for i, G in enumerate(listOfNXgraphs):
        plotG(G, file_name=dir+"_"+str(i))

def plotG(G, type="", graph_name = "Generated Graph", file_name=None):
    plt.close(graph_name)

    pos = nx.spring_layout(G, iterations=1000)
    f = plt.figure(graph_name)
    nx.draw(G, pos, node_size=20, width=1, edge_color="black")

    # plt.draw()

    # plt.show()
    # plt.pause(0.5)
    f.savefig(type+ "_graph.png" if file_name == None else file_name)
    # plt.show()
    #


dir = "/local-scratch/kiarash/LLGF_ruleLearner/sfu-graphlearning/GeneratedSamples/"

ref_file_name = "/refGraphs.npy"
gen_graphs_file_name = "/generatedGraphs.npy"

pattern = ""

Feature_Eval = True
Structural_Feature_attri=True
import glob
sub_dirs =glob.glob(dir + '*',recursive = True)
print(sub_dirs)
reprot = [["path", "mmd_rbf_mean","mmd_rbf_std", "mmd_linear_mean","mmd_linear_std"]]
for path in sub_dirs:
    if pattern in path:
        try:
            if  not os.path.isdir(path):
                continue
            generated, g_x = load_attributedGraph_list(path+gen_graphs_file_name)[:1000]
            refrence, r_x = load_attributedGraph_list(path + ref_file_name)[:1000] # max 1000 graph

            print(path)
            random.shuffle(generated)
            # generated = generated[:len(refrence)]
            generated = preprocess_graphs(generated)

            refrence = preprocess_graphs(refrence)

            device =  torch.device('cpu')
            # graphPloter(generated[:20], path + "/generated_sample")
            # toDGL(generated[0])
            if Structural_Feature_attri:
                generated = [toDGL(g) for g in generated]
                refrence = [toDGL(g) for g in refrence]
            else:
                generated = [dgl.DGLGraph(g).to(device) for g in generated] # Convert graphs to DGL from NetworkX
                refrence = [dgl.DGLGraph(g).to(device) for g in refrence] # Convert graphs to DGL from NetworkX

            # Compute all GNN-based metrics at once
            from evaluation.evaluator import Evaluator

            mmd_rbf = []
            mmd_linear = []
            for i in range(10):
                evaluator = Evaluator(input_dim=3, device=device)
                result = evaluator.evaluate_all(generated, refrence)

                mmd_linear.append(result["mmd_linear"])
                mmd_rbf.append(result["mmd_rbf"])




            mmd_rbf_std= np.array(mmd_rbf).std()
            mmd_rbf_mean= np.array(mmd_rbf).mean()
            mmd_linear_std= np.array(mmd_linear).std()
            mmd_linear_mean= np.array(mmd_linear).mean()
            res = [path,  mmd_rbf_mean,mmd_rbf_std,mmd_linear_mean,mmd_linear_std]
            print(res)
            reprot.append(res)
        except  Exception as e:
            reprot.append([path,"Error"])
            print(e)
import csv

if Structural_Feature_attri:
    dir = dir+"RandomGNN_"+"_Structural_Feature_"+"_"
else:
    dir ="RandomGNN_"+dir

with open(dir+pattern+"RuleEval.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(reprot)

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

dir = "/local-scratch/kiarash/AAAI/reachabilityWith3/dataset/"
ref_file_name = "/_Target_Graphs_adj_Test.npy"
gen_graphs_file_name = "/Max_Con_comp_generatedGraphs_adj_Test.npy"

dir = "/local-scratch/kiarash/AAAI/LDPVAE/MMD_statistic_GMMs_result_.1.1/"
ref_file_name = "/_test_set_.npy"
gen_graphs_file_name= "/_generated_set_.npy"







#GraphVAE-MM
dir = "/local-scratch/kiarash/AAAI/Arxived_BaseLines/GraphVAE-MM/"
ref_file_name = "/testGraphs_adj_.npy"
gen_graphs_file_name= "/Single_comp_generatedGraphs_adj_final_eval.npy"



#Bigg
dir = "/local-scratch/kiarash/AAAI/Arxived_BaseLines/bigg_result/data/"
ref_file_name = "/test.npy"
gen_graphs_file_name= "/generated.npy"


#GRAN
dir = "/local-scratch/kiarash/AAAI/Arxived_BaseLines/GGRAN/"
ref_file_name = "/__test_test_adj3000.npy"
gen_graphs_file_name= "/__gen_adj3000.npy"

# GraphRNN-RNN
dir = "/local-scratch/kiarash/AAAI/Arxived_BaseLines/Graphrnn/"
ref_file_name = "/_Target_Graphs_adj_Test.npy"
gen_graphs_file_name= "/generated.npy"



dir = "/local-scratch/kiarash/AAAI/Arxived_BaseLines/GraphVAE-MM/"
ref_file_name = "/testGraphs_adj_.npy"
gen_graphs_file_name= "/Single_comp_generatedGraphs_adj_final_eval.npy"

#Bigg
dir = "/local-scratch/kiarash/AAAI/Arxived_BaseLines/bigg_result/data/"
dir = "/local-scratch/kiarash/BiGG/google-research-0c1bbe5fc971a1de1a427debc814e66ab4f1e7fa/bigg/data/LDP/ogbg-molbbbp/"
ref_file_name = "/test-graphs.pkl"
gen_graphs_file_name= "/generated.npy"
pattern = ""

# # LDVAE
# dir = "/local-scratch/kiarash/AAAI/LDPVAE/dataset/"
# ref_file_name = "/_Target_Graphs_adj_Test.npy"
# gen_graphs_file_name = "/Max_Con_comp_generatedGraphs_adj_Test.npy"
# MMD_fileName = "/MMD.log"
# pattern = ""

# LDP-LDVAE
# dir = "/local-scratch/kiarash/AAAI/LDP_result_2/Remove_temp/"
# dir = "/local-scratch/kiarash/AAAI/LDP_result_2/dataset/"
# dir = "/local-scratch/kiarash/AAAI/LDPVAE/dataset/"
# dir = "/local-scratch/kiarash/AAAI/GenStat/dataset/"
# dir = "/local-scratch/kiarash/AAAI/LDPVAE/lr/"
# ref_file_name = "/_Target_Graphs_adj_Test.npy"
# gen_graphs_file_name = "/Max_Con_comp_generatedGraphs_adj_Test.npy"
# MMD_fileName = "/MMD.log"
# pattern = ""

# # LDP-LDVAE
# dir = "/local-scratch/kiarash/AAAI/_HiddenLAyers_beta4/"
# ref_file_name = "/_Target_Graphs_adj_Test.npy"
# gen_graphs_file_name = "/Max_Con_comp_generatedGraphs_adj_Test.npy"
# MMD_fileName = "/MMD.log"
# pattern = ""
dir = "/local-scratch/kiarash/AAAI/LDPExperiments2023Sep/DD_not_aggregated/"
dir = "/local-scratch/kiarash/AAAI/LDPExperiments2023Sep/_HiddenLAyers_2*2/"
ref_file_name = "/_Target_Graphs_adj_Test.npy"
gen_graphs_file_name = "/Max_Con_comp_generatedGraphs_adj_Test.npy"
pattern = "reachactleakydatasetDDlr00003beta4e20000stepsofreachability2argstepnum3NumDecLayers4decoderFCR"
pattern = "lobster"

MMD_fileName = "/MMD.log"
Structural_Feature_attri=True
import glob
sub_dirs =glob.glob(dir + '*',recursive = True)
print(sub_dirs)
reprot = [["path", "f1_mean",  "f1_std" , "mmd_rbf_mean","mmd_rbf_std", "precision_mean","precision_std","recall_mean","recall_std","Statistics Report"]]
for path in sub_dirs:
    if pattern in path:
        try:
            if  not os.path.isdir(path):
                continue
            generated = load_graph_list(path+gen_graphs_file_name)[:1000]
            refrence = load_graph_list(path + ref_file_name)[:1000] # max 1000 graph

            print(path)
            random.shuffle(generated)
            generated = generated[:len(refrence)]
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
            Stats = read_MMD(path+MMD_fileName)
            recall_mean = np.array(recall).mean()
            recall_std = np.array(recall).std()
            precision_mean = np.array(precision).mean()
            precision_std = np.array(precision).std()
            res = [path, f1_mean,  f1_std , mmd_rbf_mean,mmd_rbf_std,precision_mean,precision_std,recall_mean,recall_std, Stats]
            print(res)
            reprot.append(res)
        except  Exception as e:
            reprot.append([path,"Error"])
            print(e)
import csv

if Structural_Feature_attri:
    dir = dir+"RandomGNN_"+"Structural_Feature_attri_orbit_too"+"_"
else:
    dir ="RandomGNN_"+dir

with open(dir+pattern+"report_of_dir_withPrecision.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(reprot)

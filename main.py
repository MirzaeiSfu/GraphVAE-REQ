# | Action               | Shortcut            |
# | -------------------- | ------------------- |
# | Fold current block   | `Ctrl + Shift + [`  |
# | Unfold current block | `Ctrl + Shift + ]`  |
# | Fold all             | `Ctrl + K Ctrl + 0` |
# | Unfold all           | `Ctrl + K Ctrl + J` |


#====================================================================================
# region imports
import logging
import os
from pathlib import Path
import plotter
import torch.nn.functional as F
import argparse
try:
    import yaml
except ImportError:
    yaml = None
from model import *
from data import *
import pickle
import random as random
from GlobalProperties import *
from stat_rnn import mmd_eval
import time
import timeit
import dgl
from util import *
from motif_counting.motif_store import RuleBasedMotifStore
from motif_counting.motif_counter import RelationalMotifCounter
from motif_counting.motif_loss_utils import (
    compute_hard_motif_metrics,
    compute_motif_loss,
    get_motif_temperature,
    get_reconstructed_adj_probs,
    summarize_hard_motif_threshold_sweep,
    summarize_single_graph_motif_counts,
)
from motif_counting.sanity_check_compare import (
    compare_aggregated_counts_to_factorbase_detailed,
)
#endregion
#====================================================================================

# region seeding for reproducibility
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
#endregion

subgraphSize = None
keepThebest = False

# Choose which BFS reordering to use before training/counting.
# False -> legacy BFS from node 0 only
# True  -> BFS over all connected components (safe for disconnected graphs)
USE_ALL_COMPONENTS_BFS = True

#====================================================================================
#region arguments
def str2bool(value):
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(
        f"Expected a boolean value, received '{value}'."
    )


def _flatten_config_sections(config_data):
    flat_config = {}
    for key, value in config_data.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if nested_key in flat_config:
                    raise ValueError(
                        f"Duplicate config key '{nested_key}' found while flattening sections."
                    )
                flat_config[nested_key] = nested_value
        else:
            if key in flat_config:
                raise ValueError(f"Duplicate config key '{key}' found in config file.")
            flat_config[key] = value
    return flat_config


def load_config_defaults(config_path, valid_keys):
    if yaml is None:
        raise ImportError(
            "PyYAML is required for --config support. Install it with 'pip install PyYAML'."
        )

    resolved_path = Path(config_path).expanduser()
    with resolved_path.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}

    if not isinstance(config_data, dict):
        raise ValueError(
            f"Config file '{resolved_path}' must contain a YAML mapping at the top level."
        )

    flat_config = _flatten_config_sections(config_data)
    unknown_keys = sorted(set(flat_config) - set(valid_keys))
    if unknown_keys:
        raise ValueError(
            f"Unknown config keys in '{resolved_path}': {', '.join(unknown_keys)}"
        )

    return flat_config


parser = argparse.ArgumentParser(description='Kernel VGAE')

#===============================
# Config file
#===============================
parser.add_argument(
    '--config',
    type=str,
    default=None,
    help='Path to a single YAML config file.'
)

#===============================
# Data arguments
#===============================
parser.add_argument(
    '-dataset',
    dest="dataset",
    default="GRID",
    help="possible choices are: wheel_graph, PTC, FIRSTMM_DB, star, TRIANGULAR_GRID, multi_community, NCI1, ogbg-molbbbp, IMDbMulti, GRID, community, citeseer, LOBSTER, DD"
)
parser.add_argument(
    '-f',
    dest="use_feature",
    default=True,
    type=str2bool,
    help="either use features or identity matrix"
)
parser.add_argument(
    '-BFS',
    dest="bfsOrdering",
    default=True,
    type=str2bool,
    help="use bfs for graph permutations"
)
parser.add_argument(
    '-directed',
    dest="directed",
    default=True,
    type=str2bool,
    help="is the dataset directed?!"
)
parser.add_argument(
    '--database_name',
    type=str,
    default='grid_experiment'
)  # qm9_experiment, ogbg-molbbbp_experiment, PTC_experiment, MUTAG_experiment, PVGAErandomGraphs_experiment, FIRSTMM_DB_experiment, DD_experiment, GRID_experiment, PROTEINS_experiment, lobster_experiment, wheel_graph_experiment, TRIANGULAR_GRID_experiment, tree_experiment
parser.add_argument(
    '--graph_type',
    type=str,
    default='homogeneous',
    choices=['homogeneous', 'heterogeneous']
)
parser.add_argument(
    '--graph_index_start',
    type=int,
    default=None,
    help='First graph index to count (inclusive). Only valid when dataset has more than one graph.'
)
parser.add_argument(
    '--graph_index_end',
    type=int,
    default=None,
    help='Last graph index to count (inclusive). Only valid when dataset has more than one graph.'
)
parser.add_argument(
    '--data_dir',
    type=str,
    default=None,
    help='Optional dataset root. If set, main.py exports DATA_DIR for data.py; otherwise data.py uses DATA_DIR or local data/.'
)

#===============================
# Model arguments
#===============================
parser.add_argument(
    '-model',
    dest="model",
    default="GraphVAE-MM",
    help="KernelAugmentedWithTotalNumberOfTriangles and kipf are the main options in this repo; NOTE KernelAugmentedWithTotalNumberOfTriangles=GraphVAE-MM and kipf=GraphVAE"
)
parser.add_argument(
    '-encoder',
    dest="encoder_type",
    default="AvePool",
    help="the encoder: only option in this rep is 'AvePool'"
)  # only option in this rep is "AvePool"
parser.add_argument(
    '-decoder',
    dest="decoder",
    default="FC",
    help="the decoder type, FC is only option in this rep"
)
parser.add_argument(
    '-graphEmDim',
    dest="graphEmDim",
    default=1024,
    type=int,
    help="the dimention of graph Embeding LAyer; z"
)
parser.add_argument(
    '-beta',
    dest="beta",
    default=None,
    help="beta coefiicieny",
    type=float
)

#===============================
# Experiment arguments
#===============================
parser.add_argument(
    '-e',
    dest="epoch_number",
    default=10,
    type=int,
    help="Number of Epochs to train the model"
)
parser.add_argument(
    '-v',
    dest="Vis_step",
    default=2,
    type=int,
    help="at every Vis_step 'minibatch' the plots will be updated"
)
parser.add_argument(
    '-redraw',
    dest="redraw",
    default=False,
    type=str2bool,
    help="either update the log plot each step"
)
parser.add_argument(
    '-lr',
    dest="lr",
    default=0.0003,
    type=float,
    help="model learning rate"
)
parser.add_argument(
    '--train_batch_size',
    dest="train_batch_size",
    default=200,
    type=int,
    help="training mini-batch size"
)
parser.add_argument(
    '-task',
    dest="task",
    default="graphGeneration",
    help="only option in this rep is graphGeneration"
)

#===============================
# Motif arguments
#===============================
parser.add_argument('--motif_loss', type=str2bool, default=True)
# The default motif loss is now symmetric: zero-observed motifs are included
# through Laplace smoothing so extra motifs in the reconstruction are penalized
# too. This flag only chooses between absolute and squared log-ratio penalties.
parser.add_argument(
    '--motif_loss_mode',
    type=str,
    default='abs_log_ratio',
    choices=['abs_log_ratio', 'squared_log_ratio'],
    help='Motif loss variant: symmetric abs(log-ratio) or squared log-ratio.'
)
# Motif-temperature annealing only affects motif counting, not the main
# reconstruction loss. Keep start=end=1.0 to disable it, or use a schedule like
# start=1.0, end=0.5, start_frac=0.5 to keep training smooth early and sharpen
# the motif probabilities during the second half of training.
parser.add_argument(
    '--motif_temperature_start',
    type=float,
    default=1.0,
    help='Starting temperature for motif-count probabilities; lower than 1 sharpens logits.'
)
parser.add_argument(
    '--motif_temperature_end',
    type=float,
    default=0.5,
    help='Final temperature for motif-count probabilities after annealing.'
)
parser.add_argument(
    '--motif_temperature_anneal_start_frac',
    type=float,
    default=0.5,
    help='Fraction of training to keep the starting motif temperature before annealing.'
)
parser.add_argument('--rule_prune', type=str2bool, default=False)
parser.add_argument(
    '--motif_batch_size',
    type=int,
    dest="motif_batch_size",
    default=50000,
    help='motif-counting batch size. Only used for multi-graph datasets (QM9). Tune to your VRAM: 8 GB -> 2000 | 16 GB -> 5000 | 24 GB+ -> 30000.'
)

#===============================
# Loss arguments
#===============================
parser.add_argument(
    '--alpha_kernel_cost',
    type=float,
    default=0.0,
    help='Weight for kernel_cost in the total loss.'
)
parser.add_argument(
    '--alpha_node_feat',
    type=float,
    default=10.0,
    help='Weight for node feature reconstruction loss.'
)
parser.add_argument(
    '--alpha_edge_feat',
    type=float,
    default=10.0,
    help='Weight for edge feature reconstruction loss.'
)
parser.add_argument(
    '--alpha_motif_loss',
    type=float,
    default=1.0,
    help='Weight for motif loss.'
)
parser.add_argument(
    '--alpha_adj_recon',
    type=float,
    default=0.01,
    help='Weight for adjacency reconstruction loss.'
)

#===============================
# Runtime, output, and evaluation arguments
#===============================
parser.add_argument(
    '-graph_save_path',
    dest="graph_save_path",
    default=None,
    help="the direc to save generated synthatic graphs"
)
parser.add_argument(
    '-PATH',
    dest="PATH",
    default="model",
    help="a string which determine the path in wich model will be saved"
)
parser.add_argument(
    '-UseGPU',
    dest="UseGPU",
    default=True,
    type=str2bool,
    help="either use GPU or not if availabel"
)
parser.add_argument(
    '-device',
    '--device',
    dest="device",
    default="cuda",
    help="Which device should be used, e.g. cuda, cuda:0, cpu"
)
parser.add_argument(
    '-plot_testGraphs',
    dest="plot_testGraphs",
    default=True,
    type=str2bool,
    help="shall the test set be printed"
)
parser.add_argument(
    '-ideal_Evalaution',
    dest="ideal_Evalaution",
    default=False,
    type=str2bool,
    help="if you want to comapre the 50%50 subset of dataset comparision?!"
)
parser.add_argument(
    '--tiny_overfit',
    action='store_true',
    default=True,
    help='Use a tiny fixed training subset, disable shuffling, and train with one fixed batch.'
)
parser.add_argument(
    '--tiny_overfit_size',
    type=int,
    default=32,
    help='Number of training graphs to keep in --tiny_overfit mode.'
)
parser.add_argument('--interactive', action='store_true', default=False)
parser.add_argument(
    '--sanity_check',
    action='store_true',
    default=False,
    help='Run sanity check and print readable results.'
)
parser.add_argument(
    '--sanity_check_only',
    action='store_true',
    default=False,
    help='Run sanity check and exit before training.'
)


config_args, _ = parser.parse_known_args()
if config_args.config is not None:
    valid_config_keys = {action.dest for action in parser._actions}
    parser.set_defaults(**load_config_defaults(config_args.config, valid_config_keys))

args = parser.parse_args()

#===============================
# Data settings
#===============================
dataset = args.dataset  # possible choices are: cora, citeseer, karate, pubmed, DBIS
use_feature = args.use_feature
bfs_ordering = args.bfsOrdering
directed = args.directed
database_name = args.database_name
graph_type = args.graph_type
graph_index_start = args.graph_index_start
graph_index_end = args.graph_index_end
data_dir = args.data_dir

#===============================
# Model settings
#===============================
model_name = args.model
encoder_type = args.encoder_type
graphEmDim = args.graphEmDim
decoder_type = args.decoder
beta = args.beta

#===============================
# Experiment settings
#===============================
visulizer_step = args.Vis_step
redraw = args.redraw
task = args.task
epoch_number = args.epoch_number
lr = args.lr
train_batch_size = args.train_batch_size

#===============================
# Motif settings
#===============================
use_motif_loss = args.motif_loss
motif_loss_mode = args.motif_loss_mode
motif_temperature_start = max(float(args.motif_temperature_start), 1e-3)
motif_temperature_end = max(float(args.motif_temperature_end), 1e-3)
motif_temperature_anneal_start_frac = min(
    max(float(args.motif_temperature_anneal_start_frac), 0.0), 1.0
)
rule_prune = args.rule_prune
motif_batch_size = args.motif_batch_size

#===============================
# Loss settings
#===============================
alpha_kernel_cost = args.alpha_kernel_cost
alpha_node_feat = args.alpha_node_feat
alpha_edge_feat = args.alpha_edge_feat
alpha_motif_loss = args.alpha_motif_loss
alpha_adj_recon = args.alpha_adj_recon

#===============================
# Runtime, output, and evaluation settings
#===============================
device = args.device
use_gpu = args.UseGPU
graph_save_path = args.graph_save_path
PATH = args.PATH  # the dir to save the with the best performance on validation data
plot_testGraphs = args.plot_testGraphs
ideal_Evalaution = args.ideal_Evalaution
interactive = args.interactive
sanity_check = args.sanity_check
sanity_check_only = args.sanity_check_only
# endregion
#====================================================================================

if data_dir is not None:
    os.environ["DATA_DIR"] = str(Path(data_dir).expanduser())


#====================================================================================
# region Tiny overfit debug mode
tiny_overfit = args.tiny_overfit
tiny_overfit_size = args.tiny_overfit_size
if tiny_overfit:
    # Tiny overfit is a deterministic debug preset for checking whether the
    # current loss can be overfit on a tiny fixed subset. The model is later
    # switched to AutoEncoder=True below, so latent sampling is disabled here.
    tiny_overfit_size = 1
    epoch_number = min(int(epoch_number), 1000)
    visulizer_step = min(int(visulizer_step), 100)

    use_motif_loss = True
    args.motif_loss = True
    ideal_Evalaution = False
    args.ideal_Evalaution = False
    plot_testGraphs = False
    args.plot_testGraphs = False
    redraw = False
    task = 'debug'
    args.task = task
    args.tiny_overfit_size = tiny_overfit_size
    args.epoch_number = epoch_number
    args.Vis_step = visulizer_step
    print(f"[TinyOverfit] Auto preset: size={tiny_overfit_size}, epochs={epoch_number}, "
          f"vis_step={visulizer_step}, motif_loss={use_motif_loss}, task={task}")
    logging.info(f"[TinyOverfit] Auto preset: size={tiny_overfit_size}, epochs={epoch_number}, "
                 f"vis_step={visulizer_step}, motif_loss={use_motif_loss}, task={task}")
# end of tiny overfit debug mode
# endregion
#====================================================================================


if graph_save_path is None:
    run_name = "MMD_" + encoder_type + "_" + decoder_type + "_" + dataset + "_" + task + "_" + model_name + "BFS" + str(
        bfs_ordering) + str(epoch_number) + str(time.time())
    graph_save_dir = Path("runs") / run_name
else:
    graph_save_dir = Path(graph_save_path)
    run_name = graph_save_dir.name if graph_save_dir.name else "run_" + str(int(time.time()))

graph_save_dir.mkdir(parents=True, exist_ok=True)
graph_save_path = str(graph_save_dir) + "/"

runlog_dir = Path("runlog")
runlog_dir.mkdir(parents=True, exist_ok=True)
run_log_path = runlog_dir / f"{run_name}.log"
run_mmd_log_path = runlog_dir / f"{run_name}_MMD.log"

# maybe to the beest way
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=str(run_log_path), filemode='w', level=logging.INFO)

# **********************************************************************
# setting; general setting and hyper-parameters for each dataset
# region general settings
print("KernelVGAE SETING: " + str(args))
logging.info("KernelVGAE SETING: " + str(args))

kernl_type = []
#---------------------------------------------------------------------
if model_name == "KernelAugmentedWithTotalNumberOfTriangles" or model_name == "GraphVAE-MM":
    kernl_type = ["trans_matrix", "in_degree_dist", "out_degree_dist", "TotalNumberOfTriangles"]
    if dataset=="mnist":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 10, 50]
        step_num = 5
    if dataset=="zinc":
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 10, 50]
        step_num = 5
    if dataset == "large_grid":
        step_num = 5 # s in s-step transition
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 20, 100]
    elif dataset == "ogbg-molbbbp":
        # leision study
        alpha = [1, 1, 1, 1, 1, 0, 0, 0, 40, 1500]
        alpha = [0, 0, 0, 0, 0, 1, 1, 0, 40, 1500]
        alpha = [0, 0, 0, 0, 0, 0, 0, 1, 40, 1500]
        # -----------------------------------------
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 40, 1500]
        step_num = 5
    elif dataset == "IMDBBINARY":
        alpha = [ 1, 1, 1, 1, 1, 1, 2, 50]
        step_num = 5
    elif dataset == "QM9":
        step_num = 2
        alpha = [ 1, 1, 1, 1, 1, 20, 200]
    elif dataset == "PTC":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 2, 1]
    elif dataset =="MUTAG":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 4, 60]
    elif dataset =="PVGAErandomGraphs":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 4, 1]
    elif dataset == "FIRSTMM_DB":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 100]
    elif dataset == "DD":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 1000]
    elif dataset == "GRID":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 2000]
    elif dataset == "PROTEINS":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 2000]

    elif dataset == "LOBSTER":
        step_num = 5
        # leision study
        alpha = [1, 1, 1, 1, 1, 0, 0, 0, 40, 2000]  # degree
        alpha = [0, 0, 0, 0, 0, 1, 1, 0, 40, 2000]  # degree
        alpha = [1, 1, 1, 1, 1, 0, 0, 0, 40, 2000]
        # -------------------------------------------------
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 40, 2000]
    elif dataset == "wheel_graph":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 3000000, 20000 * 50000]
    elif dataset == "TRIANGULAR_GRID":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 2000]
    elif dataset == "tree":
        step_num = 5
        alpha = [1, 1, 1, 1, 1, 1, 1, 1, 50, 2000]
#---------------------------------------------------------------------

elif model_name == "kipf" or model_name == "graphVAE":
    alpha = [1, 1]
    step_num = 0

AutoEncoder = False

# Make sure if we are using tiny overfit debug mode, we are actually training an autoencoder (no kernel loss).
if tiny_overfit:
    AutoEncoder = True

if AutoEncoder == True:
    alpha[-1] = 0

if beta != None:
    alpha[-1] = beta

latent_mode = "AE" if AutoEncoder else "VAE"
print("latent_mode:" + latent_mode)
print("kernl_type:" + str(kernl_type))
print("alpha: " + str(alpha) + " num_step:" + str(step_num))
print(
    "loss_weights:"
    + f" kernel={alpha_kernel_cost},"
      f" node_feat={alpha_node_feat},"
      f" edge_feat={alpha_edge_feat},"
      f" motif={alpha_motif_loss},"
      f" adj_recon={alpha_adj_recon}"
)
print("motif_loss_mode:" + str(motif_loss_mode))
print(
    "motif_temperature_anneal:"
    + f"start={motif_temperature_start}, end={motif_temperature_end}, "
      f"start_frac={motif_temperature_anneal_start_frac}"
)

logging.info("latent_mode:" + latent_mode)
logging.info("kernl_type:" + str(kernl_type))
logging.info("alpha: " + str(alpha) + " num_step:" + str(step_num))
logging.info(
    "loss_weights:"
    + f" kernel={alpha_kernel_cost},"
      f" node_feat={alpha_node_feat},"
      f" edge_feat={alpha_edge_feat},"
      f" motif={alpha_motif_loss},"
      f" adj_recon={alpha_adj_recon}"
)
logging.info("motif_loss_mode:" + str(motif_loss_mode))
logging.info(
    "motif_temperature_anneal:"
    + f"start={motif_temperature_start}, end={motif_temperature_end}, "
      f"start_frac={motif_temperature_anneal_start_frac}"
)

  # with is propertion to revese of this value;

device = torch.device(device if torch.cuda.is_available() and use_gpu else "cpu")
print("the selected device is :", device)
logging.info("the selected device is :" + str(device))

# setting the plots legend
functions = ["Accuracy", "loss"]
if model_name == "kernel" or model_name == "KernelAugmentedWithTotalNumberOfTriangles" or model_name == "GraphVAE-MM":
    functions.extend(["Kernel" + str(i) for i in range(step_num)])
    functions.extend(kernl_type[1:])

if model_name == "TrianglesOfEachNode":
    functions.extend(kernl_type)

if model_name == "ThreeStepPath":
    functions.extend(kernl_type)

if model_name == "TotalNumberOfTriangles":
    functions.extend(kernl_type)

functions.append("Binary_Cross_Entropy")
functions.append("KL-D")
#endregion
# ========================================================================


pltr = plotter.Plotter(save_to_filepath="kernelVGAE_Log", functions=functions)

synthesis_graphs = {"wheel_graph", "star", "TRIANGULAR_GRID", "DD", "ogbg-molbbbp", "GRID", "small_lobster",
                    "small_grid", "community", "LOBSTER", "ego", "one_grid", "IMDBBINARY", ""}


# region Modules for latent space transformation and upsampling (not used in the current model, but can be useful for future extensions)
class NodeUpsampling(torch.nn.Module):
    def __init__(self, InNode_num, outNode_num, InLatent_dim=None):
        super(NodeUpsampling, self).__init__()
        self.Node_mlp = torch.nn.Linear(InNode_num * InLatent_dim, InLatent_dim * outNode_num)

    def forward(self, inTensor, activation=torch.nn.LeakyReLU(0.001)):
        Z = self.Node_mlp(inTensor.reshpe(inTensor.shape[0], -1))
        Z = torch.matmul(Z.reshpe(inTensor.shape[0], -1).permute(0, 2, 1), inTensor)

        return activation(Z)


class LatentMtrixTransformer(torch.nn.Module):
    def __init__(self, InNode_num, InLatent_dim=None, OutLatentDim=None):
        super(NodeUpsampling, self).__init__()
        self.Node_mlp = torch.nn.Linear(InNode_num * InLatent_dim, InNode_num * OutLatentDim)

    def forward(self, inTensor, activation=torch.nn.LeakyReLU(0.001)):
        Z = self.Node_mlp(inTensor.reshpe(inTensor.shape[0], -1))
        Z = torch.matmul(inTensor, Z.reshpe(inTensor.shape[-1], -1))

        return activation(Z)
#endregion

# ============================================================================

#region Testing and evaluation and helper functions
def test_(number_of_samples, model, graph_size, path_to_save_g, remove_self=True, save_graphs=True):
    import os
    if not os.path.exists(path_to_save_g):
        os.makedirs(path_to_save_g)
    # model.eval()
    generated_graph_list = []
    if not os.path.isdir(path_to_save_g):
        os.makedirs(path_to_save_g)
    k = 0
    for g_size in graph_size:
        for j in range(number_of_samples):
            z = torch.tensor(numpy.random.normal(size=[1, model.embeding_dim]))
            z = torch.randn_like(z)
            start_time = time.time()

            adj_logit = model.decode(z.to(device).float())
            print("--- %s seconds ---" % (time.time() - start_time))
            logging.info("--- %s seconds ---" % (time.time() - start_time))
            reconstructed_adj = torch.sigmoid(adj_logit)
            sample_graph = reconstructed_adj[0].cpu().detach().numpy()
            # sample_graph = sample_graph[:g_size,:g_size]
            sample_graph[sample_graph >= 0.5] = 1
            sample_graph[sample_graph < 0.5] = 0
            G = nx.from_numpy_array(sample_graph)
            # generated_graph_list.append(G)
            f_name = path_to_save_g + str(k) + str(g_size) + str(j) + dataset
            k += 1
            # plot and save the generated graph
            # plotter.plotG(G, "generated" + dataset, file_name=f_name)
            if remove_self:
                G.remove_edges_from(nx.selfloop_edges(G))

            G.remove_nodes_from(list(nx.isolates(G)))
            generated_graph_list.append(G)
            if save_graphs:
                plotter.plotG(G, "generated" + dataset, file_name=f_name + "_ConnectedComponnents")
    # ======================================================
    # save nx files
    if save_graphs:
        nx_f_name = path_to_save_g + "_" + dataset + "_" + decoder_type + "_" + model_name + "_" + task
        with open(nx_f_name, 'wb') as f:
            pickle.dump(generated_graph_list, f)
    # # ======================================================
    return generated_graph_list


def EvalTwoSet(model, test_list_adj, graph_save_path, Save_generated=True, _f_name=None, onlyTheBigestConCom = True):
    generated_graphs = test_(1, model, [x.shape[0] for x in test_list_adj], graph_save_path, save_graphs=Save_generated)
    graphs_to_writeOnDisk = [nx.to_numpy_array(G) for G in generated_graphs]
    if (onlyTheBigestConCom==False):
        if Save_generated:
            np.save(graph_save_path + 'generatedGraphs_adj_' + str(_f_name) + '.npy',
                    np.array(graphs_to_writeOnDisk, dtype=object),
                    allow_pickle=True)


            logging.info(mmd_eval(generated_graphs, [nx.from_numpy_array(graph.toarray()) for graph in test_list_adj]))
    print("====================================================")
    logging.info("====================================================")

    print("result for subgraph with maximum connected componnent")
    logging.info("result for subgraph with maximum connected componnent")
    generated_graphs = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in generated_graphs if
                        not nx.is_empty(G)]

    statistic_   = mmd_eval(generated_graphs, [nx.from_numpy_array(graph.toarray()) for graph in test_list_adj], diam=True)
    # if writeThem_in!=None:
    #     with open(writeThem_in+'MMD.log', 'w') as f:
    #         f.write(statistic_)
    logging.info(statistic_)
    if Save_generated:
        graphs_to_writeOnDisk = [nx.to_numpy_array(G) for G in generated_graphs]
        np.save(graph_save_path + 'Single_comp_generatedGraphs_adj_' + str(_f_name) + '.npy',
                np.array(graphs_to_writeOnDisk, dtype=object),
                allow_pickle=True)

        graphs_to_writeOnDisk = [G.toarray() for G in test_list_adj]
        np.save(graph_save_path + 'testGraphs_adj_.npy',
                np.array(graphs_to_writeOnDisk, dtype=object),
                allow_pickle=True)
    return  statistic_


def get_subGraph_features(org_adj, subgraphs_indexes, kernel_model):
    subgraphs = []
    target_kelrnel_val = None

    for i in range(len(org_adj)):
        subGraph = org_adj[i]
        if subgraphs_indexes != None:
            subGraph = subGraph[:, subgraphs_indexes[i]]
            subGraph = subGraph[subgraphs_indexes[i], :]
        # Converting sparse matrix to sparse tensor
        subGraph = torch.tensor(subGraph.todense())
        subgraphs.append(subGraph)
    subgraphs = torch.stack(subgraphs).to(device)

    if kernel_model != None:
        target_kelrnel_val = kernel_model(subgraphs)
        target_kelrnel_val = [val.to("cpu") for val in target_kelrnel_val]
    subgraphs = subgraphs.to("cpu")
    torch.cuda.empty_cache()
    return target_kelrnel_val, subgraphs


# the code is a hard copy of https://github.com/orybkin/sigma-vae-pytorch
def log_guss(mean, log_std, samples):
    return 0.5 * torch.pow((samples - mean) / log_std.exp(), 2) + log_std + 0.5 * np.log(2 * np.pi)


def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor
#endregion

#============================================================================
# region Loss functions for node and edge feature decoders
def compute_true_node_feat_loss(node_feat_logits, target_node_onehot, true_node_num):
    """
    Compute node feature BCE loss on real nodes only (ignore padded nodes).

    Expected shapes:
    - node_feat_logits:  (B, N_max, D)
    - target_node_onehot: (B, N_max, D)
    - true_node_num: length-B iterable/tensor with true node counts per graph

    Returns:
    - Scalar tensor: mean BCE over valid node-feature entries only.
    """
    # Validate tensor ranks early so a shape mismatch fails with a clear error.
    if node_feat_logits.ndim != 3 or target_node_onehot.ndim != 3:
        raise ValueError(
            "node_feat_logits and target_node_onehot must be 3D tensors with shape (B, N_max, D)."
        )

    # Validate that predicted and target tensors are aligned elementwise.
    if node_feat_logits.shape != target_node_onehot.shape:
        raise ValueError(
            f"Shape mismatch: logits {tuple(node_feat_logits.shape)} vs targets {tuple(target_node_onehot.shape)}."
        )

    # Read batch and padded-node dimensions.
    batch_size, max_nodes, node_feat_dim = node_feat_logits.shape

    # Convert node counts to a tensor on the same device as logits.
    # This supports Python lists, NumPy arrays, or torch tensors as input.
    true_node_num = torch.as_tensor(true_node_num, device=node_feat_logits.device)

    # Ensure we have exactly one node-count value per graph in the batch.
    if true_node_num.ndim != 1 or true_node_num.numel() != batch_size:
        raise ValueError(
            f"true_node_num must be 1D with length {batch_size}; got shape {tuple(true_node_num.shape)}."
        )

    # Node counts are indices for masking, so cast to long and clamp to valid bounds.
    # Clamping prevents accidental out-of-range values from breaking mask creation.
    true_node_num = true_node_num.long().clamp(min=0, max=max_nodes)

    # Build a boolean mask of shape (B, N_max):
    # True for real nodes [0, true_node_num[b]) and False for padded rows.
    node_positions = torch.arange(max_nodes, device=node_feat_logits.device).unsqueeze(0)
    valid_node_mask = node_positions < true_node_num.unsqueeze(1)

    # Compute elementwise BCE loss (no reduction) so we can apply the node mask manually.
    # Targets are cast to logits dtype/device to avoid mixed-type issues.
    per_entry_loss = F.binary_cross_entropy_with_logits(
        node_feat_logits,
        target_node_onehot.to(device=node_feat_logits.device, dtype=node_feat_logits.dtype),
        reduction='none'
    )

    # Expand node mask to feature dimension: (B, N_max) -> (B, N_max, 1).
    # Multiply to zero out padded-node contributions.
    valid_entry_mask = valid_node_mask.unsqueeze(-1).to(per_entry_loss.dtype)
    masked_loss_sum = (per_entry_loss * valid_entry_mask).sum()

    # Number of valid entries is (#valid nodes) * D.
    # Clamp denominator to avoid division-by-zero if a degenerate batch has zero valid nodes.
    valid_entry_count = (valid_entry_mask.sum() * node_feat_dim).clamp(min=1.0)

    # Return mean loss over valid node-feature entries only.
    return masked_loss_sum / valid_entry_count
def compute_true_edge_feat_loss(edge_feat_logits, target_edge_onehot, true_node_num):
    """
    Compute edge-feature loss using one-hot edge labels on real, existing edges only.

    Why this function exists:
    - Edge feature targets are one-hot encoded over C edge classes.
    - Padded nodes and non-edge pairs should not contribute to edge-feature loss.
    - We therefore treat edge feature prediction as multi-class classification
      (CrossEntropy), but only on positions where an edge actually exists.

    Expected shapes:
    - edge_feat_logits:   (B, C, N_max, N_max)
    - target_edge_onehot: (B, C, N_max, N_max) with one-hot labels on true edges
    - true_node_num:      length-B iterable/tensor with true node counts per graph

    Returns:
    - Scalar tensor: mean cross-entropy on valid existing edges only.
    """
    # Validate tensor ranks and alignment first for clearer runtime errors.
    if edge_feat_logits.ndim != 4 or target_edge_onehot.ndim != 4:
        raise ValueError(
            "edge_feat_logits and target_edge_onehot must be 4D tensors with shape (B, C, N_max, N_max)."
        )
    if edge_feat_logits.shape != target_edge_onehot.shape:
        raise ValueError(
            f"Shape mismatch: logits {tuple(edge_feat_logits.shape)} vs targets {tuple(target_edge_onehot.shape)}."
        )

    # Unpack dimensions.
    batch_size, _, max_nodes, _ = edge_feat_logits.shape

    # Convert and validate true node counts.
    true_node_num = torch.as_tensor(true_node_num, device=edge_feat_logits.device)
    if true_node_num.ndim != 1 or true_node_num.numel() != batch_size:
        raise ValueError(
            f"true_node_num must be 1D with length {batch_size}; got shape {tuple(true_node_num.shape)}."
        )
    true_node_num = true_node_num.long().clamp(min=0, max=max_nodes)

    # Build node-valid mask (B, N_max), then pair-valid mask (B, N_max, N_max).
    # This removes any contribution from padded node rows/cols.
    node_positions = torch.arange(max_nodes, device=edge_feat_logits.device).unsqueeze(0)
    valid_node_mask = node_positions < true_node_num.unsqueeze(1)
    valid_pair_mask = valid_node_mask.unsqueeze(1) & valid_node_mask.unsqueeze(2)

    # Use the one-hot target to detect where an edge actually exists.
    # For non-edges, target one-hot channels are all zeros.
    target_edge_onehot = target_edge_onehot.to(
        device=edge_feat_logits.device,
        dtype=edge_feat_logits.dtype
    )
    edge_exists_mask = target_edge_onehot.sum(dim=1) > 0

    # Final supervision mask: only real-node pairs that correspond to real edges.
    supervision_mask = valid_pair_mask & edge_exists_mask

    # Convert one-hot labels -> class indices for cross-entropy.
    # Cross-entropy is applied per (i, j) pair, then masked.
    target_edge_class = target_edge_onehot.argmax(dim=1).long()
    per_pair_loss = F.cross_entropy(edge_feat_logits, target_edge_class, reduction='none')

    # Average only over supervised edge positions.
    supervision_mask_f = supervision_mask.to(per_pair_loss.dtype)
    masked_loss_sum = (per_pair_loss * supervision_mask_f).sum()
    supervised_count = supervision_mask_f.sum().clamp(min=1.0)

    return masked_loss_sum / supervised_count
#endregion
#
def OptimizerVAE(reconstructed_adj, reconstructed_kernel_val, targert_adj, target_kernel_val, log_std, mean, alpha,
                 reconstructed_adj_logit, pos_wight, norm):
    loss = norm * torch.nn.functional.binary_cross_entropy_with_logits(reconstructed_adj_logit.float(),
                                                                       targert_adj.float(), pos_weight=pos_wight)

    norm = mean.shape[0] * mean.shape[1]
    kl = (1 / norm) * -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - torch.exp(log_std).pow(2))

    acc = (reconstructed_adj.round() == targert_adj).sum() / float(
        reconstructed_adj.shape[0] * reconstructed_adj.shape[1] * reconstructed_adj.shape[2])
    kernel_diff = 0
    each_kernel_loss = []
    log_sigma_values = []
    for i in range(len(target_kernel_val)):
        log_sigma = ((reconstructed_kernel_val[i] - target_kernel_val[i]) ** 2).mean().sqrt().log()
        log_sigma = softclip(log_sigma, -6)
        log_sigma_values.append(log_sigma.detach().cpu().item())
        step_loss = log_guss(target_kernel_val[i], log_sigma, reconstructed_kernel_val[i]).mean()
        each_kernel_loss.append(step_loss.cpu().detach().numpy() * alpha[i])
        kernel_diff += step_loss * alpha[i]

    kernel_diff += loss * alpha[-2]
    kernel_diff += kl * alpha[-1]
    each_kernel_loss.append((loss * alpha[-2]).item())
    each_kernel_loss.append((kl * alpha[-1]).item())
    return kl, loss, acc, kernel_diff, each_kernel_loss,log_sigma_values


def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])


# test_(5, "results/multiple graph/cora/model" , [x**2 for x in range(5,10)])





#====================================================================================
# load the data
#region Load the data

cache_dir  = "dataset_cached"
os.makedirs(cache_dir, exist_ok=True)        
cache_path = os.path.join(cache_dir, f"{dataset}.pkl")

self_for_none = True
if (decoder_type) in ("FCdecoder"): 
    self_for_none = True
    
use_cache = True  # Set to True to enable caching of processed datasets for faster subsequent loading.
if use_cache and os.path.exists(cache_path):
    print(f"[Cache] Loading '{dataset}' from {cache_path}")
    logging.info(f"[Cache] Loading '{dataset}' from {cache_path}")
    with open(cache_path, "rb") as _f:
        _cache = pickle.load(_f)

    list_adj          = _cache["list_adj"]
    list_x            = _cache["list_x"]
    list_label        = _cache["list_label"]
    list_node_feature = _cache["list_node_feature"]
    list_edge_feature = _cache["list_edge_feature"]
    node_feature_info = _cache["node_feature_info"]
    edge_feature_info = _cache["edge_feature_info"]
    list_node_onehot  = _cache["list_node_onehot"]
    list_edge_onehot  = _cache["list_edge_onehot"]
    node_onehot_info  = _cache["node_onehot_info"]
    edge_onehot_info  = _cache["edge_onehot_info"]

    test_list_adj     = _cache["test_list_adj"]
    val_adj           = _cache["val_adj"]
    list_graphs       = _cache["list_graphs"]
    list_test_graphs  = _cache["list_test_graphs"]

    if not _cache["single_graph"]:
        list_x_train     = _cache["list_x_train"]
        list_x_test      = _cache["list_x_test"]
        list_label_train = _cache["list_label_train"]
        list_label_test  = _cache["list_label_test"]
        list_noh_train   = _cache["list_noh_train"]
        list_noh_test    = _cache["list_noh_test"]
        list_eoh_train   = _cache["list_eoh_train"]
        list_eoh_test    = _cache["list_eoh_test"]

else:
    print(f"[Cache] No cache found for '{dataset}'. Running data pipeline ...")
    logging.info(f"[Cache] No cache found for '{dataset}'. Running data pipeline ...")

    (list_adj, list_x, list_label,
     list_node_feature, list_edge_feature,
     node_feature_info, edge_feature_info) = list_graph_loader(dataset, return_labels=True)

    # list_adj   = list_adj[:400]
    # list_x     = list_x[:400]
    # list_label = list_label[:400]

    bfs_reorder_fn = BFS_all_components if USE_ALL_COMPONENTS_BFS else BFS
    print("[BFS] Using {} ordering.".format(
        "all-components BFS" if USE_ALL_COMPONENTS_BFS else "legacy single-component BFS"
    ))

    list_adj, list_node_feature, list_edge_feature = bfs_reorder_fn(
        list_adj, list_node_feature, list_edge_feature
    )

    list_node_onehot, list_edge_onehot, node_onehot_info, edge_onehot_info = \
        build_onehot_features(list_node_feature, list_edge_feature, list_adj,
                              node_feature_info, edge_feature_info)

    # list_adj, list_x, list_label = list_graph_loader(dataset, return_labels=True, _max_list_size=80)
    # list_adj, _ = permute(list_adj, None)

    is_single_graph = len(list_adj) == 1

    if is_single_graph:
        test_list_adj = list_adj.copy()
        val_adj       = test_list_adj.copy()
        list_graphs   = Datasets(list_adj, self_for_none, list_x, None)
        list_test_graphs = Datasets(test_list_adj, self_for_none, list_x, None,
                                    Max_num=list_graphs.max_num_nodes,
                                    set_diag_of_isol_Zer=False)
    else:
        max_size = None
        # list_label = None

        (list_adj,         test_list_adj,
         list_x_train,     list_x_test,
         list_label_train, list_label_test,
         list_noh_train,   list_noh_test,
         list_eoh_train,   list_eoh_test) = data_split(
            graph_lis        = list_adj,
            list_x           = list_x,
            list_label       = list_label,
            list_node_onehot = list_node_onehot,
            list_edge_onehot = list_edge_onehot,
        )

        val_adj = list_adj[:int(len(test_list_adj))]
        list_graphs = Datasets(list_adj, self_for_none, list_x_train, list_label,
                               Max_num=max_size, set_diag_of_isol_Zer=False,
                               list_node_onehot=list_noh_train,
                               list_edge_onehot=list_eoh_train)
        list_test_graphs = Datasets(test_list_adj, self_for_none, list_x_test, list_label_test,
                                    Max_num=list_graphs.max_num_nodes, set_diag_of_isol_Zer=False,
                                    list_node_onehot=list_noh_test,
                                    list_edge_onehot=list_eoh_test)
        if plot_testGraphs:
            print("printing the test set...")
            # for i, G in enumerate(test_list_adj):
            #     G = nx.from_numpy_array(G.toarray())
            #     plotter.plotG(G, graph_save_path+"_test_graph" + str(i))

    _cache = {
        "list_adj":          list_adj,
        "list_x":            list_x,
        "list_label":        list_label,
        "list_node_feature": list_node_feature,
        "list_edge_feature": list_edge_feature,
        "node_feature_info": node_feature_info,
        "edge_feature_info": edge_feature_info,
        "list_node_onehot":  list_node_onehot,
        "list_edge_onehot":  list_edge_onehot,
        "node_onehot_info":  node_onehot_info,
        "edge_onehot_info":  edge_onehot_info,
        "single_graph":      is_single_graph,
        "test_list_adj":     test_list_adj,
        "val_adj":           val_adj,
        "list_graphs":       list_graphs,
        "list_test_graphs":  list_test_graphs,
        "self_for_none":     self_for_none,
    }

    if not is_single_graph:
        _cache.update({
            "list_x_train":     list_x_train,
            "list_x_test":      list_x_test,
            "list_label_train": list_label_train,
            "list_label_test":  list_label_test,
            "list_noh_train":   list_noh_train,
            "list_noh_test":    list_noh_test,
            "list_eoh_train":   list_eoh_train,
            "list_eoh_test":    list_eoh_test,
        })

    print(f"[Cache] Saving to {cache_path} ...")
    logging.info(f"[Cache] Saving to {cache_path} ...")
    with open(cache_path, "wb") as _f:
        pickle.dump(_cache, _f)
    print("[Cache] Saved successfully.")
    logging.info("[Cache] Saved successfully.")

#endregion
#====================================================================================


#====================================================================================
# Tiny-overfit mode: keep only a small fixed training subset.
# region Tiny-overfit mode
if tiny_overfit:
    keep_n = max(1, min(int(tiny_overfit_size), len(list_graphs.list_adjs)))
    list_graphs = Datasets(
        list_graphs.list_adjs[:keep_n],
        self_for_none,
        list_graphs.list_Xs[:keep_n] if list_graphs.list_Xs is not None else None,
        list_graphs.labels[:keep_n] if list_graphs.labels is not None else None,
        Max_num=list_graphs.max_num_nodes,
        set_diag_of_isol_Zer=list_graphs.set_diag_of_isol_Zer,
        list_node_onehot=(list_graphs.list_node_onehot[:keep_n]
                          if list_graphs.list_node_onehot is not None else None),
        list_edge_onehot=(list_graphs.list_edge_onehot[:keep_n]
                          if list_graphs.list_edge_onehot is not None else None),
    )
    train_batch_size = keep_n
    print(f"[TinyOverfit] Enabled: using {keep_n} fixed training graphs, "
          f"train_batch_size={train_batch_size}, shuffle=off")
    logging.info(f"[TinyOverfit] Enabled: using {keep_n} fixed training graphs, "
                 f"train_batch_size={train_batch_size}, shuffle=off")
#endregion
#====================================================================================


#====================================================================================
#region Motif Loss Setup: build motif store and precompute dataset motif counts
# This block prepares motif-count targets used by the motif-loss term.
if use_motif_loss:
    # Initializes the motif rule store (RuleBasedMotifStore).
    RuleBasedMotifStore(database_name=database_name, args=args) 

    # Builds the dataset to count on (train only, or train+test in sanity mode).
    if sanity_check or sanity_check_only:
        remove_self_loops(list_graphs)
        remove_self_loops(list_test_graphs)
        dataa = merge_datasets(list_graphs, list_test_graphs)  
    else :
        dataa = merge_datasets(list_graphs)

    # Creates a relational motif counter and wraps data for counting on CUDA.
    motif_counter = RelationalMotifCounter(database_name=database_name, args=args)
    wrapper = DataWrapper(dataa, motif_counter.relation_keys,node_onehot_info, device='cuda')

    # Computes motif counts in batches.
    counts  = motif_counter.count_batch(wrapper, batch_size=motif_batch_size)
    list_graphs.motif_counts = counts

    # In sanity mode, sums counts across all samples and prints them for inspection.
    if sanity_check or sanity_check_only:
        # Previous sanity-check output:
        # aggregated = counts.sum(0)
        # print(aggregated)
        aggregated = motif_counter.aggregate_motif_counts(counts)
        print("\n" + "=" * 80)
        print("SANITY CHECK: AGGREGATED MOTIF COUNTS")
        print("=" * 80)
        print(aggregated)

        if sanity_check or sanity_check_only:
            motif_counter.display_rules_and_motifs(aggregated)

            try:
                matches_factorbase, mismatches = (
                    compare_aggregated_counts_to_factorbase_detailed(
                        aggregated_counts=aggregated,
                        motif_counter=motif_counter,
                        database_name=database_name,
                    )
                )
                print("\n" + "=" * 80)
                print("FACTORBASE LOCAL_MULT COMPARISON")
                print("=" * 80)
                print(f"Counts match database local_mult values: {matches_factorbase}")
                if not matches_factorbase:
                    print("First mismatches:")
                    for mismatch in mismatches[:20]:
                        print(f"  {mismatch}")
                    if len(mismatches) > 20:
                        print(f"  ... and {len(mismatches) - 20} more mismatches")
            except Exception as exc:
                print("\n[SanityCheck] FactorBase comparison could not be completed:")
                print(f"  {exc}")

            if dataset == "PROTEINS":
                print("\nCompare these counts against FactorBase local_mult columns in:")
                print("  proteins_experiment_BN.`edges(nodes0,nodes1)_CP`")
                print("  proteins_experiment_BN.`node_feature(nodes0)_CP`")
                print("  proteins_experiment_BN.`node_feature(nodes1)_CP`")

        if sanity_check_only:
            print("\nSanity-check-only mode enabled; exiting before model training.")
            raise SystemExit(0)
#endregion
#====================================================================================




print("#------------------------------------------------------")
if ideal_Evalaution:
    fifty_fifty_dataset = list_adj + test_list_adj

    fifty_fifty_dataset = [nx.from_numpy_array(graph.toarray()) for graph in fifty_fifty_dataset]
    random.shuffle(fifty_fifty_dataset)
    print("50%50 Evalaution of dataset")
    logging.info(mmd_eval(fifty_fifty_dataset[:int(len(fifty_fifty_dataset)/2)],fifty_fifty_dataset[int(len(fifty_fifty_dataset)/2):],diam=True))

    graphs_to_writeOnDisk = [nx.to_numpy_array(G) for  G in fifty_fifty_dataset]
    np.save(graph_save_path+dataset+'_dataset.npy',
            np.array(graphs_to_writeOnDisk, dtype=object),
            allow_pickle=True)
print("#------------------------------------------------------")

SubGraphNodeNum = subgraphSize if subgraphSize != None else list_graphs.max_num_nodes
in_feature_dim = list_graphs.feature_size  # ToDo: consider none Synthasis data
nodeNum = list_graphs.max_num_nodes

degree_center = torch.tensor([[x] for x in range(0, SubGraphNodeNum, 1)])
degree_width = torch.tensor([[.1] for x in range(0, SubGraphNodeNum,1)])  # ToDo: both bin's center and widtg also maximum value of it should be determinde auomaticly
# ToDo: both bin's center and widtg also maximum value of it should be determinde auomaticly

bin_center = torch.tensor([[x] for x in range(0, SubGraphNodeNum, 1)])
bin_width = torch.tensor([[1] for x in range(0, SubGraphNodeNum, 1)])

kernel_model = kernel(device=device, kernel_type=kernl_type, step_num=step_num,
                      bin_width=bin_width, bin_center=bin_center, degree_bin_center=degree_center,
                      degree_bin_width=degree_width)

if encoder_type == "AvePool":
    encoder = AveEncoder(in_feature_dim, [256], graphEmDim)
else:
    print("requested encoder is not implemented")
    exit(1)

if decoder_type == "FC":
    decoder = GraphTransformerDecoder_FC(graphEmDim, 256, nodeNum, directed)
else:
    print("requested decoder is not implemented")
    exit(1)


if (subgraphSize == None):
    list_graphs.processALL(self_for_none=self_for_none)
    adj_list = list_graphs.get_adj_list()
    graphFeatures, _ = get_subGraph_features(adj_list, None, kernel_model)
    list_graphs.set_features(graphFeatures)


#====================================================================================
# %% Node and edge feature decoders
# region Node and edge feature decoders
# Added for feature decoding, I implemented it as a simple MLP that takes the graph 
# embedding as input and outputs the node and edge features, and then I added a loss 
# term to the total loss 

if not list_graphs.node_onehot_s or list_graphs.edge_onehot_s[0] is None:
    raise RuntimeError("Node or edge one-hot features are missing.")

node_onehot_dim = list_graphs.node_onehot_s[0].shape[-1]  
edge_onehot_dim = list_graphs.edge_onehot_s[0].shape[0] 

node_feat_decoder = NodeFeatureDecoder(graphEmDim, list_graphs.max_num_nodes, node_onehot_dim)
edge_feat_decoder = EdgeFeatureDecoder(graphEmDim, list_graphs.max_num_nodes, edge_onehot_dim)
#endregion
#====================================================================================

#====================================================================================
model = kernelGVAE(kernel_model, encoder, decoder, AutoEncoder, graphEmDim=graphEmDim,
                   node_feature_decoder=node_feat_decoder,
                   edge_feature_decoder=edge_feat_decoder)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr)

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000,6000,7000,8000,9000], gamma=0.5)
# A simple schedule helps the tiny motif-only run keep improving after the
# fast early drop. These milestones all occur within the 1000-epoch debug run.
# scheduler = torch.optim.lr_scheduler.MultiStepLR(
#    optimizer, milestones=[300, 600, 900], gamma=0.5
#)

# pos_wight = torch.true_divide((list_graphs.max_num_nodes**2*len(list_graphs.processed_adjs)-list_graphs.toatl_num_of_edges),
#                               list_graphs.toatl_num_of_edges) # addrressing imbalance data problem: ratio between positve to negative instance
# pos_wight = torch.tensor(40.0)
# pos_wight/=10
num_nodes = list_graphs.max_num_nodes
# ToDo Check the effect of norm and pos weight

# target_kelrnel_val = kernel_model(target_adj)

if not tiny_overfit:
    list_graphs.shuffle()
start = timeit.default_timer()
# Parameters
step = 0
swith = False
print(model)
logging.info(model.__str__())
min_loss = float('inf')


# 50%50 Evaluation

#region model loading
load_model = False
if load_model == True:  # I used this in line code to load a model #TODO: fix it
    # ========================================
    model_dir = "/local-scratch/kiarash/AAAI/Graph-Generative-Models/MMD_AvePool_FC_DD_graphGeneration_KernelAugmentedWithTotalNumberOfTrianglesBFSTrue100001651364417.4785793/"
    model.load_state_dict(torch.load(model_dir + "model_9999_3"))
    # EvalTwoSet(model, test_list_adj, model_dir+"/", Save_generated= False, )

# model_dir1 = "/local-scratch/kiarash/AAAI/Graph-Generative-Models/FinalResultHopefully/"
# model.load_state_dict(torch.load(model_dir1+"model_9999_3"))
# EvalTwoSet(model, test_list_adj, model_dir+"/", Save_generated= False, )
#endregion

#=========================================================================================
# %% Training loop
#region Training loop
for epoch in range(epoch_number):

    if not tiny_overfit:
        list_graphs.shuffle()
    batch = 0
    for iter in range(0, max(int(len(list_graphs.list_adjs) / train_batch_size), 1) * train_batch_size, train_batch_size):
        from_ = iter
        to_ = train_batch_size * (batch + 1)
        # for iter in range(0, len(list_graphs.list_adjs), train_batch_size):
        #     from_ = iter
        #     to_= train_batch_size*(batch+1) if train_batch_size*(batch+2)<len(list_graphs.list_adjs) else len(list_graphs.list_adjs)

        if subgraphSize == None:
            org_adj, x_s, node_num, subgraphs_indexes, target_kelrnel_val = list_graphs.get__(from_, to_, self_for_none,
                                                                                              bfs=subgraphSize)
        else:
            org_adj, x_s, node_num, subgraphs_indexes = list_graphs.get__(from_, to_, self_for_none, bfs=subgraphSize)

        # Keep an immutable copy of real node counts for masked node-feature loss.
        # `node_num` may be overwritten below for decoder-specific behavior.
        true_node_num = list(node_num)

        if (type(decoder)) in [GraphTransformerDecoder_FC]:  #
            node_num = len(node_num) * [list_graphs.max_num_nodes]

        x_s = torch.cat(x_s)
        x_s = x_s.reshape(-1, x_s.shape[-1])

        model.train()
        if subgraphSize == None:
            _, subgraphs = get_subGraph_features(org_adj, None, None)
        else:
            target_kelrnel_val, subgraphs = get_subGraph_features(org_adj, subgraphs_indexes, kernel_model)

        # target_kelrnel_val = kernel_model(org_adj, node_num)

        # batchSize = [org_adj.shape[0], org_adj.shape[1]]

        batchSize = [len(org_adj), org_adj[0].shape[0]]

        # org_adj_dgl = [dgl.from_scipy(sp.csr_matrix(graph.cpu().detach().numpy())) for graph in org_adj]
        [graph.setdiag(1) for graph in org_adj]
        org_adj_dgl = [dgl.from_scipy(graph) for graph in org_adj]
        org_adj_dgl = dgl.batch(org_adj_dgl).to(device)
        pos_wight = torch.true_divide(sum([x.shape[-1] ** 2 for x in subgraphs]) - subgraphs.sum(), subgraphs.sum())
        
        # added for feature decoding 
        # reconstructed_adj, prior_samples, post_mean, post_log_std, generated_kernel_val, reconstructed_adj_logit = model(
        (reconstructed_adj, prior_samples, post_mean, post_log_std, generated_kernel_val, reconstructed_adj_logit,
            node_feat_logits, edge_feat_logits) = model(
            org_adj_dgl.to(device), x_s.to(device), batchSize, subgraphs_indexes)
        kl_loss, reconstruction_loss, acc, kernel_cost, each_kernel_loss,log_sigma_values = OptimizerVAE(reconstructed_adj,
                                                                                        generated_kernel_val,
                                                                                        subgraphs.to(device),
                                                                                        [val.to(device) for val in
                                                                                         target_kelrnel_val],
                                                                                        post_log_std, post_mean, alpha,
                                                                                        reconstructed_adj_logit,
                                                                                        pos_wight, 2)

        # Added loss for feature decoding ============================================        
        #=============================================================================
        target_node_oh = torch.stack(
            [torch.tensor(list_graphs.node_onehot_s[i]) for i in range(from_, to_)]
        ).to(device)   
        target_edge_oh = torch.stack(
            [torch.tensor(list_graphs.edge_onehot_s[i]) for i in range(from_, to_)]
        ).to(device)   

        node_feat_loss = compute_true_node_feat_loss(
            node_feat_logits=node_feat_logits,
            target_node_onehot=target_node_oh,
            true_node_num=true_node_num
        )
        # Edge-feature supervision now treats channels as one-hot classes and
        # ignores padded/non-edge positions; only real existing edges contribute.
        edge_feat_loss = compute_true_edge_feat_loss(
            edge_feat_logits=edge_feat_logits,
            target_edge_onehot=target_edge_oh,
            true_node_num=true_node_num
        )
        # These hard metrics are evaluation-only diagnostics. They answer a
        # stricter question than the soft training loss: after discretizing the
        # current reconstruction, do the motif counts still match exactly?
        hard_motif_loss = torch.tensor(0.0, device=device)
        hard_motif_exact_zero = torch.tensor(False, device=device)
        hard_motif_exact_zero_per_graph = torch.zeros(
            len(true_node_num), dtype=torch.bool, device=device
        )
        hard_threshold_sweep_summary = None
        motif_temperature = get_motif_temperature(
            epoch=epoch,
            total_epochs=epoch_number,
            start_temp=motif_temperature_start,
            end_temp=motif_temperature_end,
            anneal_start_frac=motif_temperature_anneal_start_frac,
        )
        if use_motif_loss:
            observed_motif_counts = list_graphs.motif_counts[from_:to_].to(device)
            recon_wrapper = ReconstructedDataWrapper(
                reconstructed_adj=reconstructed_adj_logit,
                node_feat_logits=node_feat_logits,
                edge_feat_logits=edge_feat_logits,
                relation_keys=motif_counter.relation_keys,
                node_onehot_info=node_onehot_info,
                feature_onehot_mapping=wrapper.feature_onehot_mapping,
                use_soft_adj=True,
                prob_temperature=motif_temperature,
                device=device,
            )

            recon_counts = motif_counter.count_batch(recon_wrapper, batch_size=motif_batch_size)
            motif_loss = compute_motif_loss(
                observed_counts=observed_motif_counts,
                predicted_counts=recon_counts,
                loss_mode=motif_loss_mode,
            )

            # The hard wrapper thresholds adjacency and converts categorical
            # predictions to one-hot assignments, so these metrics reflect the
            # discrete graph you would inspect after training.
            with torch.no_grad():
                hard_recon_wrapper = ReconstructedDataWrapper(
                    reconstructed_adj=reconstructed_adj_logit.detach(),
                    node_feat_logits=node_feat_logits.detach(),
                    edge_feat_logits=edge_feat_logits.detach() if edge_feat_logits is not None else None,
                    relation_keys=motif_counter.relation_keys,
                    node_onehot_info=node_onehot_info,
                    feature_onehot_mapping=wrapper.feature_onehot_mapping,
                    use_soft_adj=False,
                    prob_temperature=motif_temperature,
                    device=device,
                )
                hard_recon_counts = motif_counter.count_batch(hard_recon_wrapper, batch_size=motif_batch_size)
                (hard_motif_loss,
                 hard_motif_exact_zero,
                 hard_motif_exact_zero_per_graph) = compute_hard_motif_metrics(
                    observed_counts=observed_motif_counts,
                    hard_predicted_counts=hard_recon_counts,
                )

                should_report_hard_sweep = (tiny_overfit and (step % 10 == 0)) or \
                    ((step + 1) % visulizer_step == 0) or (epoch_number == epoch + 1)
                if should_report_hard_sweep:
                    hard_threshold_sweep_summary = summarize_hard_motif_threshold_sweep(
                        observed_counts=observed_motif_counts,
                        adj_probs=get_reconstructed_adj_probs(
                            reconstructed_adj_logit,
                            prob_temperature=motif_temperature,
                        ),
                        hard_recon_wrapper=hard_recon_wrapper,
                        motif_counter=motif_counter,
                        batch_size=motif_batch_size,
                    )
            #m_loss = motif_loss * alpha_motif_loss


        else:
            #m_loss = torch.tensor(0.0, device=device)
            motif_loss=torch.tensor(0.0, device=device)
#====================-------=-==-=-=-===-*****%%%%%%%%%%%@@@@@@@@@@@@@@@@@@@@@

        loss = alpha_kernel_cost * kernel_cost + \
            alpha_node_feat * node_feat_loss +\
            alpha_edge_feat * edge_feat_loss+\
            motif_loss * alpha_motif_loss + \
            reconstruction_loss * alpha_adj_recon

        hard_exact_match_count = int(hard_motif_exact_zero_per_graph.sum().item())
        hard_exact_match_total = int(hard_motif_exact_zero_per_graph.numel())
        detailed_hard_motif_counts = None
        should_report_detailed_hard_counts = (
            tiny_overfit
            and hard_exact_match_total == 1
            and ((step + 1) % visulizer_step == 0 or (epoch_number == epoch + 1))
        )
        if should_report_detailed_hard_counts and use_motif_loss:
            detailed_hard_motif_counts = summarize_single_graph_motif_counts(
                observed_counts=observed_motif_counts,
                hard_predicted_counts=hard_recon_counts,
            )

        if tiny_overfit and (step % 10 == 0):
            print(f"[TinyOverfit] step={step} total={loss.item():.6f} "
                  f"motif={motif_loss.item():.6f} hard_motif={hard_motif_loss.item():.6f} "
                  f"motif_temp={motif_temperature:.3f} "
                  f"hard_exact_all={bool(hard_motif_exact_zero.item())} "
                  f"hard_exact_graphs={hard_exact_match_count}/{hard_exact_match_total} "
                  f"recon={reconstruction_loss.item():.6f} "
                  f"node={node_feat_loss.item():.6f} edge={edge_feat_loss.item():.6f}")
    #    
    #   loss = kernel_cost  # Graph generation loss without feature decoding
 
        tmp = [None for x in range(len(functions))]
        pltr.add_values(step, [acc.cpu().item(), loss.cpu().item(), *each_kernel_loss], tmp,
                        redraw=redraw)  # ["Accuracy", "loss", "AUC"])

        step += 1
        optimizer.zero_grad()
        loss.backward()

        if keepThebest and min_loss > loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), "model")
        # torch.nn.utils.clip_grad_norm(model.parameters(),  1.0044e-05)
        optimizer.step()

        if (step + 1) % visulizer_step == 0 or epoch_number==epoch+1:
            model.eval()
            if not tiny_overfit:
                pltr.redraw()
            if not tiny_overfit:
                dir_generated_in_train = "generated_graph_train/"
                if not os.path.isdir(dir_generated_in_train):
                    os.makedirs(dir_generated_in_train)
                rnd_indx = random.randint(0, len(node_num) - 1)
                sample_graph = reconstructed_adj[rnd_indx].cpu().detach().numpy()
                sample_graph = sample_graph[:node_num[rnd_indx], :node_num[rnd_indx]]
                sample_graph[sample_graph >= 0.5] = 1
                sample_graph[sample_graph < 0.5] = 0


                G = nx.from_numpy_array(sample_graph)
                plotter.plotG(G, "generated" + dataset,
                              file_name=graph_save_path + "generatedSample_At_epoch" + str(epoch))
                print("reconstructed graph vs Validation:")
                logging.info("reconstructed graph vs Validation:")
                reconstructed_adj = reconstructed_adj.cpu().detach().numpy()
                reconstructed_adj[reconstructed_adj >= 0.5] = 1
                reconstructed_adj[reconstructed_adj < 0.5] = 0
                reconstructed_adj = [nx.from_numpy_array(reconstructed_adj[i]) for i in range(reconstructed_adj.shape[0])]
                reconstructed_adj = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in
                                    reconstructed_adj if not nx.is_empty(G)]

                target_set = [nx.from_numpy_array(val_adj[i].toarray()) for i in range(len(val_adj))]
                target_set = [nx.Graph(G.subgraph(max(nx.connected_components(G), key=len))) for G in target_set if
                            not nx.is_empty(G)]
                reconstruc_MMD_loss = mmd_eval(reconstructed_adj, target_set[:len(reconstructed_adj)], diam=True)
                logging.info(reconstruc_MMD_loss)

            #todo: instead of printing diffrent level of logging shoud be used
            model.eval()
            if (not tiny_overfit) and task == "graphGeneration":
                # print("generated vs Validation:")
                mmd_res= EvalTwoSet(model, val_adj[:1000], graph_save_path, Save_generated=True, _f_name=epoch)
                with open(run_mmd_log_path, 'a') as f:
                        f.write(str(step)+" @ loss @ , "+str(loss.item())+" , @ Reconstruction @ , "+reconstruc_MMD_loss+" , @ Val @ , " +mmd_res+"\n")

                if ((step + 1) % visulizer_step * 2):
                    torch.save(model.state_dict(), graph_save_path + "model_" + str(epoch) + "_" + str(batch))
            stop = timeit.default_timer()
            # print("trainning time at this epoch:", str(stop - start))
            model.train()
            # if reconstruction_loss.item()<0.051276 and not swith:
            #     alpha[-1] *=2
            #     swith = True
        k_loss_str = ""
        for indx, l in enumerate(each_kernel_loss):
            k_loss_str += functions[indx + 2] + ":"
            k_loss_str += str(l) + ".   "

        epoch_status = (
            f"Epoch: {epoch + 1:03d} |Batch: {batch:03d} | latent_mode: {latent_mode} "
            f"| loss: {loss.item():05f} | motif_loss: {motif_loss.item():05f} "
            f"| motif_temp: {motif_temperature:.3f} "
            f"| hard_motif_loss: {hard_motif_loss.item():05f} "
            f"| hard_exact_all: {int(bool(hard_motif_exact_zero.item()))} "
            f"| hard_exact_graphs: {hard_exact_match_count}/{hard_exact_match_total} "
            f"| reconstruction_loss: {reconstruction_loss.item():05f} "
            f"| z_kl_loss: {kl_loss.item():05f} | accu: {(acc.item() if torch.is_tensor(acc) else float(acc)):03f}"
        )
        print(epoch_status, k_loss_str)
        logging.info(epoch_status + " " + str(k_loss_str))
        if hard_threshold_sweep_summary is not None:
            print(hard_threshold_sweep_summary)
            logging.info(hard_threshold_sweep_summary)
        if detailed_hard_motif_counts is not None:
            for detail_line in detailed_hard_motif_counts:
                print(detail_line)
                logging.info(detail_line)
        # print(log_sigma_values)
        log_std = ""
        for indx, l in enumerate(log_sigma_values):
            log_std += "log_std " + functions[indx + 2] + ":"
            log_std += str(l) + ".   "
        print(log_std)
        logging.info(log_std)
        batch += 1
        # scheduler.step()
        # scheduler.step()
model.eval()
if not tiny_overfit:
    torch.save(model.state_dict(), graph_save_path + "model_" + str(epoch) + "_" + str(batch))
#endregion
#=========================================================================================

stop = timeit.default_timer()
print("trainning time:", str(stop - start))
logging.info("trainning time: " + str(stop - start))
# save the train loss for comparing the convergence
import json

file_name = graph_save_path + "_" + encoder_type + "_" + decoder_type + "_" + dataset + "_" + task + "_" + model_name + "_elbo_loss.txt"

if not tiny_overfit:
    with open(file_name, "w") as fp:
        json.dump(list(np.array(pltr.values_train[-2]) + np.array(pltr.values_train[-1])), fp)

# with open(file_name + "/_CrossEntropyLoss.txt", "w") as fp:
#     json.dump(list(np.array(pltr.values_train[-2])), fp)
#
# with open(file_name + "/_train_loss.txt", "w") as fp:
#     json.dump(pltr.values_train[1], fp)

# save the log plot on the current directory
if not tiny_overfit:
    pltr.save_plot(graph_save_path + "KernelVGAE_log_plot")


#==========================================================================================
#   %% Evaluation of the model on graph generation task
# region graph generation task
if task == "graphGeneration":
    EvalTwoSet(model, test_list_adj, graph_save_path, Save_generated=True, _f_name="final_eval")
# endregion
#==========================================================================================

#==========================================================================================
# %% Evaluation of the model on graph generation task
# region graph Classification task
# if task == "graphClasssification":
#
#
#     org_adj,x_s, node_num, subgraphs_indexes,  labels = list_graphs.adj_s, list_graphs.x_s, list_graphs.num_nodes, list_graphs.subgraph_indexes, list_graphs.labels
#
#     if(type(decoder))in [  GraphTransformerDecoder_FC]: #
#         node_num = len(node_num)*[list_graphs.max_num_nodes]
#
#     x_s = torch.cat(x_s)
#     x_s = x_s.reshape(-1, x_s.shape[-1])
#
#     model.eval()
#     # if subgraphSize == None:
#     #     _, subgraphs = get_subGraph_features(org_adj, None, None)
#
#     batchSize = [len(org_adj), org_adj[0].shape[0]]
#
#     [graph.setdiag(1) for graph in org_adj]
#     org_adj_dgl = [dgl.from_scipy(graph) for graph in org_adj]
#
#     org_adj_dgl = dgl.batch(org_adj_dgl).to(device)
#     mean, std = model.encode(org_adj_dgl.to(device), x_s.to(device), batchSize)
#
#     prior_samples = model.reparameterize(mean, std)
#     # model.encode(org_adj_dgl.to(device), x_s.to(device), batchSize)
#     # _, prior_samples, _, _, _,_ = model(org_adj_dgl.to(device), x_s.to(device), node_num, batchSize, subgraphs_indexes)
#
#
#
#     import classification as CL
#
#     # NN Classifier
#     labels_test, labels_pred, accuracy, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, report  = CL.NN(prior_samples.cpu().detach(), labels)
#
#     print("Accuracy:{}".format(accuracy),
#           "Macro_AvgPrecision:{}".format(macro_precision), "Micro_AvgPrecision:{}".format(micro_precision),
#           "Macro_AvgRecall:{}".format(macro_recall), "Micro_AvgRecall:{}".format(micro_recall),
#           "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
#           "confusion matrix:{}".format(conf_matrix))
#
#     # KNN clasiifier
#     labels_test, labels_pred, accuracy, micro_recall, macro_recall, micro_precision, macro_precision, micro_f1, macro_f1, conf_matrix, report  = CL.knn(prior_samples.cpu().detach(), labels)
#     print("Accuracy:{}".format(accuracy),
#           "Macro_AvgPrecision:{}".format(macro_precision), "Micro_AvgPrecision:{}".format(micro_precision),
#           "Macro_AvgRecall:{}".format(macro_recall), "Micro_AvgRecall:{}".format(micro_recall),
#           "F1 - Macro,Micro: {} {}".format(macro_f1, micro_f1),
#           "confusion matrix:{}".format(conf_matrix))
# # evaluatin graph statistics in graph generation tasks
# endregion
#==========================================================================================

#==========================================================================================
# %% Evaluation of the model on graph representation learning task
# region graph representation learning task
# if task == "GraphRepresentation":
#
#     list_test_graphs.processALL(self_for_none=self_for_none)
#
#     test_adj_list = list_test_graphs.get_adj_list()
#     graphFeatures, _ = get_subGraph_features(test_adj_list, None, kernel_model)
#     list_test_graphs.set_features(graphFeatures)
#
#     from_ = 0
#     ro = [-1]
#     org_adj = list_test_graphs.adj_s[from_:to_]
#     x_s = list_test_graphs.x_s[from_:to_]
#     # test_adj_list.num_nodes[from_:to_]
#     labels = list_test_graphs.labels
#
#     x_s = torch.cat(x_s)
#     x_s = x_s.reshape(-1, x_s.shape[-1])
#
#     model.eval()
#     # if subgraphSize == None:
#     #     _, subgraphs = get_subGraph_features(org_adj, None, None)
#     # else:
#     #     target_kelrnel_val, subgraphs = get_subGraph_features(org_adj, subgraphs_indexes, kernel_model)
#
#     # target_kelrnel_val = kernel_model(org_adj, node_num)
#
#     # batchSize = [org_adj.shape[0], org_adj.shape[1]]
#
#     batchSize = [len(org_adj), org_adj[0].shape[0]]
#
#     # org_adj_dgl = [dgl.from_scipy(sp.csr_matrix(graph.cpu().detach().numpy())) for graph in org_adj]
#     [graph.setdiag(1) for graph in org_adj]
#     org_adj_dgl = [dgl.from_scipy(graph) for graph in org_adj]
#     org_adj_dgl = dgl.batch(org_adj_dgl).to(device)
#     pos_wight = torch.true_divide(sum([x.shape[-1] ** 2 for x in subgraphs]) - subgraphs.sum(), subgraphs.sum())
#
#     reconstructed_adj, prior_samples, post_mean, post_log_std, generated_kernel_val, reconstructed_adj_logit = model(
#         org_adj_dgl.to(device), x_s.to(device), batchSize, subgraphs_indexes)
#
#     i = 0
#     dic = {}
#     digit_labels = []
#     for labl in labels:
#         if labl not in dic:
#             dic[labl] = i
#             i += 1
#         digit_labels.append(dic[labl])
#
#     plotter.featureVisualizer(prior_samples.detach().cpu().numpy(), digit_labels)
#endregion
#==========================================================================================

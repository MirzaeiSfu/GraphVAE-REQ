"""Sample from a trained TU unconditional diffusion checkpoint.

Decoupled from training on purpose: `custom_train_tu` (lgd/train/tu_generation.py)
only trains and checkpoints. Run this script afterwards, against a saved
checkpoint, to generate graphs, compute structural stats, pickle the raw
NetworkX graphs (consumable by `reexport_tu_graphvae_req.py`), and — if
`ggm_eval` is importable in the current environment — export/evaluate them in
GraphVAE-REQ's format directly.
"""

import argparse
import logging
import pickle
from pathlib import Path

import torch
from torch_geometric.graphgym.checkpoint import load_ckpt
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric import seed_everything

import lgd.config  # noqa, register custom config
import lgd.encoder.type_dict_encoder  # noqa, register TU encoders
import lgd.loss.l1  # noqa, register l1/smoothl1 losses
import lgd.loader.master_loader  # noqa, register PyG-TUDataset loader
import lgd.model.GraphTransformerEncoder  # noqa, register graph transformer models
from lgd.ddpm.LGD import DDPM, LatentDiffusion
from lgd.ddpm.LGD_Inductive import LatentDiffusionInductive
from lgd.asset.stats import eval_graph_list
from lgd.train.tu_generation import _evaluate, _reference_graphs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest="cfg_file", required=True,
                        help="The training config file used to produce the checkpoint.")
    parser.add_argument("--run-id", required=True,
                        help="Run id (seed) sub-directory under out_dir/{cfg-name}/ to load the checkpoint from.")
    parser.add_argument("--epoch", type=int, default=-1,
                        help="Checkpoint epoch to load (-1 = latest available).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write the sampled-graph pickle and (if available) the "
                             "GraphVAE-REQ export. Defaults to '{run_dir}/sampled'.")
    parser.add_argument("--bins", type=int, default=8,
                        help="Quantile bins for continuous node attributes in the GraphVAE-REQ export.")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Config overrides, as space-separated 'key value' pairs.")
    return parser.parse_args()


def build_model():
    model_cls = eval(cfg.model.get('type', 'LatentDiffusion'))
    return model_cls(
        timesteps=cfg.diffusion.get('timesteps', 1000),
        conditioning_key=cfg.diffusion.conditioning_key,
        hid_dim=cfg.diffusion.hid_dim,
        parameterization=cfg.diffusion.get("parameterization", "x0"),
        cond_stage_key=cfg.diffusion.cond_stage_key,
        first_stage_config=cfg.diffusion.first_stage_config,
        cond_stage_config=cfg.diffusion.cond_stage_config,
        edge_factor=cfg.diffusion.get("edge_factor", 1.0),
        graph_factor=cfg.diffusion.get("graph_factor", 1.0),
        train_mode=cfg.diffusion.get("train_mode", 'sample'),
    ).to(torch.device(cfg.accelerator))


def main():
    args = parse_args()
    set_cfg(cfg)
    cfg.set_new_allowed(True)
    load_cfg(cfg, args)

    run_name = Path(args.cfg_file).stem
    run_name += f"-{cfg.name_tag}" if cfg.name_tag else ""
    cfg.out_dir = str(Path(cfg.out_dir) / run_name)
    cfg.run_dir = str(Path(cfg.out_dir) / args.run_id)
    set_printing()
    seed_everything(cfg.seed)
    auto_select_device()

    loaders = create_loader()
    model = build_model()
    loaded_epoch = load_ckpt(model, epoch=args.epoch)
    if loaded_epoch == 0:
        raise FileNotFoundError(
            f"No checkpoint found for epoch={args.epoch} under {cfg.run_dir}")
    logging.info("Loaded checkpoint from epoch %s", loaded_epoch - 1)

    reference = _reference_graphs(loaders[2])
    # The model is unconditional apart from graph size.  Draw generation
    # templates from the training loader so held-out test graph sizes are not
    # leaked into sampling, while still generating exactly |test| graphs.
    loss, generated = _evaluate(loaders[0], model, max_graphs=len(reference))
    results = eval_graph_list(
        reference, generated,
        methods=cfg.dataset.get("methods", None),
        kernels=cfg.dataset.get("kernels", None),
    )
    logging.info("loss=%s structural=%s", loss, results)

    output_dir = args.output_dir or (Path(cfg.run_dir) / "sampled")
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs_path = output_dir / f"epoch_{loaded_epoch - 1}_graphs.pkl"
    with graphs_path.open("wb") as stream:
        pickle.dump({"reference": reference, "generated": generated}, stream)
    logging.info("Saved sampled graphs to %s", graphs_path)

    try:
        from lgd.asset.graphvae_req_export import (
            evaluate_exported_tu_pyg_collections,
            export_tu_pyg_collections,
        )
        summary = export_tu_pyg_collections(
            train_loader=loaders[0], reference_loader=loaders[2],
            generated_graphs=generated,
            output_dir=output_dir / "graphvae_req_export",
            dataset_name=cfg.dataset.name,
            bins=args.bins,
        )
        evaluate_exported_tu_pyg_collections(
            summary, cfg.dataset.name, output_dir / "graphvae_req_eval",
            cfg.accelerator,
        )
    except RuntimeError as exc:
        logging.warning(
            "Skipping GraphVAE-REQ export/eval (%s). Use "
            "reexport_tu_graphvae_req.py on %s from an environment with "
            "ggm_eval installed instead.", exc, graphs_path)


if __name__ == "__main__":
    main()

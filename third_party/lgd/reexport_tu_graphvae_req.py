"""Re-export graphs saved by the TU generation loop for GraphVAE-REQ."""

import argparse
import pickle
from pathlib import Path

from lgd.asset.graphvae_req_export import export_tu_pyg_collections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("graphs", type=Path, help="Pickled NetworkX graph list")
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bins", type=int, default=8)
    args = parser.parse_args()
    from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
    from torch_geometric.graphgym.loader import create_loader
    import lgd.config  # noqa
    import lgd.encoder.type_dict_encoder  # noqa
    import lgd.model.GraphTransformerEncoder  # noqa

    set_cfg(cfg)
    cfg.set_new_allowed(True)
    # GraphGym's load_cfg expects the argparse-style cfg_file attribute.
    config_args = argparse.Namespace(cfg_file=str(args.train_config), opts=[])
    load_cfg(cfg, config_args)
    with args.graphs.open("rb") as stream:
        generated = pickle.load(stream)["generated"]
    loaders = create_loader()
    summary = export_tu_pyg_collections(
        train_loader=loaders[0], reference_loader=loaders[2],
        generated_graphs=generated, output_dir=args.output_dir,
        dataset_name=args.dataset, bins=args.bins,
    )
    print(summary)


if __name__ == "__main__":
    main()

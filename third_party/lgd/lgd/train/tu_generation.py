"""Unconditional diffusion training loop for attributed TU graphs.

Generation/export is intentionally *not* run inline during training (it used
to call into the optional `ggm_eval` package every eval epoch, which crashes
the whole training run if that package isn't installed in the training
environment, and ties expensive full reverse-diffusion sampling to the
training loop). Use `sample_tu.py` to sample from a saved checkpoint and
export/evaluate the generated graphs as a separate step. `_reference_graphs`
and `_evaluate` are kept here so that script can reuse them.
"""

import logging
import time

import networkx as nx
import torch
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_ckpt_epoch


@torch.no_grad()
def _reference_graphs(loader):
    graphs = []
    for batch in loader:
        batch = batch.to("cpu")
        offset = 0
        for graph_id in range(int(batch.num_graphs)):
            n = int(batch.num_node_per_graph[graph_id])
            edge_slice = slice(offset, offset + n ** 2)
            edge_values = batch.edge_attr[edge_slice]
            if edge_values.dim() > 1:
                edge_values = edge_values[:, 0]
            adjacency = edge_values.reshape(n, n)
            create_using = nx.Graph if cfg.diffusion.get('force_undirected', False) else nx.DiGraph
            graph = nx.from_numpy_array(adjacency.numpy(), create_using=create_using)
            graph.remove_edges_from(nx.selfloop_edges(graph))
            graphs.append(graph)
            offset += n ** 2
    return graphs


def _train_epoch(loader, model, optimizer, scheduler, accumulation):
    model.train()
    optimizer.zero_grad()
    for iteration, batch in enumerate(loader):
        batch.split = "train"
        batch.to(torch.device(cfg.accelerator))
        loss, loss_task, _, loss_node, loss_edge, _, _ = model.training_step(batch)
        if cfg.diffusion.cond_stage_key != "unconditional":
            loss = loss + loss_task * cfg.diffusion.get("task_factor", 0.0)
        loss.backward()
        if (iteration + 1) % accumulation == 0 or iteration + 1 == len(loader):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()


@torch.no_grad()
def _evaluate(loader, model, max_graphs=None):
    """Generate from template batches, optionally stopping at an exact count."""
    model.eval()
    generated = []
    total_loss = 0.0
    for batch in loader:
        batch.split = "test"
        batch.to(torch.device(cfg.accelerator))
        loss, graphs = model.inference(
            batch,
            ddim_steps=cfg.diffusion.get("ddim_steps", None),
            ddim_eta=cfg.diffusion.get("ddim_eta", 0.0),
            use_ddpm_steps=cfg.diffusion.get("use_ddpm_steps", False),
        )
        total_loss += float(loss.detach().cpu())
        generated.extend(graphs)
        if max_graphs is not None and len(generated) >= max_graphs:
            generated = generated[:max_graphs]
            break
    return total_loss, generated


@register_train("tu_unconditional")
def custom_train_tu(loggers, loaders, model, optimizer, scheduler):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume)
    for epoch in range(start_epoch, cfg.optim.max_epoch):
        start = time.perf_counter()
        _train_epoch(loaders[0], model, optimizer, scheduler,
                     cfg.optim.batch_accumulation)
        if cfg.optim.scheduler == "reduce_on_plateau":
            scheduler.step()
        else:
            scheduler.step()
        if cfg.train.enable_ckpt and is_ckpt_epoch(epoch):
            save_ckpt(model, optimizer, scheduler, epoch)
        logging.info("epoch %s completed in %.1fs", epoch, time.perf_counter() - start)
    for logger in loggers:
        logger.close()

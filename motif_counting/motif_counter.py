"""Evaluate cached relational motifs on graph batches.

CP-smoothed pruning workflow
----------------------------
Selecting ``motif_cp_table_source=cp_smoothed`` changes the runtime pipeline:

1. Load every combination from the cached ``_CP_smoothed`` tables.
2. Count each combination over all graphs in the active training split; the
   aggregate count is that combination's ``local_mult``.
3. Copy ``prior`` from the ordinary ``_CP`` table by matching the first
   functor's child value.
4. For each fixed assignment of the remaining parent functors, compute
   ``CP = local_mult / sum(local_mult across child values)``.
5. Apply the existing likelihood-based rule-pruning score to the derived
   metadata and use only the retained combinations for training targets.

This derivation stays in memory because local multiplicities and the resulting
pruning decision are specific to the current graph split.
"""

import os
import math
import torch
import pickle
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union

from motif_counting.motif_representations import (
    MOTIF_OUTPUT_MODES,
    canonicalize_motif_output_mode,
    compute_total_motif_count,
    pad_full_motif_matrix,
    represent_full_motif_matrices,
)
from motif_counting.motif_store import normalize_cp_table_source

MotifBatchResult = Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
]


def get_motif_cache_dir(args=None) -> Path:
    configured_dir = getattr(args, 'motif_cache_dir', None) if args is not None else None
    if configured_dir is not None:
        return Path(configured_dir).expanduser()
    return Path(os.environ.get("MOTIF_CACHE_DIR", "cache_motifs")).expanduser()


def use_syntactic_literal_rules(args=None) -> bool:
    return getattr(args, 'use_syntactic_literal_rules', True) if args is not None else True


def syntactic_literal_rule_mode(args=None) -> str:
    if not use_syntactic_literal_rules(args):
        return "original"

    mode = getattr(args, 'syntactic_literal_rule_mode', 'both') if args is not None else 'both'
    if mode not in {"original", "literals", "both"}:
        raise ValueError(f"Unknown syntactic_literal_rule_mode: {mode}")
    return mode


def get_motif_pickle_path(database_name: str, args=None) -> Path:
    # Cache files are flag-neutral: the pickle stores the complete rule/value
    # superset, and runtime flags filter that data after loading.
    return get_motif_cache_dir(args) / f"{database_name}.pkl"


class RelationalMotifCounter:
    """
    Counts motifs in a graph using relational algebra and Bayesian Network rules.
    Loads all required data from pickle file in the motif cache directory.

    STATELESS design
    ----------------
    self.matrices  → template dict loaded from the pickle (DB schema only).
                     NEVER written after __init__.

    Each call to count(graph_data) receives graph_data built by DataLoader
    and pre-processed by DataPreprocessor:
        graph_data['matrices']             → {relation_name: (N_max, N_max) tensor}
        graph_data['features']             → (N_max, F) node features  (pre-padded)
        graph_data['feat_onehot']          → (N_max, D) one-hot features (pre-padded)
        graph_data['feature_onehot_mapping'] → {col_idx: {val_int: oh_col_idx}}
        graph_data['labels']               → edge-feature tensors | None (pre-padded)
        graph_data['N_max']                → int — global N_max for the dataset

    GRADIENT-SAFE feature predicates
    ---------------------------------
    The old code computed feature masks with:
        (feat_b[:, :, indx] == val).float()          ← boolean, no gradient
    This is replaced throughout by direct indexing into the pre-built one-hot
    matrix:
        feat_onehot_b[:, :, mapping[indx][val]]      ← pure slice, gradient ✓
    The boolean comparison is performed ONCE during DataPreprocessor.preprocess()
    — outside any gradient-tracked computation.

    BATCHED design (count_batch)
    ----------------------------
    Since DataPreprocessor already pads ALL graphs to the global N_max, every
    graph_data tensor has identical shape.  _build_batch_tensors() now only
    needs to torch.stack() — no per-batch shape checks or zero-padding.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, database_name: str, args):
        self.database_name = database_name
        self.args = args

        pickle_path = get_motif_pickle_path(database_name, args)

        if not pickle_path.exists():
            raise FileNotFoundError(
                f"Pickle file not found: {pickle_path}\n"
                f"Please ensure motif store has been initialised first."
            )

        print(f"  Loading motif data from: {pickle_path}")
        self._load_from_pickle(pickle_path)
        print(f"  Loaded {self.num_motifs} motif rules")

    def _load_from_pickle(self, pickle_path: Path):
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        self.entities              = data["entities"]
        self.relations             = data["relations"]
        self.keys                  = data["keys"]
        self.rules                 = data["rules"]
        self.indices               = data["indices"]
        self.attributes            = data["attributes"]
        self.base_indices          = data["base_indices"]
        self.mask_indices          = data["mask_indices"]
        self.sort_indices          = data["sort_indices"]
        self.stack_indices         = data["stack_indices"]
        self.functors              = data["functors"]
        self.variables             = data["variables"]
        self.nodes                 = data["nodes"]
        self.states                = data["states"]
        self.masks                 = data["masks"]
        self.multiples             = data["multiples"]
        self.entity_feature_columns   = data.get("entity_feature_columns", {})
        self.entity_literal_values    = data.get("entity_literal_values", {})
        self.relation_feature_columns = data.get("relation_feature_columns", {})
        self.relation_entity_tables   = data.get("relation_entity_tables", {})
        self.relation_literal_values  = data.get("relation_literal_values", {})
        self.relation_occurrence_counts = data.get("relation_occurrence_counts", {})
        self.rule_sources          = data.get("rule_sources")
        if self.rule_sources is None:
            self.rule_sources = self._infer_rule_sources_for_legacy_pickle(data)
        if len(self.rule_sources) != len(self.rules):
            raise RuntimeError(
                "Motif cache rule_sources length does not match rules length. "
                "Delete the motif pickle and regenerate the cache."
            )
        self.feature_info_mapping  = data.get("feature_info_mapping", {})
        self.num_nodes_graph       = data.get("num_nodes_graph", 0)
        self.motif_cp_table_source = normalize_cp_table_source(self.args)
        self.cp_reference_values = [
            list(rows) for rows in data.get("values_full", [])
        ]
        self.cp_reference_columns = [
            list(columns) for columns in data.get(
                "value_columns",
                [[] for _ in self.rules],
            )
        ]
        self.syntactic_literal_rule_mode = syntactic_literal_rule_mode(self.args)
        self.use_syntactic_literal_rules = self.syntactic_literal_rule_mode != "original"
        loaded_total_relation_occurrences = data.get("total_relation_occurrences", {})
        if isinstance(loaded_total_relation_occurrences, dict):
            self.total_relation_occurrences = loaded_total_relation_occurrences
        else:
            self.total_relation_occurrences = dict(self.relation_occurrence_counts)

        # ── Select CP vs CP_smoothed value combinations ───────────────
        self.data_driven_smoothed_pruning_pending = False
        if self.motif_cp_table_source == "cp_smoothed":
            smoothed_values = data.get("values_smoothed_full")
            smoothed_columns = data.get("value_smoothed_columns")
            if (
                smoothed_values is None
                or smoothed_columns is None
                or len(smoothed_values) != len(self.rules)
                or any(
                    rows is None
                    for rows, source in zip(smoothed_values, self.rule_sources)
                    if source == "factorbase"
                )
            ):
                raise RuntimeError(
                    "The motif cache does not contain `_CP_smoothed` rows for "
                    f"every FactorBase rule: {pickle_path}. Delete/regenerate "
                    "the cache while the smoothed BN tables are available."
                )
            self.values = [list(rows) for rows in smoothed_values]
            self.value_columns = [
                list(columns) if columns is not None else []
                for columns in smoothed_columns
            ]
            self.data_driven_smoothed_pruning_pending = bool(
                getattr(self.args, "rule_prune", False)
            )
            if self.data_driven_smoothed_pruning_pending:
                print(
                    "  motif_cp_table_source=cp_smoothed: loaded all "
                    f"{sum(len(rows) for rows in self.values)} combinations; "
                    "local_mult/CP/prior calculation and pruning are deferred "
                    "until graph data is available"
                )
            else:
                print(
                    "  motif_cp_table_source=cp_smoothed: using all "
                    f"{sum(len(rows) for rows in self.values)} combinations"
                )
        else:
            self.value_columns = [
                list(columns) for columns in data.get(
                    "value_columns",
                    [[] for _ in self.rules],
                )
            ]
            self.values = self._select_motif_values(data, pickle_path)

        self._filter_rules_for_runtime_mode()
        self._build_motif_group_masks()

        self.device = getattr(self.args, 'device', 'cuda')

        # Template matrices — kept ONLY to expose relation key names to DataLoader.
        # Never mutated after this point.
        self.matrices: Dict[str, torch.Tensor] = {}
        for key, matrix in data["matrices"].items():
            self.matrices[key] = (
                matrix.to(self.device) if isinstance(matrix, torch.Tensor) else matrix
            )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def num_motifs(self) -> int:
        return len(self.rules)

    @property
    def relation_keys(self) -> List[str]:
        """
        Relation names the DataLoader must use as keys inside graph_data['matrices'].
        Pass directly to data_loader.get_graph_data_list(relation_keys=...).

        Example:
            graph_data_list = data_loader.get_graph_data_list(
                relation_keys=motif_counter.relation_keys
            )
        """
        return list(self.matrices.keys())

    def get_syntactic_literal_motif_mask(self, device=None) -> torch.Tensor:
        mask = self.syntactic_literal_motif_mask
        if device is not None:
            return mask.to(device)
        return mask

    def get_unit_relation_motif_mask(self, device=None) -> torch.Tensor:
        mask = self.unit_relation_motif_mask
        if device is not None:
            return mask.to(device)
        return mask

    def select_rule_values_from_motif_mask(
        self,
        motif_mask: torch.Tensor,
    ) -> Dict[int, List[int]]:
        """Map a flat motif-value mask back to ordered rule/value indices."""
        motif_mask = torch.as_tensor(motif_mask, dtype=torch.bool, device="cpu")
        expected_motifs = sum(len(value_rows) for value_rows in self.values)
        if motif_mask.ndim != 1 or motif_mask.numel() != expected_motifs:
            raise ValueError(
                "Motif selection mask must match the flattened rule/value "
                f"dimension ({expected_motifs},), got {tuple(motif_mask.shape)}."
            )

        selection = {}
        offset = 0
        for rule_idx, value_rows in enumerate(self.values):
            value_count = len(value_rows)
            selected_value_indices = torch.nonzero(
                motif_mask[offset:offset + value_count],
                as_tuple=False,
            ).flatten().tolist()
            if selected_value_indices:
                selection[rule_idx] = selected_value_indices
            offset += value_count
        return selection

    def do_interactive_selection(self) -> Dict:
        """Interactive rule/value selection for multi-graph runs (ask only once)."""
        print("\n" + "="*80)
        print("INTERACTIVE RULE SELECTION")
        print("="*80)
        print("(This selection will be applied to all graphs)")
        print("="*80 + "\n")
        selected = self._interactive_rule_selection()
        print("\n" + "="*80)
        print("Selection complete — will be applied to all graphs.")
        print("="*80)
        return selected

    @property
    def requires_data_driven_smoothed_pruning(self) -> bool:
        """Whether all smoothed rows must be counted before rule pruning."""
        return bool(self.data_driven_smoothed_pruning_pending)

    @staticmethod
    def _consistent_prior_by_child(
        rule: List[str],
        cp_rows: List,
        cp_columns: List[str],
    ) -> Dict[Any, float]:
        """Read one FactorBase prior for each value of the rule child atom."""
        child_column = rule[0]
        if child_column not in cp_columns or "prior" not in cp_columns:
            raise ValueError(
                f"Original CP metadata for rule {rule!r} must contain "
                f"{child_column!r} and 'prior'; columns={cp_columns!r}."
            )
        child_idx = cp_columns.index(child_column)
        prior_idx = cp_columns.index("prior")
        result: Dict[Any, float] = {}
        for row in cp_rows:
            child_value = row[child_idx]
            prior_value = row[prior_idx]
            if prior_value is None or prior_value == "":
                continue
            prior_value = float(prior_value)
            previous = result.get(child_value)
            if previous is not None and not math.isclose(
                previous,
                prior_value,
                rel_tol=1e-7,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"Inconsistent priors for {child_column}={child_value!r}: "
                    f"{previous} vs {prior_value}."
                )
            result[child_value] = prior_value
        return result

    @classmethod
    def derive_smoothed_rule_rows(
        cls,
        rule: List[str],
        smoothed_rows: List,
        smoothed_columns: List[str],
        cp_rows: List,
        cp_columns: List[str],
        local_mults: torch.Tensor,
    ) -> List[List[Any]]:
        """Populate local_mult, CP, ParentSum, and prior for smoothed rows.

        FactorBase stores the child atom first in ``rule``. Conditional
        probabilities therefore group rows by every remaining (parent) atom
        and normalize over the different child values in that parent group.
        Priors are copied from the ordinary ``_CP`` table by child value.
        """
        required_columns = list(rule) + ["local_mult", "CP", "prior"]
        missing = [name for name in required_columns if name not in smoothed_columns]
        if missing:
            raise ValueError(
                f"Smoothed CP metadata for rule {rule!r} is missing columns: "
                f"{missing}; columns={smoothed_columns!r}."
            )
        local_mults = torch.as_tensor(local_mults, dtype=torch.float64).flatten()
        if local_mults.numel() != len(smoothed_rows):
            raise ValueError(
                f"Rule {rule!r} has {len(smoothed_rows)} smoothed rows but "
                f"{local_mults.numel()} local_mult values."
            )

        child_idx = smoothed_columns.index(rule[0])
        parent_indices = [smoothed_columns.index(atom) for atom in rule[1:]]
        local_mult_idx = smoothed_columns.index("local_mult")
        cp_idx = smoothed_columns.index("CP")
        prior_idx = smoothed_columns.index("prior")
        parent_sum_idx = (
            smoothed_columns.index("ParentSum")
            if "ParentSum" in smoothed_columns
            else None
        )

        parent_sums: Dict[Tuple[Any, ...], float] = {}
        local_values = [float(value) for value in local_mults.tolist()]
        for row, local_mult in zip(smoothed_rows, local_values):
            parent_key = tuple(row[index] for index in parent_indices)
            parent_sums[parent_key] = parent_sums.get(parent_key, 0.0) + local_mult

        prior_by_child = cls._consistent_prior_by_child(
            rule,
            cp_rows,
            cp_columns,
        )
        derived_rows: List[List[Any]] = []
        for row, local_mult in zip(smoothed_rows, local_values):
            mutable_row = list(row)
            parent_key = tuple(row[index] for index in parent_indices)
            denominator = parent_sums[parent_key]
            cp_value = local_mult / denominator if denominator > 0.0 else 0.0
            child_value = row[child_idx]
            if child_value not in prior_by_child:
                raise ValueError(
                    f"No prior found in the ordinary _CP rows for "
                    f"{rule[0]}={child_value!r}."
                )
            mutable_row[local_mult_idx] = local_mult
            mutable_row[cp_idx] = cp_value
            mutable_row[prior_idx] = prior_by_child[child_value]
            if parent_sum_idx is not None:
                mutable_row[parent_sum_idx] = denominator
            derived_rows.append(mutable_row)
        return derived_rows

    def _prune_derived_smoothed_rows(
        self,
        rule_idx: int,
        rows: List[List[Any]],
    ) -> List[List[Any]]:
        """Apply the existing FactorBase score to derived smoothed metadata."""
        rule = self.rules[rule_idx]
        if len(rule) == 1 or self.rule_sources[rule_idx] != "factorbase":
            return list(rows)

        columns = self.value_columns[rule_idx]
        required = ("local_mult", "CP", "prior")
        missing = [name for name in required if name not in columns]
        if missing:
            raise ValueError(
                f"Cannot prune smoothed rule {rule!r}; missing columns {missing}."
            )
        local_idx = columns.index("local_mult")
        cp_idx = columns.index("CP")
        prior_idx = columns.index("prior")
        scored_rows = []
        for row in rows:
            local_mult = float(row[local_idx])
            cp_value = float(row[cp_idx])
            prior_value = float(row[prior_idx])
            if local_mult <= 0.0 or cp_value <= 0.0 or prior_value <= 0.0:
                continue
            score = (
                2.0 * local_mult * (math.log(cp_value) - math.log(prior_value))
                - math.log(local_mult)
            )
            if score > 0.0:
                scored_rows.append((score, row))

        max_values = getattr(self.args, "motif_prune_max_values_per_rule", None)
        if max_values is not None and max_values > 0:
            scored_rows = sorted(
                scored_rows,
                key=lambda item: item[0],
                reverse=True,
            )[:max_values]
        return [row for _, row in scored_rows]

    def prepare_data_driven_smoothed_pruning(
        self,
        preprocessor,
        batch_size: int,
    ) -> Dict[str, int]:
        """Count all smoothed combinations, derive metadata, then prune them."""
        if not self.requires_data_driven_smoothed_pruning:
            return {
                "full_combinations": sum(len(rows) for rows in self.values),
                "pruned_combinations": sum(len(rows) for rows in self.values),
            }

        counts = self.count_batch(
            preprocessor,
            batch_size=batch_size,
            output_mode="total_count",
            detach_to_cpu=True,
        )
        aggregated_counts = counts.sum(dim=0)
        expected_count = sum(len(rows) for rows in self.values)
        if aggregated_counts.numel() != expected_count:
            raise RuntimeError(
                "Smoothed local_mult counting returned the wrong motif "
                f"dimension: expected {expected_count}, got "
                f"{aggregated_counts.numel()}."
            )

        derived_values = []
        offset = 0
        for rule_idx, rows in enumerate(self.values):
            value_count = len(rows)
            rule_counts = aggregated_counts[offset:offset + value_count]
            offset += value_count
            if self.rule_sources[rule_idx] == "factorbase":
                derived_rows = self.derive_smoothed_rule_rows(
                    rule=self.rules[rule_idx],
                    smoothed_rows=rows,
                    smoothed_columns=self.value_columns[rule_idx],
                    cp_rows=self.cp_reference_values[rule_idx],
                    cp_columns=self.cp_reference_columns[rule_idx],
                    local_mults=rule_counts,
                )
            else:
                derived_rows = [list(row) for row in rows]
            derived_values.append(
                self._prune_derived_smoothed_rows(rule_idx, derived_rows)
            )

        full_count = sum(len(rows) for rows in self.values)
        pruned_count = sum(len(rows) for rows in derived_values)
        self.values = derived_values
        self.data_driven_smoothed_pruning_pending = False
        self._build_motif_group_masks()
        print(
            "  cp_smoothed data-driven pruning: "
            f"{pruned_count} / {full_count} combinations kept after "
            "local_mult, CP, and prior derivation"
        )
        return {
            "full_combinations": full_count,
            "pruned_combinations": pruned_count,
        }

    # ------------------------------------------------------------------
    # Main entry point — batched (PARALLELISED OVER GRAPHS)
    # ------------------------------------------------------------------

    def count_batch(
        self,
        preprocessor: 'DataPreprocessor',
        batch_size: int = 1000,
        selected_rules_values: Optional[Dict] = None,
        output_mode: str = "total_count",
        detach_to_cpu: bool = False,
        histogram_num_bins: int = 16,
        histogram_smoothing: float = 0.25,
        histogram_spec: Optional[Dict[str, torch.Tensor]] = None,
    ) -> MotifBatchResult:
        """
        Evaluate motifs for all graphs via batched GPU tensor ops.

        Structured modes materialize the canonical padded full matrices first.
        ``total_count`` uses a scalar fast path so a large unpruned smoothed
        value inventory can be counted before pruning without retaining one
        ``N_max x N_max`` tensor per combination.

        Canonical ``output_mode`` values are:

        * ``total_count``: one scalar per graph/motif, summed over all entries;
        * ``full_matrix``: padded ``N_max x N_max`` results plus a valid mask;
        * ``row_column_marginals``: shape-aware row/column sums with shape
          ``(B, M, 2, N_max)`` plus a valid mask;
        * ``marginal_histogram``: soft histograms of those marginals with shape
          ``(B, M, 2, histogram_num_bins)``, their mask, and the fixed histogram
          specification used for both observed and reconstructed graphs.
        * ``degree_histogram``: GraphVAE-MM triangular soft histograms of row
          sums from natural square motif matrices, shape ``(B, M, N_max)``.
        * ``kiarash_statistics``: the heterogeneous GraphVAE-MM bundle
          ``P^1..P^5``, in/out degree histograms, and total triangles; this
          requires exactly one natural square unit-edge motif.

        Legacy ``count`` and ``matrix`` names remain aliases for
        ``total_count`` and ``full_matrix`` respectively.

        No .item() is called anywhere in either path — gradient flows intact
        through all bmm and padding operations back to the adjacency tensors.

        For inference/display, call .detach() on the result.
        For training loss, use the result directly in F.mse_loss() etc.

        Parameters
        ----------
        preprocessor : DataPreprocessor
        batch_size   : graphs per GPU mini-batch
        selected_rules_values : dict, optional — subset of rules to count
        output_mode  : motif statistic representation; see above
        detach_to_cpu: detach each completed graph batch and collect it on CPU;
                       intended for fixed real-data targets, never predictions
        histogram_num_bins: number of soft bins for ``marginal_histogram``
        histogram_smoothing: sigmoid-boundary temperature as a fraction of bin width
        histogram_spec: observed-data bin edges/temperatures to reuse for predictions

        Returns
        -------
        torch.Tensor for ``total_count``;
        tuple[values, valid_mask] for matrix/marginal/composite modes; or
        tuple[histograms, valid_mask, histogram_spec] for histogram mode
        """
        output_mode = canonicalize_motif_output_mode(output_mode)
        batch_tensors = []
        matrix_valid_mask = None
        total  = preprocessor.num_graphs
        N_max  = preprocessor.N_max
        fom    = preprocessor.feature_onehot_mapping

        for start in range(0, total, batch_size):
            end_excl = min(start + batch_size, total)
            B        = end_excl - start
            t0       = time.perf_counter()

            feat_b, feat_onehot_b, adj_b, edge_b = preprocessor.get_batch(start, end_excl)

            if output_mode == "total_count":
                batch_result = self._iteration_total_counts_batched(
                    feat_b, feat_onehot_b, edge_b, adj_b, fom, B, N_max,
                    selected_rules_values,
                )
                if detach_to_cpu:
                    batch_result = batch_result.detach().cpu()
            else:
                batch_result, batch_valid_mask = self._iteration_function_batched(
                    feat_b, feat_onehot_b, edge_b, adj_b, fom, B, N_max,
                    selected_rules_values,
                )
                if detach_to_cpu:
                    batch_result = batch_result.detach().cpu()
                    batch_valid_mask = batch_valid_mask.detach().cpu()
                if matrix_valid_mask is None:
                    matrix_valid_mask = batch_valid_mask
                elif not torch.equal(matrix_valid_mask, batch_valid_mask):
                    raise RuntimeError(
                        "Motif matrix shapes changed between graph batches; "
                        "cannot construct one consistent validity mask."
                    )

            batch_tensors.append(batch_result)

            # Sync so elapsed time reflects actual GPU completion, not just launch.
            if str(self.device).startswith('cuda'):
                torch.cuda.synchronize()

            elapsed        = time.perf_counter() - t0
            graphs_per_sec = B / elapsed if elapsed > 0 else float('inf')
            eta_sec        = (total - end_excl) / graphs_per_sec if graphs_per_sec > 0 else 0
            print(
                f"  Batch {start:>7}–{end_excl-1:<7}  [{B:>5} graphs]"
                f"  {elapsed:>6.2f}s"
                f"  ({graphs_per_sec:>8.1f} graphs/s)"
                f"  {end_excl}/{total} done"
                f"  ETA {self._fmt_time(eta_sec)}"
            )

        if output_mode == "total_count":
            return torch.cat(batch_tensors, dim=0)

        full_matrices = torch.cat(batch_tensors, dim=0)
        if output_mode == "full_matrix":
            return full_matrices, matrix_valid_mask

        values, valid_mask, histogram_spec = represent_full_motif_matrices(
            full_matrices=full_matrices,
            matrix_mask=matrix_valid_mask,
            output_mode=output_mode,
            histogram_num_bins=histogram_num_bins,
            histogram_smoothing=histogram_smoothing,
            histogram_spec=histogram_spec,
        )
        if output_mode == "marginal_histogram":
            return values, valid_mask, histogram_spec
        return values, valid_mask

    def _iteration_total_counts_batched(
        self,
        feat_b: torch.Tensor,
        feat_onehot_b: torch.Tensor,
        edge_b: Optional[List[torch.Tensor]],
        adj_b: Dict[str, torch.Tensor],
        feature_onehot_mapping: Dict[int, Dict[int, int]],
        B: int,
        N_max: int,
        selected_rules_values: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Count motifs directly as scalars without retaining full matrices."""
        if selected_rules_values is not None:
            iteration_plan = [
                (rule_idx, self.values[rule_idx][value_idx])
                for rule_idx, value_indices in selected_rules_values.items()
                for value_idx in value_indices
            ]
        else:
            iteration_plan = [
                (rule_idx, table_row)
                for rule_idx in range(len(self.rules))
                for table_row in self.values[rule_idx]
            ]

        motif_counts: List[torch.Tensor] = []
        for rule_idx, table_row in iteration_plan:
            unmasked = self._compute_unmasked_matrices_batched(
                rule_idx,
                table_row,
                feat_b,
                feat_onehot_b,
                feature_onehot_mapping,
                edge_b,
                adj_b,
                B,
                N_max,
            )
            masked = self._compute_masked_matrices_batched(
                unmasked,
                self.base_indices[rule_idx],
                self.mask_indices[rule_idx],
            )
            sorted_matrices = self._compute_sorted_matrices_batched(
                masked,
                self.sort_indices[rule_idx],
            )
            stacked = self._compute_stacked_matrices_batched(
                sorted_matrices,
                self.stack_indices[rule_idx],
                B,
            )
            motif_counts.append(self._compute_result_batched(stacked))

        if not motif_counts:
            device = next(iter(adj_b.values())).device
            return torch.zeros(B, 0, dtype=torch.float32, device=device)
        return torch.stack(motif_counts, dim=1)

    # ------------------------------------------------------------------
    # Batched iteration loop  (unified — fully differentiable)
    # ------------------------------------------------------------------

    def _iteration_function_batched(
        self,
        feat_b:                torch.Tensor,                   # (B, N_max, F)
        feat_onehot_b:         torch.Tensor,                   # (B, N_max, D)
        edge_b:                Optional[List[torch.Tensor]],   # list[(B,C,N_max,N_max)] or None
        adj_b:                 Dict[str, torch.Tensor],        # {rel: (B, N_max, N_max)}
        feature_onehot_mapping: Dict[int, Dict[int, int]],
        B:                     int,
        N_max:                 int,
        selected_rules_values: Optional[Dict] = None,
    ) -> MotifBatchResult:
        """
        Return canonical padded full motif matrices and their validity masks.

        Representation reduction intentionally happens outside the counting
        loop, allowing multiple group-specific losses to share this one result.
        Neither path calls ``.item()`` or detaches, so gradients flow back to
        adjacency and feature tensors.

        count_batch()               — call .detach() on result for display/inference
        training loss               — use result directly in F.mse_loss() etc.
        """
        if selected_rules_values is not None:
            iteration_plan = [
                (rule_idx, value_idx, self.values[rule_idx][value_idx])
                for rule_idx, value_indices in selected_rules_values.items()
                for value_idx in value_indices
            ]
        else:
            iteration_plan = [
                (table, indexx, table_row)
                for table in range(len(self.rules))
                for indexx, table_row in enumerate(self.values[table])
            ]

        motif_tensors: List[torch.Tensor] = []
        matrix_masks: List[torch.Tensor] = []

        for table, indexx, table_row in iteration_plan:

            unmasked = self._compute_unmasked_matrices_batched(
                table, table_row,
                feat_b, feat_onehot_b, feature_onehot_mapping,
                edge_b, adj_b, B, N_max
            )
            masked  = self._compute_masked_matrices_batched(
                unmasked, self.base_indices[table], self.mask_indices[table]
            )
            sorted_ = self._compute_sorted_matrices_batched(
                masked, self.sort_indices[table]
            )
            stacked = self._compute_stacked_matrices_batched(
                sorted_, self.stack_indices[table], B
            )
            result_matrix = self._compute_result_matrix_batched(stacked)
            result, result_mask = pad_full_motif_matrix(
                result_matrix,
                n_max=N_max,
            )
            matrix_masks.append(result_mask)

            motif_tensors.append(result)

            del unmasked, masked, sorted_, stacked

        if not motif_tensors:
            device = next(iter(adj_b.values())).device
            return (
                torch.zeros(B, 0, N_max, N_max, dtype=torch.float32, device=device),
                torch.zeros(0, N_max, N_max, dtype=torch.bool, device=device),
            )

        values = torch.stack(motif_tensors, dim=1)
        return values, torch.stack(matrix_masks, dim=0)

    # ------------------------------------------------------------------
    # Batched state handlers
    # ------------------------------------------------------------------

    def _compute_unmasked_matrices_batched(
        self,
        table:                 int,
        table_row,
        feat_b:                torch.Tensor,                  # (B, N_max, F)
        feat_onehot_b:         torch.Tensor,                  # (B, N_max, D)
        feature_onehot_mapping: Dict[int, Dict[int, int]],
        edge_b:                Optional[List[torch.Tensor]],
        adj_b:                 Dict[str, torch.Tensor],
        B:                     int,
        N_max:                 int,
    ) -> List[torch.Tensor]:
        """Batched counterpart of _compute_unmasked_matrices (mode='test' path only)."""
        unmasked: List[torch.Tensor] = []

        for column in range(len(self.rules[table])):
            functor             = self.functors[table][column]
            table_functor_value = table_row[column + self.multiples[table]]
            state               = self.states[table][column]

            if state == 0:
                unmasked.append(
                    self._compute_state_zero_batched(
                        functor, table_functor_value,
                        feat_b, feat_onehot_b, feature_onehot_mapping,
                    )
                )
            elif state == 1:
                mats = self._compute_state_one_batched(
                    functor, table_functor_value,
                    self.variables[table][column],
                    self.masks[table][column],
                    feat_b, feat_onehot_b, feature_onehot_mapping,
                )
                unmasked.extend(mats)
            elif state == 2:
                unmasked.append(
                    self._compute_state_two_batched(functor, table_functor_value, adj_b)
                )
            elif state == 3:
                unmasked.append(
                    self._compute_state_three_batched(edge_b, functor, table_functor_value)
                )

        return unmasked

    def _compute_state_zero_batched(
        self,
        functor:               str,
        table_functor_value,
        feat_b:                torch.Tensor,                  # (B, N_max, F)
        feat_onehot_b:         torch.Tensor,                  # (B, N_max, D)
        feature_onehot_mapping: Dict[int, Dict[int, int]],
    ) -> torch.Tensor:
        """
        Unary feature predicate.
        Returns (B, N_max, 1)  — 1 where node matches the predicate, 0 elsewhere.

        GRADIENT-SAFE: uses a direct column slice of the pre-built one-hot
        matrix instead of the old boolean comparison `(fv == val).float()`.

        Padding rows in feat_onehot_b are all-zero by construction, so padded
        nodes naturally contribute 0 to all downstream products.
        """
        found, indx, _ = self._find_feature(functor)
        if found:
            val      = int(table_functor_value)
            col_map  = feature_onehot_mapping.get(indx, {})
            if val in col_map:
                oh_col = col_map[val]
                # Direct index into pre-built one-hot — no == comparison here
                return feat_onehot_b[:, :, oh_col].unsqueeze(2)      # (B, N_max, 1)
            else:
                # Value not seen during preprocessing — return zeros
                return torch.zeros(
                    feat_onehot_b.shape[0], feat_onehot_b.shape[1], 1,
                    dtype=torch.float32, device=self.device,
                )

        # Fallback: treat value as a raw feature column index (e.g. label column)
        col = int(table_functor_value)
        return feat_b[:, :, col].float().unsqueeze(2)                 # (B, N_max, 1)

    def _compute_state_one_batched(
        self,
        functor:               str,
        table_functor_value,
        variable:              str,
        masks_list:            List,
        feat_b:                torch.Tensor,                  # (B, N_max, F)
        feat_onehot_b:         torch.Tensor,                  # (B, N_max, D)
        feature_onehot_mapping: Dict[int, Dict[int, int]],
    ) -> List[torch.Tensor]:
        """
        Masked-variable predicate.
        Returns one (B, N_max, 1) or (B, 1, N_max) tensor per mask entry.

        GRADIENT-SAFE: same one-hot index strategy as _compute_state_zero_batched.
        """
        mats: List[torch.Tensor] = []
        found, indx, _ = self._find_feature(functor)

        for mask_info in masks_list:
            if found:
                val     = int(table_functor_value)
                col_map = feature_onehot_mapping.get(indx, {})
                if val in col_map:
                    oh_col   = col_map[val]
                    col_vals = feat_onehot_b[:, :, oh_col]            # (B, N_max)
                else:
                    col_vals = torch.zeros(
                        feat_onehot_b.shape[0], feat_onehot_b.shape[1],
                        dtype=torch.float32, device=self.device,
                    )

                if variable == mask_info[1]:
                    mats.append(col_vals.unsqueeze(2))                # (B, N_max, 1)
                else:
                    mats.append(col_vals.unsqueeze(1))                # (B, 1, N_max)
            else:
                # Fallback: raw feature column
                col = int(table_functor_value)
                fv  = feat_b[:, :, col].float()                       # (B, N_max)
                if variable == mask_info[1]:
                    mats.append(fv.unsqueeze(2))                      # (B, N_max, 1)
                else:
                    mats.append(fv.unsqueeze(1))                      # (B, 1, N_max)

        return mats

    def _compute_state_two_batched(
        self,
        functor: str,
        table_functor_value,
        adj_b: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Adjacency / relation matrix.
        Returns (B, N_max, N_max).
        """
        adj = adj_b[functor]                                          # (B, N_max, N_max)
        return (1 - adj) if table_functor_value == 'F' else adj

    def _compute_state_three_batched(
        self,
        edge_b: List[torch.Tensor],   # list of (B, C, N_max, N_max)
        functor: str,
        table_functor_value,
    ) -> torch.Tensor:
        """
        Edge feature predicate (QM9 bond types).
        Returns (B, N_max, N_max).
        """
        feature_idx = next(
            idx for idx, info in self.feature_info_mapping.items()
            if info['feature_name'] == functor
        )
        target = edge_b[feature_idx]                                  # (B, C, N_max, N_max)

        if table_functor_value == 'N/A':
            return torch.sum(target, dim=1)                           # (B, N_max, N_max)

        value_mapping   = self.feature_info_mapping[feature_idx]['value_index_mapping']
        reverse_mapping = {v: k for k, v in value_mapping.items()}
        val_idx         = reverse_mapping[int(table_functor_value)]
        return target[:, val_idx, :, :]                               # (B, N_max, N_max)

    # ------------------------------------------------------------------
    # Batched matrix algebra
    # ------------------------------------------------------------------

    def _compute_masked_matrices_batched(
        self,
        unmasked:     List[torch.Tensor],
        base_indices: List[int],
        mask_indices: List[List[int]],
    ) -> List[torch.Tensor]:
        """
        Element-wise masking — identical logic to the single-graph version.
        Tensors carry a leading batch dimension but broadcasting handles it.
        """
        masked = [unmasked[k] for k in base_indices]
        for k in mask_indices:
            masked[k[0]] = masked[k[0]] * unmasked[k[1]]
        return masked

    def _compute_sorted_matrices_batched(
        self,
        masked:       List[torch.Tensor],
        sort_indices: List,
    ) -> List[torch.Tensor]:
        """
        Transpose swaps dims 1 and 2 (batch dim 0 is untouched).
          (B, N, 1)  ↔  (B, 1, N)
          (B, N, N)  ↔  (B, N, N)^T
        """
        result = []
        for si in sort_indices:
            m = masked[si[1]]
            result.append(m.transpose(1, 2) if si[0] else m)
        return result

    def _compute_stacked_matrices_batched(
        self,
        sorted_:       List[torch.Tensor],
        stack_indices: List,
        B:             int,
    ) -> List[torch.Tensor]:
        """
        Batched matrix chain multiplication using torch.bmm.
        The diagonal masking step uses a (B, N, N) identity expanded over B.

        Key shapes (example for a 3-atom rule):
          (B, 1, N) @ (B, N, N) = (B, 1, N)
          (B, 1, N) @ (B, N, 1) = (B, 1, 1)   → squeezed to (B,) by _compute_result_batched
        """
        stacked     = sorted_.copy()
        pop_counter = 0

        for k in stack_indices:
            for _ in range(k[1] - k[0] - pop_counter):
                stacked[k[0]] = torch.bmm(stacked[k[0]], stacked[k[0] + 1])
                stacked.pop(k[0] + 1)
                pop_counter += 1

            # Diagonal masking — only for square matrices
            mat = stacked[k[0]]
            if mat.shape[1] == mat.shape[2]:
                N   = mat.shape[1]
                eye = (
                    torch.eye(N, dtype=torch.float32, device=self.device)
                    .unsqueeze(0)
                    .expand(B, -1, -1)
                )
                stacked[k[0]] = mat * eye

        return stacked

    def _compute_result_batched(
        self,
        stacked: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Final batched chain multiply → sum all spatial dims → (B,).

        The spatial shape after multiplication is NOT always (B, 1, 1):
          - Relational rules  (B, 1, N) @ (B, N, N) @ (B, N, 1) = (B, 1, 1)
          - Unary rules       result stays (B, N, 1) or (B, N, N)

        Mirroring the single-graph path which does torch.sum(result) over
        the whole matrix, we flatten every dim except the batch dim and sum.
        This is correct for all rule types.
        """
        result = self._compute_result_matrix_batched(stacked)
        return compute_total_motif_count(result)

    def _compute_result_matrix_batched(
        self,
        stacked: List[torch.Tensor],
    ) -> torch.Tensor:
        """Return the full final matrix-chain result without summing it."""
        result = stacked[0]
        for k in range(1, len(stacked)):
            result = torch.bmm(result, stacked[k])
        return result

    @staticmethod
    def _pad_result_matrix_batched(
        result: torch.Tensor,
        N_max: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bottom/right-pad one motif result to ``(B, N_max, N_max)``.

        Matrix-chain results always retain the batch dimension and have two
        spatial dimensions, but those dimensions may be 1 or ``N_max``. The
        boolean mask is graph-independent because the chain shape is fixed by
        the rule rather than by a particular graph.
        """
        return pad_full_motif_matrix(result, n_max=N_max)
    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """Human-readable duration string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        m, s = divmod(seconds, 60)
        if m < 60:
            return f"{int(m)}m {s:.0f}s"
        h, m = divmod(m, 60)
        return f"{int(h)}h {int(m)}m {s:.0f}s"

    @staticmethod
    def _strip_trailing_digits(variable_name: str) -> str:
        return variable_name.rstrip("0123456789")

    @staticmethod
    def _parse_atom(atom: str) -> Tuple[str, List[str]]:
        functor, rest = atom.split("(", 1)
        arguments = rest[:-1].split(",")
        return functor, arguments

    def _infer_rule_sources_for_legacy_pickle(self, data: Dict) -> List[str]:
        """
        Old pickles did not mark which rules were injected by the cache builder.
        Treat all rules as FactorBase-derived unless the pickle explicitly says
        it was built in a literal-only mode, in which case the only defensible
        fallback is to treat syntactic-shaped rules as synthetic literals.
        """
        stored_mode = data.get("syntactic_literal_rule_mode")
        if stored_mode == "literals":
            return [
                "synthetic_literal" if self._is_syntactic_literal_rule(rule) else "factorbase"
                for rule in self.rules
            ]
        return ["factorbase"] * len(self.rules)

    def _is_entity_syntactic_literal_rule(self, rule: List[str]) -> bool:
        if len(rule) != 1:
            return False

        functor, arguments = self._parse_atom(rule[0])
        if len(arguments) != 1:
            return False

        variable_name = arguments[0]
        for table_name, feature_list in self.entity_feature_columns.items():
            if functor in feature_list and self._strip_trailing_digits(variable_name) == table_name:
                return True
        return False

    def _is_relation_syntactic_literal_rule(self, rule: List[str]) -> bool:
        if len(rule) != 2:
            return False

        parsed_atoms = [self._parse_atom(atom) for atom in rule]

        for relation_name, feature_list in self.relation_feature_columns.items():
            entity_tables = self.relation_entity_tables.get(relation_name)
            if entity_tables is None:
                continue

            feature_arguments = None
            relation_arguments = None
            for functor, arguments in parsed_atoms:
                if functor in feature_list:
                    feature_arguments = arguments
                elif functor == relation_name:
                    relation_arguments = arguments

            if feature_arguments is None or relation_arguments is None:
                continue
            if len(feature_arguments) != 2 or len(relation_arguments) != 2:
                continue
            if feature_arguments != relation_arguments:
                continue

            variable_bases = tuple(
                self._strip_trailing_digits(variable_name)
                for variable_name in relation_arguments
            )
            if variable_bases == tuple(entity_tables):
                return True

        return False

    def _is_syntactic_literal_rule(self, rule: List[str]) -> bool:
        return (
            self._is_entity_syntactic_literal_rule(rule)
            or self._is_relation_syntactic_literal_rule(rule)
        )

    def _is_unit_relation_rule(self, rule: List[str]) -> bool:
        """Return whether a rule is one bare binary database relation atom."""
        if len(rule) != 1:
            return False
        functor, arguments = self._parse_atom(rule[0])
        return functor in self.relations and len(arguments) == 2

    def _is_positive_unit_relation_value(self, rule_idx: int, table_row) -> bool:
        """Identify value rows that materialize a relation, not its complement."""
        if not self._is_unit_relation_rule(self.rules[rule_idx]):
            return False
        relation_value = table_row[self.multiples[rule_idx]]
        # State-two counting uses the relation adjacency for every value except
        # the explicit false state, for which it uses the complement.
        return relation_value != 'F'

    def _select_motif_values(self, data: Dict, pickle_path: Path):
        """Select motif rows while preserving every single-atom rule.

        New caches already store all rows for single-atom rules in
        ``values_pruned``. Restoring them from ``values_full`` here also makes
        the guarantee apply to caches generated before that exemption existed.
        The unit-relation option remains a relation-specific safety check for
        the protected positive rows.
        """
        rule_prune = getattr(self.args, 'rule_prune', False)
        protect_unit_relations = getattr(
            self.args,
            'protect_unit_relation_motifs_from_pruning',
            False,
        )

        if "values_full" not in data:
            if rule_prune and protect_unit_relations:
                raise RuntimeError(
                    "Cannot protect unit-relation motifs with an old-format "
                    f"cache at {pickle_path}; regenerate it with values_full."
                )
            print(
                f"  Warning: old-format pickle — delete {pickle_path} "
                "to regenerate with both value sets cached."
            )
            return [list(rows) for rows in data["values"]]

        if not rule_prune:
            values = [list(rows) for rows in data["values_full"]]
            print(
                "  rule_prune=False: using all "
                f"{sum(len(rows) for rows in values)} value combinations"
            )
            return values

        values = [
            list(full_rows) if len(rule) == 1 else list(pruned_rows)
            for rule, full_rows, pruned_rows in zip(
                self.rules,
                data["values_full"],
                data["values_pruned"],
            )
        ]
        restored_rule_indices = []
        if protect_unit_relations:
            for rule_idx, rule in enumerate(self.rules):
                if not self._is_unit_relation_rule(rule):
                    continue
                selected_negative_rows = [
                    row for row in values[rule_idx]
                    if not self._is_positive_unit_relation_value(rule_idx, row)
                ]
                selected_positive_count = (
                    len(values[rule_idx]) - len(selected_negative_rows)
                )
                full_positive_rows = [
                    row for row in data["values_full"][rule_idx]
                    if self._is_positive_unit_relation_value(rule_idx, row)
                ]
                if selected_positive_count != len(full_positive_rows):
                    restored_rule_indices.append(rule_idx)
                values[rule_idx] = selected_negative_rows + list(full_positive_rows)

        n_full = sum(len(rows) for rows in data["values_full"])
        n_selected = sum(len(rows) for rows in values)
        print(
            "  rule_prune=True: "
            f"{n_selected} / {n_full} value combinations kept "
            "(formula-pruned; single-atom rules keep all rows)"
        )
        if protect_unit_relations:
            print(
                "  protected unit-relation motifs: "
                f"restored {len(restored_rule_indices)} rule(s), "
                f"indices={restored_rule_indices}"
            )
        return values

    def _filter_rules_for_runtime_mode(self) -> None:
        mode = self.syntactic_literal_rule_mode
        if mode == "both":
            print(f"  syntactic_literal_rule_mode=both: using all {len(self.rules)} rules")
            return

        if mode == "original":
            keep_indices = [
                rule_idx for rule_idx, source in enumerate(self.rule_sources)
                if source != "synthetic_literal"
            ]
        elif mode == "literals":
            keep_indices = [
                rule_idx for rule_idx, rule in enumerate(self.rules)
                if self._is_syntactic_literal_rule(rule)
            ]
        else:
            raise ValueError(f"Unknown syntactic_literal_rule_mode: {mode}")

        print(
            f"  syntactic_literal_rule_mode={mode}: "
            f"{len(keep_indices)} / {len(self.rules)} rules kept after loading cache"
        )
        self._keep_rule_indices(keep_indices)

    def _keep_rule_indices(self, keep_indices: List[int]) -> None:
        list_attrs = (
            "rules",
            "multiples",
            "states",
            "values",
            "value_columns",
            "cp_reference_values",
            "cp_reference_columns",
            "rule_sources",
            "base_indices",
            "mask_indices",
            "sort_indices",
            "stack_indices",
        )
        dict_attrs = ("functors", "variables", "nodes", "masks")

        for attr in list_attrs:
            if not hasattr(self, attr):
                continue
            old_values = getattr(self, attr)
            setattr(self, attr, [old_values[old_idx] for old_idx in keep_indices])

        for attr in dict_attrs:
            old_values = getattr(self, attr)
            setattr(
                self,
                attr,
                {new_idx: old_values[old_idx] for new_idx, old_idx in enumerate(keep_indices)}
            )

    def _build_motif_group_masks(self):
        rule_mask: List[bool] = []
        motif_mask: List[bool] = []

        if self.use_syntactic_literal_rules:
            for rule_idx, rule in enumerate(self.rules):
                is_literal_rule = self._is_syntactic_literal_rule(rule)
                rule_mask.append(is_literal_rule)
                motif_mask.extend([is_literal_rule] * len(self.values[rule_idx]))
        else:
            rule_mask = [False] * len(self.rules)
            motif_mask = [False] * sum(len(value_rows) for value_rows in self.values)

        self.syntactic_literal_rule_mask = rule_mask
        self.syntactic_literal_rule_indices = [
            rule_idx for rule_idx, is_literal_rule in enumerate(rule_mask)
            if is_literal_rule
        ]
        self.syntactic_literal_motif_mask = torch.tensor(motif_mask, dtype=torch.bool)
        self.num_syntactic_literal_motifs = int(self.syntactic_literal_motif_mask.sum().item())
        self.num_non_syntactic_literal_motifs = int(
            self.syntactic_literal_motif_mask.numel() - self.num_syntactic_literal_motifs
        )

        self.unit_relation_rule_mask = []
        unit_relation_motif_mask: List[bool] = []
        for rule_idx, rule in enumerate(self.rules):
            is_unit_relation_rule = self._is_unit_relation_rule(rule)
            row_mask = [
                is_unit_relation_rule
                and self._is_positive_unit_relation_value(rule_idx, table_row)
                for table_row in self.values[rule_idx]
            ]
            self.unit_relation_rule_mask.append(any(row_mask))
            unit_relation_motif_mask.extend(row_mask)
        self.unit_relation_rule_indices = [
            rule_idx
            for rule_idx, is_unit_relation in enumerate(self.unit_relation_rule_mask)
            if is_unit_relation
        ]
        self.unit_relation_motif_mask = torch.tensor(
            unit_relation_motif_mask,
            dtype=torch.bool,
        )
        self.num_unit_relation_motifs = int(
            self.unit_relation_motif_mask.sum().item()
        )

    def _find_feature(self, functor: str) -> Tuple[bool, Optional[int], Optional[str]]:
        for key, feature_list in self.entity_feature_columns.items():
            if functor in feature_list:
                return True, feature_list.index(functor), key
        for key, feature_list in self.relation_feature_columns.items():
            if functor in feature_list:
                return True, feature_list.index(functor), key
        return False, None, None

    # ------------------------------------------------------------------
    # Aggregation & display
    # ------------------------------------------------------------------

    def get_rule_motif_mapping(self) -> List[Tuple[int, int]]:
        return [(i, len(self.values[i])) for i in range(len(self.rules))]

    def aggregate_motif_counts(self, counts: torch.Tensor) -> torch.Tensor:
        """
        Sum motif counts across all graphs.

        Parameters
        ----------
        counts : (num_graphs, num_motifs) tensor — output of count_batch()

        Returns
        -------
        (num_motifs,) tensor — summed counts, gradient intact.
        Call .detach().tolist() for display.
        """
        return counts.sum(dim=0)                                        # (num_motifs,)

    def display_rules_and_motifs(
        self, aggregated_counts: torch.Tensor, selected_rules_values: Dict = None
    ):
        # Convert to plain list only at the display boundary
        counts_list = aggregated_counts.detach().cpu().tolist()
        print("\n" + "="*80)
        print("RULES AND MOTIF COUNTS")
        print("="*80)
        if selected_rules_values is not None:
            self._display_selective_results(counts_list, selected_rules_values)
        else:
            self._display_full_results(counts_list)

    def _display_full_results(self, aggregated_counts: List[float]):
        count_idx = 0
        for rule_idx in range(len(self.rules)):
            rule = self.rules[rule_idx]
            num_values = len(self.values[rule_idx])
            start_idx = self.multiples[rule_idx]
            print(f"\nRule {rule_idx + 1}: {rule}")
            print("-" * 80)
            for value_idx in range(num_values):
                table_row = self.values[rule_idx][value_idx]
                functor_vals = [
                    f"{f}={table_row[start_idx + fi]}"
                    for fi, f in enumerate(rule)
                    if start_idx + fi < len(table_row)
                ]
                print(
                    f"  [{value_idx}] "
                    + (", ".join(functor_vals) if functor_vals else f"Value {value_idx + 1}/{num_values}")
                    + f" -> {aggregated_counts[count_idx]:.4f}"
                )
                count_idx += 1

    def _display_selective_results(
        self, aggregated_counts: List[float], selected_rules_values: Dict
    ):
        count_idx = 0
        for rule_idx, value_indices in selected_rules_values.items():
            rule = self.rules[rule_idx]
            print(f"\nRule {rule_idx + 1}: {rule}")
            print("-" * 80)
            start_idx = self.multiples[rule_idx]
            for value_idx in value_indices:
                count     = aggregated_counts[count_idx]
                table_row = self.values[rule_idx][value_idx]
                functor_vals = [
                    f"{f}={table_row[start_idx + fi]}"
                    for fi, f in enumerate(rule)
                    if start_idx + fi < len(table_row)
                ]
                print(f"  [{value_idx}] {', '.join(functor_vals)} -> {count:.4f}")
                count_idx += 1

    # ------------------------------------------------------------------
    # Interactive selection helpers
    # ------------------------------------------------------------------

    def _interactive_rule_selection(self) -> Dict:
        print("\n" + "="*80)
        print("AVAILABLE RULES")
        print("="*80)

        for rule_idx in range(len(self.rules)):
            print(f"\n[{rule_idx}] Rule {rule_idx + 1}: {self.rules[rule_idx]}")
            print(f"    Number of value combinations: {len(self.values[rule_idx])}")

        print("\n" + "="*80)

        while True:
            rule_selection = input(
                "\nEnter rule indices to count (comma-separated, or 'all'): "
            ).strip()
            if rule_selection.lower() == 'all':
                selected_rule_indices = list(range(len(self.rules)))
                break
            try:
                selected_rule_indices = [int(x.strip()) for x in rule_selection.split(',')]
                if all(0 <= idx < len(self.rules) for idx in selected_rule_indices):
                    break
                print(f"Error: indices must be 0-{len(self.rules)-1}")
            except ValueError:
                print("Error: enter numbers separated by commas, or 'all'")

        selected_rules_values = {}
        for rule_idx in selected_rule_indices:
            print(f"\n{'='*80}")
            print(f"Selecting values for Rule {rule_idx + 1}: {self.rules[rule_idx]}")
            print("="*80)

            functor_value_options = self._get_functor_value_options(rule_idx)
            if not functor_value_options:
                print("No value combinations available. Skipping.")
                continue

            selected_functor_values = {}
            for functor_name, unique_values in functor_value_options.items():
                print(f"\n{functor_name}\n  Possible values: {unique_values}")
                while True:
                    val_sel = input("  Select values (comma-separated, or 'all'): ").strip()
                    if val_sel.lower() == 'all':
                        selected_functor_values[functor_name] = unique_values
                        break
                    selected_vals, invalid_vals = [], []
                    for v in val_sel.split(','):
                        matched = self._match_value_to_options(v.strip(), unique_values)
                        if matched is not None:
                            selected_vals.append(matched)
                        else:
                            invalid_vals.append(v.strip())
                    for iv in invalid_vals:
                        print(f"  Warning: '{iv}' is not a valid option")
                    if selected_vals:
                        selected_functor_values[functor_name] = selected_vals
                        break
                    print("  Error: no valid values selected. Try again.")

            while True:
                filtered = self._filter_combinations_by_functor_values(
                    rule_idx, selected_functor_values
                )
                if filtered:
                    print(f"\n  -> {len(filtered)} combinations match your selection")
                    break
                print(f"\n  -> 0 combinations match your selection — no rows in the database "
                      f"have this exact combination. Please try different values.")
                # Re-prompt all functors for this rule
                selected_functor_values = {}
                for functor_name, unique_values in functor_value_options.items():
                    print(f"\n{functor_name}\n  Possible values: {unique_values}")
                    while True:
                        val_sel = input("  Select values (comma-separated, or 'all'): ").strip()
                        if val_sel.lower() == 'all':
                            selected_functor_values[functor_name] = unique_values
                            break
                        selected_vals, invalid_vals = [], []
                        for v in val_sel.split(','):
                            matched = self._match_value_to_options(v.strip(), unique_values)
                            if matched is not None:
                                selected_vals.append(matched)
                            else:
                                invalid_vals.append(v.strip())
                        for iv in invalid_vals:
                            print(f"  Warning: '{iv}' is not a valid option")
                        if selected_vals:
                            selected_functor_values[functor_name] = selected_vals
                            break
                        print("  Error: no valid values selected. Try again.")

            selected_rules_values[rule_idx] = filtered

        return selected_rules_values

    def _match_value_to_options(self, user_input: str, options: List) -> Any:
        if user_input in options:
            return user_input
        try:
            user_float = float(user_input)
            user_int   = int(user_float) if user_float == int(user_float) else None
            if user_float in options:           return user_float
            if user_int is not None:
                if user_int in options:         return user_int
                if str(user_int) in options:    return str(user_int)
            if str(user_float) in options:      return str(user_float)
        except ValueError:
            pass
        return None

    def _get_functor_value_options(self, rule_idx: int) -> Dict[str, List]:
        rule = self.rules[rule_idx]
        functor_values: Dict[str, set] = {f: set() for f in rule}
        start_idx = self.multiples[rule_idx]
        for table_row in self.values[rule_idx]:
            for fi, functor in enumerate(rule):
                vi = start_idx + fi
                if vi < len(table_row):
                    functor_values[functor].add(table_row[vi])
        return {
            f: sorted(list(vs), key=lambda x: (isinstance(x, str), x))
            for f, vs in functor_values.items()
        }

    def _filter_combinations_by_functor_values(
        self, rule_idx: int, selected_functor_values: Dict[str, List]
    ) -> List[int]:
        rule = self.rules[rule_idx]
        matching = []
        start_idx = self.multiples[rule_idx]
        for row_idx, table_row in enumerate(self.values[rule_idx]):
            matches = True
            for fi, functor in enumerate(rule):
                vi = start_idx + fi
                if vi < len(table_row) and functor in selected_functor_values:
                    if table_row[vi] not in selected_functor_values[functor]:
                        matches = False
                        break
            if matches:
                matching.append(row_idx)
        return matching

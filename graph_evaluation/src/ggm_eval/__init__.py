"""Reusable, PyG-first evaluation interfaces for graph generative models.

The package deliberately keeps graph generators and evaluation engines
separate.  Generator repositories only need to construct PyG ``Data`` objects
that satisfy :mod:`ggm_eval.contract`.  The optional DGL adapter exists for
historical artifacts and the legacy Random-GIN evaluator.
"""

from .contract import (
    FEATURE_MODES,
    CollectionSummary,
    collection_digest,
    prepare_collection,
    validate_collection,
    validate_pyg_graph,
)
from .io import (
    load_pyg_collection,
    load_pyg_collection_with_metadata,
    save_pyg_collection,
)

__all__ = [
    "FEATURE_MODES",
    "CollectionSummary",
    "collection_digest",
    "load_pyg_collection",
    "load_pyg_collection_with_metadata",
    "prepare_collection",
    "save_pyg_collection",
    "validate_collection",
    "validate_pyg_graph",
]

__version__ = "0.1.0"

"""Canonical dataset identities shared by exporters and trained evaluators.

Graph-generator command lines have historically used a few spellings for the
same dataset (for example, ``protein`` and ``PROTEINS``).  Artifact metadata
and trained-checkpoint lookup must use one stable identity, so normalization
lives in this backend-independent module rather than in a generator script.
"""

from __future__ import annotations


CANONICAL_DATASETS = (
    "AIDS",
    "ENZYMES",
    "MUTAG",
    "PROTEINS",
    "PTC",
    "QM9",
    "ogbg-molbbbp",
)

DATASET_ALIASES = {
    "aids": "AIDS",
    "enzymes": "ENZYMES",
    "enzymez": "ENZYMES",
    "mutag": "MUTAG",
    "protein": "PROTEINS",
    "proteins": "PROTEINS",
    "ptc": "PTC",
    "qm9": "QM9",
    "ogbg": "ogbg-molbbbp",
    "ogbg-molbbbp": "ogbg-molbbbp",
    "ogbg_molbbbp": "ogbg-molbbbp",
}


def normalize_dataset_name(raw_name: str) -> str:
    """Return the canonical identity for a supported dataset name.

    Parameters
    ----------
    raw_name:
        A canonical name or one of the documented case-insensitive aliases.

    Raises
    ------
    ValueError
        If the name has no supported canonical identity.
    """

    normalized = str(raw_name).strip().lower()
    canonical = DATASET_ALIASES.get(normalized)
    if canonical is None:
        raise ValueError(
            f"Unsupported dataset {raw_name!r}. Choose one of "
            f"{list(CANONICAL_DATASETS)}."
        )
    return canonical

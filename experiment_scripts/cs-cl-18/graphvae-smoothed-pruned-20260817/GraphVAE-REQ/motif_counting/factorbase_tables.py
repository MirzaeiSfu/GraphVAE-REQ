"""Naming helpers for the FactorBase tables used by motif counting."""


FACTORBASE_CP_TABLE_CHOICES = ("CP", "CP_smoothed")
DEFAULT_FACTORBASE_CP_TABLE = "CP_smoothed"
MOTIF_CACHE_SCHEMA_VERSION = 4


def normalize_factorbase_cp_table(source: str) -> str:
    """Normalize and validate a configured FactorBase CP table source."""
    normalized = str(source).strip().lstrip("_")
    if normalized not in FACTORBASE_CP_TABLE_CHOICES:
        choices = ", ".join(FACTORBASE_CP_TABLE_CHOICES)
        raise ValueError(
            f"Unknown FactorBase CP table source {source!r}; choose one of: {choices}"
        )
    return normalized


def factorbase_cp_table_from_args(args=None) -> str:
    configured = (
        getattr(args, "factorbase_cp_table", DEFAULT_FACTORBASE_CP_TABLE)
        if args is not None
        else DEFAULT_FACTORBASE_CP_TABLE
    )
    return normalize_factorbase_cp_table(configured)


def factorbase_cp_table_suffix(source: str = DEFAULT_FACTORBASE_CP_TABLE) -> str:
    return f"_{normalize_factorbase_cp_table(source)}"


def factorbase_cp_table_name(
    child: str,
    source: str = DEFAULT_FACTORBASE_CP_TABLE,
) -> str:
    """Return the FactorBase conditional-probability table for ``child``."""
    return f"{child}{factorbase_cp_table_suffix(source)}"


def motif_cache_filename(
    database_name: str,
    source: str = DEFAULT_FACTORBASE_CP_TABLE,
) -> str:
    """Keep caches from different FactorBase CP table sources separate."""
    return f"{database_name}{factorbase_cp_table_suffix(source)}.pkl"

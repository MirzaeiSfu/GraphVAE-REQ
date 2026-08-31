"""Loss-weight compatibility helpers for the original GraphVAE-MM setup."""

KIA_GRAPHVAE_MM_BCE_KL_BY_DATASET = {
    "AIDS": (50.0, 2000.0),
    "GRID": (50.0, 2000.0),
    "TRIANGULAR_GRID": (50.0, 2000.0),
    "LOBSTER": (40.0, 2000.0),
}


def apply_kia_bce_kl_weights(alpha, dataset, enabled):
    """Return ``alpha`` with Kia's dataset-specific BCE/KL tail weights.

    This deliberately changes only the last two base-VAE coefficients. It does
    not enable GraphVAE-MM statistics, so a plain GraphVAE can replace those
    statistics with motif and feature losses while retaining Kia's BCE/KL
    regularization.
    """
    resolved = list(alpha)
    if not enabled:
        return resolved

    dataset_key = str(dataset).upper()
    if dataset_key not in KIA_GRAPHVAE_MM_BCE_KL_BY_DATASET:
        supported = ", ".join(sorted(KIA_GRAPHVAE_MM_BCE_KL_BY_DATASET))
        raise ValueError(
            "use_graphvae_mm_bce_kl_weights is only defined for "
            f"{supported}; received dataset={dataset!r}."
        )
    resolved[-2], resolved[-1] = KIA_GRAPHVAE_MM_BCE_KL_BY_DATASET[dataset_key]
    return resolved

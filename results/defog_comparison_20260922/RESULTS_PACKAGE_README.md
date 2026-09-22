# GraphVAE-REQ, DeFoG, and FactorBase result package

This directory is the GitHub-facing snapshot of the verified experiment reports and learned FactorBase Bayesian-network XML files collected on 2026-09-22.

## Contents

- `datasets/<DATASET>/RESULTS.md`: the fullest validated report currently available for that dataset.
- `datasets/<DATASET>/Bif_<database>.xml`: the nonempty FactorBase BIF matching the database used by the selected motif=True campaign, when available.
- `MOTIF_TRUE_BEATS_DEFOG.md`: cross-dataset inventory of metrics on which motif=True has a better aggregate result than DeFoG.

## Important limitations

- A bold value is not a statistical-significance claim. The reports compare aggregate means and preserve their original seed/protocol qualifications.
- GRID uses two retained healthy DeFoG seeds. AIDS and corrected MUTAG also have only two usable DeFoG seeds.
- Some historical MUTAG and PROTEINS DeFoG artifacts are duplicated; their nominal seed SD is not independent-seed uncertainty.
- OGB has no verified common three-way evaluation. Its attempted multihop BIF is zero bytes and is deliberately excluded.
- LGD is outside this package, as requested.

## XML coverage

Verified nonempty BIF files are included for GRID, LOBSTER, TRIANGULAR_GRID, MUTAG, PTC, QM9, AIDS, and PROTEINS. OGB is the only dataset without a valid matching XML.

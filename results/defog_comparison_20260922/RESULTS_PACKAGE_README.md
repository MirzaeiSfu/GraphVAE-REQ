# GraphVAE-REQ, DeFoG, and FactorBase result package

This directory is the GitHub-facing snapshot of the verified experiment reports and learned FactorBase Bayesian-network XML files collected on 2026-09-22.

## Contents

- `datasets/<DATASET>/RESULTS.md`: the fullest validated report currently available for that dataset.
- `datasets/<DATASET>/Bif_<database>.xml`: the nonempty FactorBase BIF matching the database used by the selected motif=True campaign, when available.
- `MOTIF_TRUE_BEATS_DEFOG.md`: cross-dataset inventory of metrics on which motif=True has a better aggregate result than DeFoG.

## Important limitations

- A bold value is not a statistical-significance claim. The reports compare aggregate means and preserve their original seed/protocol qualifications.
- GRID now uses independent DeFoG seeds 0/4/5; AIDS uses independent seeds 3/4/5; TRIANGULAR_GRID uses healthy seeds 0/1/3.
- Corrected independent PROTEINS DeFoG outputs replace the duplicated historical collections for structural and RandomGIN reporting. Historical duplicated tables remain below the update section for provenance.
- OGB now has a completed topology-only common terminal-chain evaluation. It must remain separate from the historical GraphVAE final-output table; a controlled final-output three-way reevaluation is still pending. The attempted OGB multihop BIF is zero bytes and is deliberately excluded.
- LGD is outside this package, as requested.

## XML coverage

Verified nonempty BIF files are included for GRID, LOBSTER, TRIANGULAR_GRID, MUTAG, PTC, QM9, AIDS, and PROTEINS. OGB is the only dataset without a valid matching XML.

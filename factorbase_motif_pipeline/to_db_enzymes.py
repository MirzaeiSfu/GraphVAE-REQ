#!/usr/bin/env python3
"""Create a FactorBase-ready MySQL database from TU Dortmund ENZYMES."""

from tu_dataset_to_db import TU_DATASET_SPECS, main


if __name__ == "__main__":
    main(TU_DATASET_SPECS["ENZYMES"])

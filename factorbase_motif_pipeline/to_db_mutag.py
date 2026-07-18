#!/usr/bin/env python3
"""Create a FactorBase-ready MySQL database from DGL GIN MUTAG."""

from gin_dataset_to_db import main


if __name__ == "__main__":
    main("MUTAG")

"""Mechanically insert generated tables into the manuscript once."""
from pathlib import Path

root = Path(__file__).resolve().parent
paper = root.parent / 'final.md'
text = paper.read_text()
structural, gin = (root / 'tables.md').read_text().strip().split('\n\n')
assert text.count('<!-- STRUCTURAL_TABLE -->') == 1
assert text.count('<!-- GIN_TABLE -->') == 1
text = text.replace('<!-- STRUCTURAL_TABLE -->', structural)
text = text.replace('<!-- GIN_TABLE -->', gin)
paper.write_text(text)

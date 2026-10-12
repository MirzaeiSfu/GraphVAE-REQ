"""Check manuscript scope, generated numerical tables, and revision artifacts."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
paper_path = ROOT.parent / 'final.md'
paper = paper_path.read_text()
evidence = json.loads((ROOT / 'evidence.json').read_text())
tables = (ROOT / 'tables.md').read_text().strip().split('\n\n')
assert len(tables) == 2
assert all(t in paper for t in tables), 'Manuscript differs from generated tables'
assert len(tables[0].splitlines()) == 20, 'Expected header + separator + 18 structural rows'
assert len(tables[1].splitlines()) == 8, 'Expected header + separator + 6 F1 rows'
assert len(re.findall(r'^\|---', paper, flags=re.M)) == 2
assert not re.search(r'total[ _-]count', paper, flags=re.I)
assert not any(t in paper for t in ['<!-- STRUCTURAL_TABLE -->', '<!-- GIN_TABLE -->', 'TODO', 'TBD'])
assert paper.count('$$') % 2 == 0
equation_ids = re.findall(r'\\tag\{([^}]+)\}', paper)
assert len(equation_ids) == len(set(equation_ids))
for number in range(1, 8):
    assert re.search(rf'^## {number} ', paper, flags=re.M)

for table in tables:
    widths = [line.count('|') for line in table.splitlines()]
    assert len(set(widths)) == 1
for line in tables[1].splitlines()[2:]:
    columns = [x.strip().replace('**', '') for x in line.strip('|').split('|')]
    means = [float(x.split()[0]) for x in columns[2:5]]
    gains = [float(x.strip('%')) for x in columns[5:7]]
    for gain, base in zip(gains, [means[0], means[2]]):
        assert abs(gain - 100 * (means[1] - base) / base) < 0.0051
    assert means[1] > means[0], 'Narrative claims F1 improvement over GraphVAE'

hashes = {}
for i in range(7):
    path = ROOT / f'draft_{i:02}.md'
    hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
assert len(set(hashes.values())) == 7, 'Every round must preserve a distinct revision'
assert (ROOT / 'draft_06.md').read_bytes() == paper_path.read_bytes()
reviews = (ROOT / 'REVIEWS.md').read_text()
assert len(re.findall(r'^## Round \d', reviews, re.M)) == 6
for source, expected in evidence['sources_sha256'].items():
    assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == expected, source

report = {
    'status': 'PASS',
    'scope': 'Editorial/numerical consistency only; not experimental-protocol validation',
    'datasets': 6,
    'result_tables': 2,
    'structural_model_rows': 18,
    'f1_dataset_rows': 6,
    'self_review_revision_rounds': 6,
    'full_matrix_only': True,
    'source_fingerprints_verified': len(evidence['sources_sha256']),
    'final_sha256': hashlib.sha256(paper_path.read_bytes()).hexdigest(),
    'original_pdf_sha256': hashlib.sha256((ROOT.parent / 'Rule_Learning___Huawei (1).pdf').read_bytes()).hexdigest(),
    'revision_sha256': hashes,
    'word_count_approx': len(paper.split()),
}
(ROOT / 'VALIDATION.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))

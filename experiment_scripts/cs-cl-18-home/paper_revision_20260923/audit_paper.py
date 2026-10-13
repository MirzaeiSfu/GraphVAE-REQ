"""Produce an editorial part ledger and validate preservation of paper evidence."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
paper_path = ROOT.parent / 'final.md'
paper = paper_path.read_text()
body, refs = paper.split('## References', 1)
blocks = re.split(r'\n\s*\n', body.strip())

# Human editorial judgements keyed to final manuscript parts, not automatic
# claims that a generic language check can establish scientific necessity.
decisions = {
4: ('Keep/rewrite', 'The abstract must state problem, full-matrix mechanism, comparators, evidence, and scope.', 'Define technical terms in the body; retain negative DeFoG comparisons rather than promise universal superiority.'),
6: ('Keep/tighten', 'Motivates the relational pattern problem and why probabilistic rather than hard rules are relevant.', 'A familiar example supplies intuition without a lengthy application survey.'),
7: ('Keep', 'Connects the original metapath motivation to interpretable rule statistics.', 'The author–paper example defines the otherwise unfamiliar metapath term.'),
8: ('Keep', 'States the proposed intervention and its endpoint-resolved character.', 'Formal entries and learning objective follow in Sections 3–4.'),
9: ('Rewrite', 'Identifies the actual backbone and names used in the result tables.', 'Define micro–macro on first use; remove references to the act of revising another paper.'),
10: ('Keep', 'Introduces a short contribution list.', 'No extra claim is needed here.'),
11: ('Merge', 'Separates objective, computation, and evidence into three distinct contributions.', 'Do not claim unrestricted FO counting or that any graph moment loss is novel.'),
13: ('Keep/tighten', 'Positions the work relative to semantic-loss and neuro-symbolic methods.', 'Distinguish distributional statistic reconstruction from negative-log-satisfaction loss.'),
14: ('Keep/tighten', 'Preserves the original statistical-relational context and rule-learning connections.', 'Clarify regularization rather than an asserted exact maximum-entropy equivalence.'),
15: ('Keep/tighten', 'Explains how the backbone and DeFoG differ and what extra statistic the method contributes.', 'Retain graph-level versus node-level distinction; do not discuss revision history.'),
16: ('Keep/tighten', 'Establishes the computation scope and reason for complementary evaluators.', 'Use citations without duplicating the same author-year callout.'),
19: ('Rewrite', 'Defines the graph objects, attributes, functors, literals, and logical variables.', 'Remove unused edge-set notation; distinguish H from adjacency and X.'),
20: ('Keep', 'Separates a template from the value-specific state that indexes the loss.', 'Define F_r before using it in a matrix.'),
21: ('Keep', 'The two-edge formula anchors later matrix examples and synthetic rule discussion.', 'Every variable refers to a logical argument rather than a latent variable.'),
22: ('Keep', 'Explains the formula and the role of negative/absent states.', 'Relate states to exceptions without falsely identifying an implication with one conjunction.'),
23: ('Rewrite', 'Defines the grounding domain required for both statistics and probability claims.', 'Distinguish fixed population restrictions from random attribute requirements; retain graph-local scope.'),
25: ('Keep', 'Defines endpoint-restricted grounding sets before the full-matrix equation.', 'Explain which variables are exposed.'),
26: ('Keep', 'This is the central observed full-matrix statistic.', 'Retain a precise indicator-sum definition rather than code/storage details.'),
27: ('Keep', 'Gives the equation a direct interpretation and covers unary patterns.', 'Do not introduce an unrequested alternative representation.'),
30: ('Rewrite', 'Defines the generator, latent variable, posterior/prior, parameters, and base objective.', 'Use prose for inherited reconstruction components rather than an unexplained five-term schematic equation.'),
31: ('Keep', 'Connects a selected rule state to its decoder statistic and target.', 'Point to the computation and exactness conditions before asserting expected-count meaning.'),
32: ('Rewrite', 'Explains precisely which moments are matched and what reconstruction supervision does not prove.', 'Conditional indicator-sum moments must not be conflated with prior population constraints.'),
34: ('Keep', 'Specifies the alignment, comparison index set, and minibatch used in the objective.', 'These affect the loss mathematically; padding/caching machinery does not belong here.'),
35: ('Keep', 'Defines the full-matrix residual entering the calibrated objective.', 'Normalize over entries and batch; do not replace this with a statistics textbook formula.'),
36: ('Keep', 'Motivates residual scale calibration across differently sized statistics.', 'Avoid claiming complete scale invariance.'),
37: ('Simplify', 'Gives the essential Gaussian objective and residual-dependent scale.', 'Move the machine-scale clamp to the supplement; preserve its algebraic relation to the original calibration.'),
38: ('Rewrite', 'Defines the scale floor, calibration source, and normalization role.', 'Numerical constants and stabilization belong in reproducibility notes.'),
39: ('Keep', 'Connects the loss definition to the final optimization objective.', 'No implementation commentary is needed.'),
40: ('Keep', 'Specifies how rule supervision augments the backbone and how its weights are normalized.', 'This equation is essential for reproducing the method conceptually.'),
41: ('Rewrite', 'Defines lambda and rule weights and identifies the rule-disabled setting.', 'Replace undefined rule groups with general fixed normalized weights.'),
42: ('Keep/tighten', 'Explains gradient training and the absence of rule computation during generation.', 'Do not describe persistence, caching, launcher steps, or model file formats.'),
44: ('Rewrite', 'States the source-agnostic acquisition/selection interface and scientific importance of selected rules.', 'Exact support thresholds and unverified configuration details remain author-side reproducibility tasks.'),
45: ('Keep/clarify', 'Alignment is an essential property of entrywise matrix loss, not an optional coding detail.', 'Define PG and distinguish simultaneous relabelling from an independent permutation.'),
48: ('Rewrite', 'Defines relation matrices and categorical attribute indicators for the product.', 'Avoid comparing the entire feature row to an undefined scalar value.'),
49: ('Keep', 'Shows the central differentiable matrix computation.', 'Retain the grounding restrictions in the following explanation.'),
50: ('Keep', 'Explains internal-variable summation, endpoint constraints, and attribute softening.', 'Do not silently equate an unconstrained product with injective counting.'),
51: ('Compress', 'A short numerical example explains the product and its training signal.', 'Delete routine squared-error and derivative arithmetic.'),
53: ('Keep/clarify', 'Defines distinct underlying random variables and required values for probabilistic counting.', 'Explicitly handle repetition and contradictory categorical assignments.'),
54: ('Keep', 'States the conditional full-matrix statistic formally.', 'Exactness is conditional on the next proposition, not automatic for arbitrary matrix expressions.'),
55: ('Rewrite', 'States the scientific guarantee with its actual scope.', 'Add fixed node domain and grounding restrictions; preserve conditional independence.'),
56: ('Keep', 'The short proof supports the original paper\'s expected-count idea.', 'Clarify that independence across groundings is unnecessary.'),
57: ('Keep', 'A tied-edge counterexample prevents an incorrect general expected-count claim.', 'This is a model assumption, not disposable implementation detail.'),
58: ('Keep/tighten', 'Defines algorithmic scope and asymptotic cost.', 'Define n and L; avoid unmeasured runtime or arbitrary-FO scalability claims.'),
61: ('Rewrite', 'States the two empirical questions and why diverse structures are included.', 'Connect the table discussion to these questions.'),
62: ('Keep/trim', 'Each dataset needs its graph family, relevant attribute representation, and known evaluation size.', 'Do not invent missing split/filter parameters; retain those verification tasks separately.'),
63: ('Keep', 'Clarifies synthetic feature absence, trained pattern family, and dataset attribution.', 'Feature descriptions concern the used representation, not all fields in an upstream release.'),
64: ('Keep', 'Strength-based dataset selection limits generalization and must not be disguised.', 'Report all displayed metrics, including losses.'),
66: ('Keep/clarify', 'Defines comparators and one selected configuration per dataset.', 'Retain non-loss-only PTC comparison; do not call it a controlled causal ablation.'),
67: ('Keep/trim', 'Defines experimental replication and the order of evaluator/generator aggregation.', 'A prose statement of mean and sample SD is enough; no SD equation.'),
68: ('Keep/move details', 'Feature exclusion and graph preprocessing change what is being measured.', 'Keep their meaning in the paper; move routine parameters to the linked supplement.'),
70: ('Rewrite', 'Defines the descriptor panel and the previously unexplained orbit statistic.', 'Explain local/global coverage instead of adding another formula for a standard estimator.'),
71: ('Rewrite', 'Defines MMD direction and mean-edge error in clear prose.', 'Do not relabel edge-count calibration as rule accuracy.'),
73: ('Rewrite', 'Introduces the untrained GNN evaluator and distinct input symbol.', 'Define GIN and frozen parameters; do not imply a learned discriminator.'),
74: ('Keep', 'The user requested a formal random-GNN definition; this specifies it.', 'Architecture dimensions remain in the supplement.'),
75: ('Clarify', 'Defines layer representations, pointwise transformations, neighbourhoods, and readout.', 'State common frozen weights and reference-fitted standardization.'),
76: ('Keep/clarify', 'S and C are scientifically different evaluation inputs.', 'Define square-clustering and prohibit interpreting cross-mode scores as interchangeable.'),
77: ('Keep', 'Defines embedding samples and local radii before precision/recall.', 'Exclude the query itself; retain k=5 as part of the metric definition.'),
78: ('Keep', 'Formal precision/recall is necessary to meet the requested experimental design.', 'Use reference radii for precision and generated radii for recall.'),
79: ('Keep', 'Interprets the support conditions before presenting their summary.', 'Avoid conflating this with node classification.'),
80: ('Keep', 'F1 is central to Table 2 and explicitly requested.', 'Do not place a second near-identical numerical-stabilization formula in the main text.'),
81: ('Keep/trim', 'Provides the zero case, meaning, source, and aggregation pointer.', 'The exact small offset remains documented in the linked supplement.'),
83: ('Clarify', 'A table must identify the estimator, direction, and seed uncertainty.', 'Specify the first five columns are squared MMD.'),
84: ('Keep/check', 'The structural table supplies the paper\'s comparison evidence.', 'Verify every result cell is unchanged, including all losses.'),
85: ('Clarify', 'F1 caption must define S/C and GV/REQ and the percentage convention.', 'No derivation of percentages is necessary.'),
86: ('Keep/check', 'The F1 table supplies the primary distributional comparison.', 'Verify two GRID DeFoG seeds are not presented as three.'),
87: ('Rewrite', 'Answers the improvement-over-backbone question and identifies a counterexample.', 'Avoid saying every metric improves.'),
88: ('Rewrite', 'Answers the DeFoG question with structural/F1 trade-offs.', 'Retain the small LOBSTER margin relative to variability.'),
89: ('Keep/tighten', 'Summarizes remaining gaps and prevents one-metric superiority claims.', 'Bold means do not establish statistical significance.'),
91: ('Rewrite', 'Concludes with the mechanism and evidence, not revision-history commentary.', 'Do not promote the working comparison to a universal benchmark claim.'),
92: ('Keep/tighten', 'Consolidates material theoretical and experimental limitations and next research steps.', 'Unresolved experimental controls remain in the preserved author checklist.'),
}

assert len(blocks) == 92, len(blocks)
ledger = ['# Final passage-level necessity audit', '',
          'Each numbered unit is a paragraph, equation, list, or table in the final manuscript before the references. Each unit was assessed for necessity and missing context; the ledger records the resulting editorial decision. It is not a claim of independent peer review.', '',
          '| Part | Anchor | Decision | Why needed? | Complement or boundary |',
          '|---:|---|---|---|---|']
for i, block in enumerate(blocks, 1):
    if block.startswith('#'):
        verdict = ('Keep', 'Section navigation preserves the original scientific progression.', 'No additional explanation required.')
    elif block == 'Anonymous authors':
        verdict = ('Keep', 'Anonymous manuscript author metadata.', 'Do not imply actual conference submission.')
    else:
        assert i in decisions, (i, block[:90])
        verdict = decisions[i]
    anchor = block.splitlines()[0][:95].replace('|', '/').replace('$', '')
    if anchor == '$$'.replace('$', ''):
        anchor = 'Equation ' + re.search(r'\\tag\{([^}]+)\}', block).group(1)
    values = [str(i), anchor, *verdict]
    ledger.append('| ' + ' | '.join(v.replace('|', '/') for v in values) + ' |')

reference_entries = [l for l in refs.splitlines() if l.startswith('- ')]
for line in reference_entries:
    surname = line[2:].split(',')[0]
    assert surname in body, f'Uncited bibliography author: {surname}'
ledger += ['', f'All {len(reference_entries)} bibliography entries have an in-text first-author callout. Primary source checks from the previous revision are preserved; this check alone does not certify bibliographic accuracy.', '']
(ROOT / 'PASSAGE_AUDIT.md').write_text('\n'.join(ledger))

expected_tables = (ROOT.parent / 'paper_revision_20260922/tables.md').read_text().strip().split('\n\n')
assert len(expected_tables) == 2
assert all(t in paper for t in expected_tables), 'Numerical tables changed'
assert len(re.findall(r'^\|---', paper, re.M)) == 2
assert re.findall(r'\\tag\{([^}]+)\}', paper) == [str(i) for i in range(1, 11)]
assert paper.count('$$') == 20
assert paper.count('$') % 2 == 0
for pattern in [r'_CP', r'\bcach(?:e|ed|ing)\b', r'\\operatorname\{SD\}', r'\.ckpt', r'num_layers', r'total[ _-]count', r'\bTODO\b', r'\bTBD\b', r'this revision', r'original paper']:
    assert not re.search(pattern, paper, re.I), pattern
for target in re.findall(r'\]\(([^)]+)\)', paper):
    if not target.startswith('http'):
        assert (paper_path.parent / target).exists(), target

snapshots = {}
for i in range(6):
    name = f'draft_{i:02}.md'
    snapshots[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
assert len(set(snapshots.values())) == 6
assert (ROOT / 'draft_05.md').read_text() == paper
assert len(re.findall(r'^## Round \d', (ROOT / 'REVIEWS.md').read_text(), re.M)) == 5

def prose_words(text):
    text = text.split('## References')[0]
    return len('\n'.join(l for l in text.splitlines() if not l.startswith('|')).split())

previous = (ROOT / 'draft_00.md').read_text()
report = {
    'status': 'PASS',
    'scope': 'Editorial completeness and numerical preservation, not new experimental validation',
    'additional_review_rewrite_rounds': 5,
    'passages_audited': len(blocks),
    'unchanged_result_tables': 2,
    'datasets': 6,
    'equations': 10,
    'reference_entries_with_callouts': len(reference_entries),
    'words_before': len(previous.split()),
    'words_after': len(paper.split()),
    'main_text_words_excluding_tables_and_references_before': prose_words(previous),
    'main_text_words_excluding_tables_and_references_after': prose_words(paper),
    'final_sha256': hashlib.sha256(paper_path.read_bytes()).hexdigest(),
    'revision_sha256': snapshots,
}
(ROOT / 'VALIDATION.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))

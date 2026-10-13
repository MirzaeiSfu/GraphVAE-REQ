"""Extract full-only tables from frozen reports; never select individual seeds."""
from pathlib import Path
import re,json,hashlib
ROOT=Path(__file__).resolve().parent
REPORTS=ROOT/'sources/reports/datasets'
def first_table(ds,marker):
 text=(REPORTS/ds/'RESULTS.md').read_text()
 start=text.index(marker);lines=text[start:].splitlines();table=[];begun=False
 for line in lines:
  if line.startswith('|'):
   table.append([x.strip().replace('**','') for x in line.strip('|').split('|')]);begun=True
  elif begun:break
 return table[2:]
def num(cell):
 match=re.match(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',cell)
 return float(match.group()) if match else None
rows=[]
for ds in ('PTC','LOBSTER','GRID','TRIANGULAR_GRID','QM9'):
 for panel in ('Structural','GIN'):
  marker=('### Structural metrics' if panel=='Structural' else '### RandomGIN without node features') if ds=='QM9' else ('## Phase 1' if panel=='Structural' else '## Phase 2')
  for raw in first_table(ds,marker):
   metric=raw[0]
   if ds in ('GRID','TRIANGULAR_GRID'):f,t,d=raw[3],raw[7],raw[2]
   elif ds=='QM9':f,t,d=raw[3],raw[2],raw[4]
   else:f,t,d=raw[1],raw[3],raw[4]
   if 'Generated mean' in metric or 'Reference mean' in metric:continue
   direction='up' if metric in ('Precision','Recall','F1-PR') else 'down'
   v=[num(x) for x in (f,t,d)]
   if None in v:continue
   sign=1 if direction=='up' else -1
   wins=v[1]>max(v[0],v[2]) if direction=='up' else v[1]<min(v[0],v[2])
   imp=[None if b==0 else 100*sign*(v[1]-b)/abs(b) for b in (v[0],v[2])]
   rows.append(dict(dataset=ds,panel=panel,metric=metric,false=f,true=t,defog=d,direction=direction,wins_both=wins,improvement=imp,source=str(REPORTS/ds/'RESULTS.md')))
def table(selected):
 lines=['| Dataset / metric | GraphVAE | GraphVAE-REQ full | DeFoG | Gain vs GraphVAE | Gain vs DeFoG |','|---|---:|---:|---:|---:|---:|']
 for r in selected:
  arrow='↑' if r['direction']=='up' else '↓'
  changes=['N/A' if x is None else f'{x:+.2f}%' for x in r['improvement']]
  dataset_label='Triangular grid' if r['dataset']=='TRIANGULAR_GRID' else r['dataset']
  lines.append(f"| {dataset_label} / {r['metric']} {arrow} | {r['false']} | {r['true']} | {r['defog']} | "+' | '.join(changes)+' |')
 return '\n'.join(lines)
head=[r for r in rows if r['wins_both'] and ((r['dataset']=='PTC' and r['metric'] in ('Degree MMD','Clustering MMD','Orbit MMD','F1-PR','MMD-RBF')) or (r['dataset']=='LOBSTER' and r['metric'] in ('Diameter MMD','F1-PR')) or (r['dataset']=='QM9' and r['metric'] in ('Absolute error in mean edge count',)))]
explore=[r for r in rows if r['wins_both'] and r['dataset'] in ('GRID','TRIANGULAR_GRID') and r['metric'] in ('Degree MMD','Clustering MMD','F1-PR')]
assert len(head)==8 and len(explore)==6,(len(head),len(explore))
template=(ROOT/'DRAFT_GRAPHVAE_FULL.md').read_text()
assert '{{HIGHLIGHT_TABLE}}' in template
rendered=template.replace('{{HIGHLIGHT_TABLE}}',table(head)).replace('{{EXPLORATORY_TABLE}}',table(explore))
(ROOT/'PAPER_FIRST_DRAFT_GRAPHVAE_FULL.md').write_text(rendered)
appendix=['# Evidence appendix — full-matrix comparison snapshot','',
 'Only full-matrix motif=True values are included. No total-count model or motif-correlation metric appears in these extracted numerical panels. The source reports remain unchanged in sources/reports/. This appendix includes losses as well as wins. Values are copied at source precision; percentages are recalculated from displayed means, so final digits may differ from raw-precision reports.','',
 '## Qualification key','',
 '- PTC: three seeds per method, shared 70-graph reference, but latent dimensions/batch sizes differ. Not a motif-only causal ablation.',
 '- LOBSTER: three seeds, topology-derived GIN, link-correlation-enabled full campaign; complementary DeFoG wins retained.',
 '- GRID: three GraphVAE seeds versus two filtered DeFoG seeds (0,4); graph identities differ. Exploratory, not a final benchmark claim.',
 '- TRIANGULAR_GRID: three GraphVAE seeds versus filtered DeFoG seeds (0,1,3); graph identities differ. Exploratory, not a final benchmark claim.',
 '- QM9: three seeds each, common 512-graph reference; topology-control GIN differs from topology-derived synthetic/PTC GIN. GraphVAE epoch250, DeFoG selected epoch240.','',
 '## Other datasets','',
 'MUTAG, PROTEINS, AIDS and OGB are discussed in the manuscript/roadmap, but incomplete replacements, duplicate collections, or unresolved hyperparameter/protocol provenance prevent promoting their historical values as verified best-three-way headline evidence. The copied source reports document those campaigns; no missing result is converted to zero.','']
for ds in ('PTC','LOBSTER','GRID','TRIANGULAR_GRID','QM9'):
 appendix += [f'## {ds}','',f'Source: `sources/reports/datasets/{ds}/RESULTS.md`.','']
 for panel in ('Structural','GIN'):
  appendix += [f'### {panel}','',table([r for r in rows if r['dataset']==ds and r['panel']==panel]),'']
appendix += ['## Source and calculation audit','',
 'The machine-readable `TABLE_EVIDENCE.json` records every copied value, direction, strict-both-baseline win flag, calculated improvement, and source path. The highlighted tables are selected summaries, not aggregate superiority claims. A tie with one baseline does not count as beating both.','',
 'Method source inspected: `/local-scratch2/mirzaei/fb/GraphVAE-REQ/motif_counting/motif_loss_utils.py` and `motif_objective.py`. This current source is not by itself proof of what every historical checkpoint executed.']
(ROOT/'EVIDENCE_APPENDIX_FULL_ONLY.md').write_text('\n'.join(appendix)+'\n')
(ROOT/'TABLE_EVIDENCE.json').write_text(json.dumps(rows,indent=2)+'\n')
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in REPORTS.glob('*/RESULTS.md')}
(ROOT/'SOURCE_REPORT_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Extracted {len(rows)} metric rows; highlighted {len(head)} established-snapshot and {len(explore)} exploratory advantages.')

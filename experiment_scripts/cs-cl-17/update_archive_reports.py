"""Build archive indexes and LGD per-seed/aggregate tables from available files."""
import json,statistics,datetime,math
from pathlib import Path
root=Path('/local-scratch2/mirzaei/EXPERIMENT_ARCHIVE_20260921')
inventory=json.loads((root/'archive_inventory_20260921.json').read_text())
status={'MUTAG':'LGD seeds 0,1,2: completed training, evaluation launched in lab tmux.', 'PTC':'LGD seeds 0,1,2: completed training, evaluation launched in lab tmux.', 'AIDS':'LGD seeds 0,1,2: completed training on Solar. Transfer to lab then evaluation launched by transfer_aids_lgd_eval on cs-cl-18.', 'QM9':'LGD diffusion training still incomplete; LGD evaluation skipped. Existing GraphVAE/DeFoG results are being archived.', 'OGB':'No completed OGB LGD checkpoint found; LGD evaluation skipped. Available GraphVAE/DeFoG artifacts are being archived.'}
def flatten(obj,prefix=''):
    out={}
    for k,v in obj.items():
        key=prefix+k
        if isinstance(v,dict):out.update(flatten(v,key+'.'))
        elif isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v):out[key]=v
    return out
for ds in status:
    dest=root/ds;dest.mkdir(exist_ok=True,parents=True)
    rows=[x for x in inventory if x['dataset']==ds]
    lines=[f'# {ds}: results and experiment archive','',f'Updated {datetime.datetime.now().isoformat(timespec="seconds")}', '', f'Archive host: cs-cl-19. Root: `{dest}`.', '', status[ds], '', '## Transfer status', '', f'Tmux on system 19: `archive_{ds.lower()}_20260921`. Log: [transfer.log](manifests/transfer.log). `TRANSFER_COMPLETE` means the listed archive transfers succeeded; it does not mean running model training or LGD evaluation has completed.', '', 'Original files are copied and retained. Environments, tar archives, git objects, and selected redundant raw datasets inside code snapshots are excluded. Invalid/old campaigns retain their original names and are not pooled into a new aggregate.', '', '## Existing result reports', '', 'The linked historical reports preserve their dates, protocols, seeds, hyperparameters and original comparison tables. Results from different campaigns are not silently combined.']
    reports=[r for r in rows if r['source'].endswith('.md')]
    lines += [f'- [{Path(r["source"]).name}](reports/{Path(r["source"]).name})' for r in reports]
    # Include newer reports inside transferred campaign folders.
    for p in sorted((dest/'sources').glob('*/*/*.md')) if (dest/'sources').exists() else []:
        if 'report' in p.name.lower() or 'comparison' in p.name.lower():lines.append(f'- [{p.name}]({p.relative_to(dest)})')
    lines += ['', '## LGD evaluation', '', 'Each seed uses its completed diffusion checkpoint, original encoder and frozen split. Structural metrics and topology Random-GIN use the existing benchmark evaluator (10 initializations, evaluator seed 0). Native-node and node/edge-feature GIN are computed where native features exist. Motif TV uses retained multi-atom training states; counts are summed across graphs and normalized within each rule. Test and training references are reported separately. Training-reference TV uses the same generated sample set as test-reference TV; it is not a matched-collection-size replication.', '', 'Errors remain explicitly recorded in each metrics.json; absent results are not zeros.']
    metrics=[]
    for p in sorted((dest/'evaluations').glob('lgd_seed_*/metrics.json')):
        d=json.loads(p.read_text());seed=d.get('seed',d.get('training_seed'));metrics.append((seed,flatten(d)))
        lines.append(f'- Seed {seed}: [report]({p.parent.relative_to(dest)}/LGD_RESULTS.md), [metrics]({p.relative_to(dest)})')
    selected=sorted({k for _,m in metrics for k in m if k.startswith(('structural.','structural_mmd.','topology_random_gin.','feature_random_gin.','motif_tv.')) and not any(t in k for t in ('seed','count','repeats','time','dim'))})
    if metrics:
        lines += ['', '| Metric | '+ ' | '.join('Seed '+str(s) for s,_ in metrics)+' | Mean ± sample SD (available seeds) |','| --- |'+' --- |'*(len(metrics)+1)]
        for key in selected:
            values=[m[key] for _,m in metrics if key in m and math.isfinite(m[key])]
            if not values:
                continue
            agg=f'{statistics.mean(values):.8g} ± {statistics.stdev(values):.8g} (n={len(values)})' if len(values)>1 else f'{values[0]:.8g} (n=1; SD unavailable)'
            lines.append('| '+key+' | '+' | '.join(f'{m[key]:.8g}' if key in m else 'pending' for _,m in metrics)+' | '+agg+' |')
    else:lines.append('No completed LGD metric files have reached the archive yet.')
    lines += ['', '## Full source and destination inventory', '', '| Original host | Original artifact | Archive location |','| --- | --- | --- |']
    for r in rows:lines.append(f'| cs-cl-{r["host"]} | `{r["source"]}` | `{Path(r["destination"]).relative_to(dest)}` |')
    (dest/f'{ds}_RESULTS_AND_ARCHIVE.md').write_text('\n'.join(lines)+'\n')

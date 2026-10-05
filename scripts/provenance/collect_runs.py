import json, os, sys, glob, re, datetime, hashlib
A = sys.argv[1]; out = []
def mtime(p):
    try: return datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime('%Y-%m-%d %H:%M')
    except: return ''
for root, dirs, files in os.walk(A):
    dirs[:] = [d for d in dirs if d not in ('generated_graph_train','best_validation_mmd_model','wandb','__pycache__')]
    rel = os.path.relpath(root, A)
    if 'reproducibility.json' in files:
        try: d = json.load(open(os.path.join(root, 'reproducibility.json')))
        except Exception as e: d = {}
        c = str(d.get('git_commit','')); ok = re.fullmatch(r'[0-9a-f]{40}', c.strip()) is not None
        st = d.get('git_status_short') or []
        dirty = None if not ok else sum(1 for l in st if l.strip() and not l.startswith('<'))
        args = d.get('args', {}) or {}
        start = min([os.path.getmtime(os.path.join(root,f)) for f in files if f in ('reproducibility.json','run_config_used.yaml','train.log')] or [0])
        end = max([os.path.getmtime(os.path.join(root,f)) for f in files] or [0])
        out.append(dict(kind='graphvae', path=rel, commit=c.strip() if ok else '', describe=str(d.get('git_describe',''))[:60] if ok else '', dirty=dirty,
            label=d.get('run_label',''), command=d.get('command',''), config=d.get('config_path',''), dataset=args.get('dataset',''), seed=args.get('seed',''),
            motif=args.get('motif_loss',''), mode=args.get('motif_output_mode',''), alpha=args.get('alpha_motif_loss',''),
            argkeys=hashlib.sha1(','.join(sorted(args)).encode()).hexdigest()[:10], nargs=len(args), argnames=sorted(args),
            start=datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M') if start else '', end=datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M') if end else '',
            has_patch='git_diff.patch' in files))
    elif 'hydra.yaml' in files and os.path.basename(root) == '.hydra':
        out.append(dict(kind='hydra', path=rel, start=mtime(os.path.join(root,'hydra.yaml'))))
json.dump(out, open(sys.argv[2],'w'), indent=0)
from collections import Counter
print(len(out), Counter(o['kind'] for o in out))
g=[o for o in out if o['kind']=='graphvae']
print('graphvae with commit:', sum(1 for o in g if o['commit']), ' without:', sum(1 for o in g if not o['commit']))
print('distinct commits:', Counter(o['commit'][:9] for o in g if o['commit']))
print('distinct argkey fingerprints:', Counter(o['argkeys'] for o in g))

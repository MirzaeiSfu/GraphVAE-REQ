"""Sample archived synthetic LGD models; persist graphs before evaluation."""
import sys
import pickle
from pathlib import Path
import torch

ROOT = Path('/local-scratch2/mirzaei/LGD_3SEED_CAMPAIGN_20260920')
sys.path.insert(0, str(ROOT / 'repo'))
import sample_tu as s

dataset = sys.argv[1]
out = Path(sys.argv[2])
config = ROOT / 'results' / f'GraphVAEReq-diffusion-{dataset}_seed2' / 'config.yaml'
args = type('Args', (), {'cfg_file': str(config), 'opts': []})()
s.set_cfg(s.cfg)
s.cfg.set_new_allowed(True)
s.load_cfg(s.cfg, args)
s.cfg.run_dir = str(config.parent / '2')
s.cfg.train.batch_size = 1
s.cfg.accelerator = 'cuda:0'
s.seed_everything(2)
loaders = s.create_loader()
model = s.build_model()
epoch = s.load_ckpt(model, epoch=1999)
assert epoch == 2000, epoch
reference = s._reference_graphs(loaders[2])
train_reference = s._reference_graphs(loaders[0])
loss, generated = s._evaluate(loaders[0], model, max_graphs=len(reference))
assert len(generated) == len(reference)
out.mkdir(parents=True, exist_ok=True)
with (out / 'samples.pkl').open('wb') as f:
    pickle.dump(dict(reference=reference, train_reference=train_reference,
                     generated=generated, dataset=dataset, seed=2,
                     checkpoint=str(config.parent / '2/ckpt/1999.ckpt'),
                     sampling_loss=loss), f)
print('SAMPLING COMPLETE', len(generated), flush=True)

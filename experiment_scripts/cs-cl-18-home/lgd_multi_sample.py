"""Sample completed LGD checkpoints with their original categorical schema."""
import sys, json, pickle
from pathlib import Path
root, dataset, seed, output = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
sys.path.insert(0,str(root/'repo'))
import sample_tu as s
from lgd.asset.graphvae_req_export import _loader_records, _generated_records
config=root/'results'/f'GraphVAEReq-diffusion-{dataset}_seed{seed}'/'config.yaml'
assert (config.parent/str(seed)/'COMPLETE').exists(), 'Diffusion incomplete'
s.set_cfg(s.cfg); s.cfg.set_new_allowed(True)
s.load_cfg(s.cfg,type('Args',(),dict(cfg_file=str(config),opts=[]))())
s.cfg.run_dir=str(config.parent/str(seed))
s.cfg.dataset.dir=str(root/'repo/datasets/graphvae_req')
original=Path(s.cfg.diffusion.first_stage_config)
s.cfg.diffusion.first_stage_config=str(root/'results'/original.parts[-4]/original.parts[-3]/original.parts[-2]/original.name)
s.cfg.train.batch_size=1
s.cfg.accelerator='cuda:0'
s.seed_everything(seed)
loaders=s.create_loader(); model=s.build_model()
epoch=s.load_ckpt(model,epoch=-1)
assert epoch>0
references=_loader_records(loaders[2],'test')
training=_loader_records(loaders[0],'train')
_,generated=s._evaluate(loaders[0],model,max_graphs=len(references))
assert len(generated)==len(references)
output.mkdir(parents=True,exist_ok=True)
metadata=json.loads((root/'repo/datasets/graphvae_req'/dataset/'metadata.json').read_text())
with (output/'samples.pkl').open('wb') as f:
    pickle.dump(dict(dataset=dataset,seed=seed,checkpoint=str(config.parent/str(seed)/'ckpt'/f'{epoch-1}.ckpt'),metadata=metadata,reference_records=references,train_records=training,generated_records=_generated_records(generated),generated=generated),f)
print('SAMPLED',dataset,seed,len(generated),flush=True)

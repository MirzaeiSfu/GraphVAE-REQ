# GPU Inventory

Collected on 2026-06-23 with:

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits
```

Memory values reported by `nvidia-smi` are MiB; GiB below is MiB / 1024.

| Machine | SSH result | GPUs | Per-GPU VRAM | Total GPU VRAM |
| --- | --- | --- | --- | --- |
| `cs-cl-13.cmpt.sfu.ca` | OK | GPU 0: NVIDIA TITAN RTX | 24 GiB / 24576 MiB | 24 GiB |
| `cs-cl-09.cmpt.sfu.ca` | OK | GPU 0: NVIDIA GeForce GTX TITAN X<br>GPU 1: NVIDIA GeForce GTX TITAN X | 12 GiB / 12288 MiB each | 24 GiB |
| `cs-cl-26.cmpt.sfu.ca` | OK | GPU 0: NVIDIA GeForce GTX 1080 Ti<br>GPU 1: NVIDIA TITAN X (Pascal) | GPU 0: 11 GiB / 11264 MiB<br>GPU 1: 12 GiB / 12288 MiB | 23 GiB |
| `cs-cl-36.cmpt.sfu.ca` | OK | GPU 0: NVIDIA GeForce RTX 2080 | 8 GiB / 8192 MiB | 8 GiB |
| `cs-cl-16.cmpt.sfu.ca` | OK | GPU 0: Quadro RTX 4000 | 8 GiB / 8192 MiB | 8 GiB |
| `cs-cl-17.cmpt.sfu.ca` | OK | GPU 0: NVIDIA TITAN RTX<br>GPU 1: NVIDIA TITAN RTX | 24 GiB / 24576 MiB each | 48 GiB |
| `cs-cl-18.cmpt.sfu.ca` | OK | GPU 0: NVIDIA GeForce GTX 1080 Ti<br>GPU 1: NVIDIA GeForce GTX 1080 Ti | 11 GiB / 11264 MiB each | 22 GiB |
| `cs-cl-19.cmpt.sfu.ca` | OK | GPU 0: NVIDIA GeForce GTX 1080 Ti<br>GPU 1: NVIDIA GeForce GTX 1080 Ti | 11 GiB / 11264 MiB each | 22 GiB |

## Notes for Job Distribution

- Highest single-GPU VRAM: `cs-cl-13` and `cs-cl-17`, both with 24 GiB TITAN RTX GPUs.
- Best 2-GPU hosts by total VRAM: `cs-cl-17` at 48 GiB total, `cs-cl-09` at 24 GiB total, `cs-cl-26` at 23 GiB total, and `cs-cl-18` / `cs-cl-19` at 22 GiB total.

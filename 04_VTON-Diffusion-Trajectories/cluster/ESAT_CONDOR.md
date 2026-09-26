# Running the trajectory comparison on ESAT's HTCondor pool

Compares the denoising trajectories of **CatVTON**, **OOTDiffusion (HD)** and
**IDM-VTON** on the 5 garment pairs in `pairs.txt` (5 x 3 = 15 GPU runs),
then computes the per-axis onsets (pose via OpenPose, color, texture, print,
stability) and the cross-model comparison.

## TL;DR

```bash
ssh <netid>@ssh.esat.kuleuven.be
cd <somewhere with ~70GB free>          # see "Where to put it" below
git clone https://github.com/Michiel-Vanzeir/VTO_Research.git
cd VTO_Research/04_VTON-Diffusion-Trajectories

bash cluster/setup.sh     # one time: envs, model weights, dataset (~1h, mostly downloads)
bash cluster/submit.sh    # submits all runs + analysis as one DAG
condor_q -dag -nobatch    # watch
```

Results land in `outputs/<model>_<pair>/` (frames, `trajectory.mp4`,
`contact_sheet.png`, `metrics*.json/png`) and `outputs/_comparison/`
(`onset_summary.csv`, `onset_summary_masked.csv`, one plot per pair).
Pull them back with e.g.
`rsync -avz <netid>@ssh.esat.kuleuven.be:<path>/04_VTON-Diffusion-Trajectories/outputs/ outputs/`
(`--exclude 'frames_zt'` roughly halves the transfer).

## What the scripts do

| file | role |
|---|---|
| `setup.sh` | idempotent one-time setup on the login node: installs `uv` if missing, clones the 3 model repos at pinned commits into `repos/`, builds `.venv` (CatVTON + analysis), `.venv-ootd`, `.venv-idm` from `requirements/*.txt`, downloads all weights (HF cache in `.hf_cache/`) and the VITON-HD test split into `data/`, then import-checks everything. Rerun it after any failure; `bash cluster/setup.sh check` re-runs only the checks. |
| `env.sh` | all paths/caches, kept inside the project dir (small `$HOME` quota, and execute nodes see the same paths). |
| `pairs.txt` | the garment pairs; add a line to add a pair. |
| `submit.sh` | writes `generated/trajectories.dag` (one node per model x pair, plus `analyze`) and submits it. Skips runs that already finished, so resubmitting after a partial failure only redoes what's missing. `MODELS="idm" bash cluster/submit.sh` for a subset, `FORCE=1` to redo, `DRY_RUN=1` to only print the DAG. |
| `vton.sub` / `run_job.sh` | one trajectory run (+ its video/contact sheet). Jobs run with `HF_HUB_OFFLINE=1`, so execute nodes need no internet. |
| `analyze.sub` / `analyze.sh` | CPU-only final node: `metrics.py` (whole frame and mask crop) on every finished run, then `compare_trajectories.py`. A failed run doesn't block it; it's just skipped. Can also be run by hand: `bash cluster/analyze.sh`. |

## Resources per model

| model | GPU memory | RAM | why |
|---|---|---|---|
| CatVTON (512x384) | >= 7.5 GB | 16 GB | ran on an 8 GB laptop GPU |
| OOTDiffusion HD (768x1024) | >= 7.5 GB | 24 GB | ran on an 8 GB laptop GPU; laptop RAM (7.6 GB) was the problem |
| IDM-VTON (768x1024, SDXL) | >= 23 GB | 40 GB | two 2.6B-param UNets in fp16; the 12 GB fp32 UNet checkpoint is loaded through CPU RAM |

The GPU-memory floor is expressed in `vton.sub` as
`(CUDAGlobalMemoryMb >= X) || (GPUs_GlobalMemoryMb >= X)` so it works with
either GPU ClassAd naming. If the IDM jobs sit idle forever, check which
cards the pool has:

```bash
condor_status -af Machine CUDADeviceName CUDAGlobalMemoryMb | sort -u
condor_status -af Machine GPUs_DeviceName GPUs_GlobalMemoryMb | sort -u   # newer pools
```

and `condor_q -better-analyze <job id>` for why a job doesn't match.

## Where to put it

Everything (weights ~45 GB, envs ~15 GB, dataset ~3 GB) lives inside the
checkout, so clone it where you have ~70 GB of quota that is visible from
the execute nodes. Check `quota` / `df -h` after logging in; ESAT home
directories are usually small, and project storage is typically under
`/esat/<pool>/<netid>/` or similar (ask the ESAT helpdesk if unsure).
Don't clone onto node-local `/tmp`.

## Things that may need adjusting (not verifiable from outside ESAT)

- **CUDA driver.** Envs use torch 2.6 cu124 wheels (driver >= 525). If
  `nvidia-smi` in a job's `.out` shows an older driver, delete the `.venv*`
  dirs and rerun setup with
  `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 bash cluster/setup.sh envs check`.
- **Walltime attribute.** `+RequestWalltime = 14400` (4 h) is set because
  ESAT's pool asks for it; a 50-step run takes minutes, not hours.
- **Shared filesystem.** Submit files use `should_transfer_files = NO`
  (jobs read/write the checkout in place). If jobs go to *held* with a
  file-not-found reason, the checkout isn't on a filesystem the execute
  nodes mount; move it (see "Where to put it").

## Debugging a run

`cluster/logs/<model>_<pair>.{out,err,log}` per run, `logs/analyze.*` for
the analysis, `generated/trajectories.dag.dagman.out` for the DAG itself.
Each run's `.out` starts with the GPU name/driver and contains the
`[verify] ... -> OK/MISMATCH` line (scheduler's x0-hat vs the manual
formula at step 0) and per-step progress.

## Method notes

- All three models are sampled with DDIM, 50 steps, eta=1.0, seed 42, and
  each model's own upstream guidance scale and preprocessing (see the
  docstring at the top of each `scripts/*_trajectory.py`).
- IDM-VTON's loop is not reimplemented: its scheduler's `step()` is wrapped
  to capture z_t and x0-hat, and `final_pipeline.png` (the pipeline's own
  output) is saved next to `final.png` as a sanity check that both match.
- POSE = OpenPose PCK of each x0-hat frame's skeleton against the final
  frame's (see `scripts/pose.py`). For inpainting try-on the skeleton is
  usually already right at step 0, since the unmasked body pins it; PCK
  varying by < 0.1 over a run is reported as onset 0 with
  `pose_saturated=True` in the CSV, and `pose_refine_onset` (keypoint error)
  tracks the finer settling of joint positions.

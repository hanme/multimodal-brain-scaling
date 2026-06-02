# Setup Guide

Getting the environment running on the EPFL SCITAS cluster (Jed/Kuma).

---

## 1. One-time: SSH key for GitHub

Git operations on the cluster must use SSH (not HTTPS — the VSCode credential helper
blocks HTTPS). Do this once per cluster account.

```bash
# Check if you already have a key
ls ~/.ssh/id_ed25519.pub

# If not, generate one (press Enter for all prompts)
ssh-keygen -t ed25519 -C "your.email@epfl.ch"

# Print the public key and copy it
cat ~/.ssh/id_ed25519.pub
```

Go to **github.com → Settings → SSH and GPG keys → New SSH key**, paste the key, save.

Then tell git to always use SSH for GitHub (avoids HTTPS credential prompts):

```bash
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

---

## 2. Clone the repository

```bash
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/
git clone git@github.com:hanme/multimodal-brain-scaling.git
cd multimodal-brain-scaling
```

---

## 3. Load Python 3.11 and install uv

The cluster default Python is too old. Load the module:

```bash
module load gcc/13.2.0 python/3.11.7
```

Install `uv` (fast package manager, replacement for pip+venv):

```bash
pip install uv
```

`uv` lands at `~/.local/bin/uv` which is already on your PATH.

---

## 4. Create the environment

Run once from inside the repo. This creates `.venv/` in the project directory
(all packages stay on `/work/`, nothing goes to `/home/` except a download cache).

```bash
UV_CACHE_DIR=/work/upschrimpf1/mehrer/.cache/uv \
uv sync --python python3.11 --extra evaluation --extra analysis --extra visualization --extra dev --extra audio
```

This installs ~206 packages including PyTorch, scikit-learn, transformers, etc.
Takes 5–10 minutes on first run (downloads ~2 GB). Subsequent runs are fast.

---

## 5. Activate the environment

```bash
module load gcc/13.2.0 python/3.11.7
source /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/.venv/bin/activate
```

You need to run both lines every time you open a new terminal or submit a SLURM job.

**Convenience:** a small activation script is provided at the repo root:

```bash
source env.sh
```

---

## 6. Verify

```bash
mbs-extract-features --help
```

You should see the full argument list. If that works, the environment is ready.

---

## Every subsequent session

```bash
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
source env.sh
```

---

## Updating the environment

If `pyproject.toml` changes (new packages added):

```bash
UV_CACHE_DIR=/work/upschrimpf1/mehrer/.cache/uv uv sync
```

`uv sync` always brings the environment in line with `pyproject.toml` — it adds missing
packages and removes ones that were removed. No need to recreate `.venv` from scratch.

---

## Downloading EEG datasets

OpenNeuro datasets are mirrored on AWS S3 as plain files. Use `aws s3 sync` — no datalad,
no symlinks, real files in the normal BIDS folder structure.

Install the AWS CLI into the conda env (one-time — `aws` is not available system-wide):

```bash
conda activate /work/upschrimpf1/mehrer/code/20251106_fMRI_movie_watching_neuromod_friends/fMRI_movie_watching_neuromod_friends_conda_env
pip install awscli
```

Then download ds004408 (~14 GB, naturalistic speech EEG):

```bash
aws s3 sync \
  s3://openneuro.org/ds004408 \
  /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/ds004408/ \
  --no-sign-request
```

`--no-sign-request` is required — the bucket is public and needs no AWS credentials.
This downloads the full BIDS structure as real files (no git-annex, no symlinks).

To get only the stimuli and skip raw EEG (useful for a quick check of what's there):

```bash
aws s3 sync \
  s3://openneuro.org/ds004408/stimuli \
  /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/ds004408/stimuli/ \
  --no-sign-request
```

After the download, switch back to the mbs environment for all pipeline steps:

```bash
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
source env.sh
```

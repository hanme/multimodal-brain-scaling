#!/bin/bash
# =============================================================================
# GPU (Kuma L40S): D2 attention encoder, BOTH target levels (parcels + electrodes).
# The §16/§20 redo — MSE loss, EEG z-scored + scaling stored in the checkpoint, MIRAGE-style
# checkpoint selection on a train-carved validation split. Body lives in run_probe_d2_levels.sh.
#
# SUBMIT FROM THE KUMA (GPU) CLUSTER:
#   sbatch scripts/kuma_probe_d2_levels.sh                         # whisper-small (default)
#   MODEL_ID=whisper-base sbatch scripts/kuma_probe_d2_levels.sh   # any whisper size
# =============================================================================
#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=probe_d2_lvls
#SBATCH --partition l40s
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --time 03:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/probe_d2_lvls_%j.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/probe_d2_lvls_%j.error

set -euo pipefail
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
echo "STARTING AT $(date)"
source env.sh
mkdir -p logs
nvidia-smi

python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "🛑 CUDA unavailable. Fix the cu126 torch build first (see handover ⚠️ ENV NOTE), then resubmit:"
  echo "   uv pip install 'torch==2.12.0+cu126' 'torchvision==0.27.0+cu126' --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match --reinstall-package torch --reinstall-package torchvision"
  exit 1
fi

bash scripts/run_probe_d2_levels.sh

#!/bin/bash
# =============================================================================
# Learned temporal probe (Workstream B) on D2 + D3 — GPU (Kuma L40S). Plan §13 Q2.
# SUBMIT FROM THE KUMA (GPU) CLUSTER. /work is shared with jed, so all inputs are already
# in place (D2/D3 neural h5s + features, D3 dir built by the mTRF job).
#
#   sbatch scripts/kuma_probe_d2d3.sh                            # whisper-base (default)
#   MODEL_ID=whisper-small sbatch scripts/kuma_probe_d2d3.sh     # any whisper size (env propagates)
#
# Kuma rules (SCITAS docs): NO default partition -> must set --partition (h100|l40s|mig*);
# RAM is auto-assigned at 5900 MB/core (you cannot request more, so no --mem line). l40s allows
# up to 8 cores/GPU. We use l40s to match the earlier probe run (cu126 torch is for L40S/CUDA12.6).
# =============================================================================
#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=probe_d2d3
#SBATCH --partition l40s
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --time 02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/probe_d2d3_%j.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/probe_d2d3_%j.error

set -euo pipefail
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
echo "STARTING AT $(date)"
source env.sh
mkdir -p logs
nvidia-smi

# CUDA sanity (handover ENV NOTE: venv had a +cu130 build vs the L40S 12.6 driver -> cuda False).
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "🛑 CUDA unavailable. Fix the cu126 torch build first (see handover ⚠️ ENV NOTE), then resubmit:"
  echo "   uv pip install 'torch==2.12.0+cu126' 'torchvision==0.27.0+cu126' --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match --reinstall-package torch --reinstall-package torchvision"
  exit 1
fi

bash scripts/run_probe_d2d3.sh

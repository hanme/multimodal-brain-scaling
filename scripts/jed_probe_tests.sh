#!/bin/bash
# =============================================================================
# CPU (Jed): full attention-encoder test suite. EVERY test pins device="cpu", so the whole
# suite — fast + the two slow learning/selection tests — runs here, not on the GPU. Use a CPU
# compute node (NOT the login node, which is CPU-starved and times the slow tests out).
#
#   sbatch scripts/jed_probe_tests.sh
# =============================================================================
#SBATCH --chdir /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
#SBATCH --job-name=probe_tests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6900M
#SBATCH --time=01:00:00
#SBATCH --output=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/probe_tests_%j.out
#SBATCH --error=/work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling/logs/probe_tests_%j.err

set -euo pipefail
cd /work/upschrimpf1/mehrer/code/20260601_multimodal_brain_scaling_schizophrenia/multimodal-brain-scaling
echo "STARTING AT $(date)"
source env.sh
mkdir -p logs
echo "Python: $(which python)"

# -m "" runs everything incl. @pytest.mark.slow; -p no:cacheprovider keeps a read-only FS happy.
.venv/bin/python -m pytest tests/test_attn_probe_temporal.py -v -p no:cacheprovider

echo "DONE AT $(date)"

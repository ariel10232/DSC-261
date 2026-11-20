#!/bin/bash
# DSMLP background job: install a PyTorch build that supports the node's GPU, run the quick
# smoke test once, then the full (resumable) study. Re-submitting the job continues where it stopped.
# Submit from dsmlp-login with, e.g.:
#   launch.sh -W <COURSE> -i ghcr.io/ucsd-ets/scipy-ml-notebook:2025.1-stable -c 8 -m 16 -g 1 \
#             -l gpu-class=medium -N dsc261-run -B -- bash -l $HOME/private/dsc261-final/dsmlp_job.sh
set -euo pipefail
cd "$HOME/private/dsc261-final" 2>/dev/null || cd "$HOME/dsc261-final"
exec > >(tee -a job.log) 2>&1
echo "== $(date) on $(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv || true

# The image ships torch 2.2 (no Blackwell / sm_120 kernels); install 2.8 + CUDA 12.8 to the user site once.
if ! python -c "import sys, torch; sys.exit(0 if torch.__version__.startswith('2.8') else 1)" 2>/dev/null; then
  echo "== installing torch 2.8.0 (cu128)"
  pip install --user -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
fi
python - <<'EOF'
import torch, torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__,
      "|", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
x = torch.randn(512, 512, device="cuda")
print("cuda matmul ok:", float((x @ x).abs().sum()) > 0)
EOF

if [ ! -f runs/quick/all_results.csv ]; then
  echo "== quick smoke test"
  python run_experiments.py --quick
fi
echo "== full study"
python run_experiments.py --exp all
python make_report_assets.py --run runs/full --dest report_assets || true
echo "== finished $(date)"

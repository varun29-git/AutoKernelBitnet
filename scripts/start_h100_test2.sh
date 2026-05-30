#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_FILE="${MODEL_FILE:-models/test2_bitnet.py}"
CLASS_NAME="${CLASS_NAME:-BitNetLlama}"
INPUT_SHAPE="${INPUT_SHAPE:-1,2048}"
DTYPE="${DTYPE:-bfloat16}"
TOP="${TOP:-3}"
BACKEND="${BACKEND:-triton}"
TOTAL_BUDGET_MINUTES="${TOTAL_BUDGET_MINUTES:-60}"

echo "== AutoKernel Test 2 H100 start =="
echo "repo:        $(pwd)"
echo "model:       ${MODEL_FILE}"
echo "class:       ${CLASS_NAME}"
echo "shape:       ${INPUT_SHAPE}"
echo "dtype:       ${DTYPE}"
echo "top kernels: ${TOP}"
echo "backend:     ${BACKEND}"
echo "budget:      ${TOTAL_BUDGET_MINUTES} minutes total"
echo

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Run this on the cloud GPU VM."
  exit 1
fi

nvidia-smi -L
echo

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed. Install it first:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo "== Sync dependencies =="
uv sync --extra cuda --extra models

echo
echo "== Python/CUDA sanity =="
uv run python - <<'PY'
import importlib.util
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside this uv environment")
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("triton installed:", importlib.util.find_spec("triton") is not None)
PY

echo
echo "== Local scaffold tests =="
uv run python -m unittest discover -s tests -v

echo
echo "== Prepare benchmark cache =="
uv run prepare.py 2>&1 | tee prepare.log

echo
echo "== Profile Test 2 BitNet model =="
uv run profile.py \
  --model "${MODEL_FILE}" \
  --class-name "${CLASS_NAME}" \
  --input-shape "${INPUT_SHAPE}" \
  --dtype "${DTYPE}" \
  2>&1 | tee profile.log

echo
echo "== Extract top bottleneck kernels =="
uv run extract.py --top "${TOP}" --backend "${BACKEND}" 2>&1 | tee extract.log

echo
echo "== Benchmark current kernel smoke =="
uv run bench.py --quick 2>&1 | tee run.log

echo
echo "== Orchestration plan =="
uv run orchestrate.py plan
uv run orchestrate.py next

cat <<'EOF'

== Ready for the coding agent ==
From this same repo directory on the H100 VM, start Codex or your coding agent
and give it:

  Read program.md. We are optimizing the Test 2 BitNet model on this H100.
  Total GPU budget is 1 hour. Use orchestrate.py to choose the next kernel,
  edit only kernel.py, run bench.py, keep improvements, revert failures, and
  continue the AutoKernel loop. Favor quick high-impact changes over broad
  exploration.

If you want the agent to use an API key, set it in this terminal before starting
the agent. Do not paste it into source files or notebook cells.
EOF

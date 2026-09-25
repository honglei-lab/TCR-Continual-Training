#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [ -e .venv ]; then
  echo 'Refusing to replace an existing .venv. Inspect it or choose a fresh clone.' >&2
  exit 1
fi
command -v uv >/dev/null || { echo 'Install uv first (https://docs.astral.sh/uv/).' >&2; exit 1; }
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -c requirements-constraints.txt \
  -e './vendor/lerobot[training,pi,peft-dep,evaluation,libero]' \
  modelscope==1.39.1 pytest
.venv/bin/python scripts/doctor.py

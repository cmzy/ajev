#!/usr/bin/env bash
# 检查本机推理结果与 GPU 上的正式评测结果是否一致（默认 40 道 JevBench 题，约几分钟）。
# 用法：./verify.sh          ./verify.sh --n 231（全部）
set -euo pipefail
cd "$(dirname "$0")"
[[ -x .venv/bin/python ]] || { echo "请先运行 ./install.sh"; exit 1; }
export PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1
exec .venv/bin/python verify.py "$@"

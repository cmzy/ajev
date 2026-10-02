#!/usr/bin/env bash
# AJev 一键安装：Python 环境 + 依赖 + 基座模型（google/gemma-4-12B-it，约 24 GB）。
# 用法：./install.sh            （在 Apple 芯片的 Mac 上）
#       ./install.sh --force    （在其他机器上也继续安装，例如 Intel Mac 或 Linux，只用于测试）
set -euo pipefail
cd "$(dirname "$0")"
BASE_MODEL="${AJEV_BASE_MODEL:-google/gemma-4-12B-it}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# 第 1 步：检查机器。12B 模型 bf16 权重约 24 GB，建议 32 GB 以上内存（48 GB 最佳）。
say "检查机器"
os=$(uname -s); arch=$(uname -m)
mem_gb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
echo "系统 $os / $arch，内存 ${mem_gb} GB"
if [[ "$os" != "Darwin" || "$arch" != "arm64" ]] && [[ "${1:-}" != "--force" ]]; then
  echo "这个安装脚本面向 Apple 芯片的 Mac（M1–M4）。在其他机器上测试请加 --force。"; exit 1
fi
if [[ "$os" == "Darwin" && "$mem_gb" -lt 32 ]]; then
  echo "警告：内存小于 32 GB，加载 12B 模型很可能失败。"
fi
free_gb=$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')
echo "当前磁盘剩余 ${free_gb} GB（基座模型约 24 GB，Python 环境约 2 GB）"

# 第 2 步：安装 uv（一个很快的 Python 包管理器，会顺便下载合适版本的 Python，不影响系统自带的 Python）。
say "准备 Python 环境"
if ! command -v uv >/dev/null 2>&1; then
  echo "安装 uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 第 3 步：检查 PyTorch 能否使用 Apple 芯片的 GPU（MPS）。
.venv/bin/python - <<'PY'
import torch
mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
print(f"torch {torch.__version__}，MPS（Apple GPU）可用：{mps}")
PY

# 第 4 步：下载基座模型到 Hugging Face 缓存（~/.cache/huggingface）。下载中断可以重新运行本脚本，会接着下。
say "下载基座模型 $BASE_MODEL（约 24 GB，第一次需要一些时间）"
if [[ -d "$BASE_MODEL" ]]; then
  echo "使用本地目录 $BASE_MODEL，跳过下载。"
elif ! .venv/bin/hf download "$BASE_MODEL" --exclude "*.gguf" >/dev/null; then
  echo "下载失败。如果提示需要登录或授权：先运行  .venv/bin/hf auth login  ，再重新运行 ./install.sh"; exit 1
fi
echo "基座模型已就绪。"

# 第 5 步：检查 LoRA 适配器。
if [[ ! -f models/ajev-lora2/adapter_model.safetensors ]]; then
  echo "缺少 models/ajev-lora2/adapter_model.safetensors（LoRA 适配器），请确认安装包完整。"; exit 1
fi
say "安装完成。运行 ./run.sh 启动服务并打开 Playground；运行 ./verify.sh 检查推理结果是否正确。"

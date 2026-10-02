#!/usr/bin/env bash
# 启动 AJev 服务并在浏览器打开 Playground。按 Ctrl+C 停止。
# 用法：./run.sh                 默认端口 8000，只接受本机访问
#       PORT=9000 ./run.sh       换端口
#       ./run.sh --host 0.0.0.0  允许局域网内其他机器访问（其他参数也会原样传给 server.py）
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
[[ -x .venv/bin/python ]] || { echo "请先运行 ./install.sh"; exit 1; }
export PYTORCH_ENABLE_MPS_FALLBACK=1   # 个别 MPS 不支持的算子自动退回 CPU，而不是报错
export HF_HUB_OFFLINE=1                # 基座模型已经下载好，启动时不再联网检查
.venv/bin/python server.py --port "$PORT" "$@" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' INT TERM EXIT
echo "正在加载模型（第一次约 1–3 分钟）..."
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "服务启动失败，请查看上面的错误信息。"; exit 1; }
  sleep 2
done
echo "服务已就绪：http://127.0.0.1:$PORT/"
command -v open >/dev/null && open "http://127.0.0.1:$PORT/"
wait $SERVER

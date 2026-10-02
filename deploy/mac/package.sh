#!/usr/bin/env bash
# 在开发机上打包 Mac 部署工程：dist/ajev-mac/（可直接拷贝）+ dist/ajev-mac.tar.gz。
# 用法（在仓库根目录）：deploy/mac/package.sh [适配器目录，默认 runs/gemma_lora2/best]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
ADAPTER="${1:-runs/gemma_lora2/best}"
OUT=dist/ajev-mac
[[ -f "$ADAPTER/adapter_model.safetensors" ]] || { echo "找不到适配器：$ADAPTER"; exit 1; }
rm -rf "$OUT" && mkdir -p "$OUT/ajev/lm" "$OUT/models" "$OUT/verify"

# 1. 部署脚本、服务和网页
cp deploy/mac/{server.py,playground.html,install.sh,run.sh,verify.sh,verify.py,requirements.txt,README.md} "$OUT/"
# 2. 推理需要的 ajev 代码（与训练、评测完全相同的提示词和打分逻辑）
cp ajev/schema.py "$OUT/ajev/"
cp ajev/lm/prompt.py ajev/lm/predictor.py "$OUT/ajev/lm/"
touch "$OUT/ajev/__init__.py" "$OUT/ajev/lm/__init__.py"
# 3. LoRA 适配器（含校准温度）
cp -r "$ADAPTER" "$OUT/models/ajev-lora2"
# 4. 验证用的参考答案：JevBench 题目 + GPU 上正式评测得到的概率
python3 - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
ref = {json.loads(l)["id"]: l for l in open("runs/gemma_lora2/preds_jevbench_public_s16k.jsonl")}
rows = [l for l in open("data/external/jevbench_public.jsonl") if json.loads(l)["id"] in ref]
open(f"{out}/verify/jevbench.jsonl", "w").writelines(rows)
open(f"{out}/verify/reference_probs.jsonl", "w").writelines(ref[json.loads(l)["id"]] for l in rows)
print(f"verify: {len(rows)} JevBench decisions with reference probabilities")
PY
chmod +x "$OUT"/*.sh
COPYFILE_DISABLE=1 tar --no-xattrs -czf dist/ajev-mac.tar.gz -C dist ajev-mac 2>/dev/null || COPYFILE_DISABLE=1 tar -czf dist/ajev-mac.tar.gz -C dist ajev-mac
du -sh "$OUT" dist/ajev-mac.tar.gz

"""检查本机推理结果与 GPU（Colab G4）上正式评测的结果是否一致。

    ./verify.sh              # 40 道 JevBench 题
    ./verify.sh --n 231      # 全部 231 道

为什么要做这一步：同一个模型换了硬件（NVIDIA GPU → Apple GPU）、换了算子实现、又把 LoRA 合并进了权重，
bf16 下的数值会有细微差别。只要差别足够小，就说明部署是对的；如果差很多（比如 transformers 版本不对、
适配器没加载上、温度没读到），这里会直接报出来。

判定标准：
1. 最可能的选项一致：参考结果里第一名和第二名概率差不到 0.02 的“几乎打平”的题不计入（这种题的第一名
   换个硬件就可能互换，不代表出错）；其余的题一致率要达到 95% 以上；
2. 概率的平均绝对差小于 0.02。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40, help="number of reference decisions to check")
    ap.add_argument("--base-model", default=os.environ.get("AJEV_BASE_MODEL", "google/gemma-4-12B-it"))
    ap.add_argument("--adapter", default=os.path.join(HERE, "models", "ajev-lora2"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-merge", action="store_true")
    a = ap.parse_args()

    from ajev.lm.predictor import LMPredictor
    from ajev.schema import read_jsonl

    decisions = read_jsonl(os.path.join(HERE, "verify", "jevbench.jsonl"))[: a.n]
    with open(os.path.join(HERE, "verify", "reference_probs.jsonl"), encoding="utf-8") as f:
        ref = {r["id"]: r["probs"] for r in map(json.loads, f)}

    t0 = time.perf_counter()
    predictor = LMPredictor(a.base_model, adapter=a.adapter, device=a.device, batch_tokens=8192, merge=not a.no_merge)
    print(f"模型加载 {time.perf_counter() - t0:.1f} 秒，设备 {predictor.device}，校准温度 {predictor.temperatures}")

    t1 = time.perf_counter()
    probs = predictor.predict(decisions)
    secs = time.perf_counter() - t1
    print(f"推理 {len(decisions)} 道题 {secs:.1f} 秒（平均每题 {1000 * secs / len(decisions):.0f} ms）")

    agree = counted = 0
    diffs = []
    for d, p in zip(decisions, probs):
        r = ref[d.id]
        diffs += [abs(x - y) for x, y in zip(p, r)]
        top2 = sorted(r, reverse=True)[:2]
        if top2[0] - top2[1] < 0.02:  # 几乎打平，不计入
            continue
        counted += 1
        agree += max(range(len(p)), key=p.__getitem__) == max(range(len(r)), key=r.__getitem__)
    rate = agree / max(1, counted)
    mad = sum(diffs) / len(diffs)
    print(f"最可能选项一致：{agree}/{counted}（{rate:.1%}，已排除几乎打平的题）")
    print(f"概率平均绝对差 {mad:.4f}，最大差 {max(diffs):.4f}")
    ok = rate >= 0.95 and mad < 0.02
    print("结论：" + ("通过 ✅ 本机推理结果与 GPU 评测一致。" if ok else "未通过 ❌ 请检查 transformers / peft 版本和适配器文件。"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

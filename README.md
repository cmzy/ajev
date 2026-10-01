# AJev

自己训练的 Jev 类决策模型：输入一段材料（state）和若干带类型的问题，题型有三种：
- `noul`：是/否
- `choice`：单选
- `score`：打分

一次前向就输出校准过的概率分布。支持中文和英文。

技术调研与整体方案见 [`docs/01-技术调研与方案.md`](docs/01-技术调研与方案.md)。

## 目录结构

```
ajev/
  schema.py                Decision / Option 数据结构，Jev 请求 ↔ Decision 互转，JSONL 读写
  data/sources.py          公开 HF 数据集 → Decision（23 个数据源，中英文）
  data/typed_decisions.py  LocalLLaMA/typed-decisions 加载器（主测试集）
  data/build.py            构建 train / val / test JSONL 数据集
  metrics.py               准确率、扣除随机猜测的准确率、Brier、KL、ECE、打乱选项后答案改变的比例
  predictors.py            预测器接口 + 基线（均匀分布 / 随机 / 训练集答案频率）
  eval/evaluate.py         评测命令行
tests/                     单元测试
docs/                      调研与方案文档
```

## 使用方法

```bash
uv sync                                   # 本地环境（只负责数据和评测，不需要 torch）
uv run pytest -q

# 构建数据集（首次运行会从 HF Hub 下载约 1 GB）
uv run python -m ajev.data.build --out data/build
uv run python -m ajev.data.build --out data/smoke --train-cap 40 --eval-cap 20   # 小规模快速验证

# 在主测试集上跑基线
uv run python -m ajev.eval.evaluate --data data/build/test_typed.jsonl --predictor prior --train data/build/train.jsonl
```

构建产物：

| 文件 | 内容 |
|---|---|
| `train.jsonl` | 各公开数据集的 train split（每个源有采样上限，中文源上限 ×2）+ typed-decisions train |
| `val.jsonl` | 各公开数据集评测 split 的前 `--eval-cap` 条，用于选 checkpoint、拟合温度 |
| `test_public.jsonl` | 各公开数据集评测 split 的后 `--eval-cap` 条，用于分题型、分语言出报告 |
| `test_typed.jsonl` | typed-decisions test，共 2,000 道题（主指标） |
| `stats.json` | 按文件、数据源、题型、语言统计的数量 |

## 数据格式

每行一个 `Decision`：

```json
{"id": "tnews/12", "source": "tnews", "type": "choice", "lang": "zh",
 "state": "国足昨晚2比0战胜对手……", "instructions": "这篇文章属于哪个类别？",
 "options": [{"name": "体育", "desc": "体育赛事与运动员"}, {"name": "财经", "desc": "经济、金融、商业"}],
 "target": [1.0, 0.0], "group": "", "meta": {}}
```

- `noul` 的选项固定为 `true` / `false`，Jev 返回的 `noul` 值就是 P(`true`)。
- `score` 的选项是按顺序排列的等级 `"0".."K-1"`，永远不打乱。
- `target` 一律是概率分布：硬标签存成 one-hot，多人标注或教师模型给出的就是软标签。

## 对标成绩（typed-decisions test，2,000 道题）

| 系统 | 准确率 |
|---|---|
| 按训练集答案频率猜（本仓库基线） | 0.479 |
| AJev sft2（第一版数据，8.6 万题） | 0.7185 |
| AJev sft3（第二版数据，20 万题） | 0.7260 |
| **AJev sft4**（修复数据问题 + typed-decisions 训练题 ×3） | **0.7745** |
| TypeSafe Jev | 0.727 |
| Laya（421M） | 0.766 |
| Verdict 2.0（151M） | 0.771 |

## 部署到 MacBook（Apple 芯片，如 M4）

模型约 1.2 GB（3 亿参数，fp32），推理自动使用 Apple 芯片的 GPU（MPS）。

```bash
git clone git@github.com:cmzy/ajev.git && cd ajev
uv venv --python 3.12 && source .venv/bin/activate
# transformers 必须与训练环境一致（Colab 上是 5.16.1）：旧版 4.x 加载同一个模型，结果会明显不同
uv pip install -e ".[train,serve]" "transformers==5.16.1"

# 把训练好的模型目录拷到 runs/sft4/best（scp / AirDrop / U 盘）
# 一致性检查：本机预测与训练环境保存的预测逐题比较，通过才上线
python scripts/verify_deploy.py --checkpoint runs/sft4/best \
    --data data/build4/test_typed.jsonl --reference runs/sft4/preds_test_typed.jsonl --limit 200

# 启动服务（Jev 兼容接口）
python -m ajev.server --checkpoint runs/sft4/best --port 8000
curl localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "我的订单三天了还没发货"},
  "questions": {
    "category": {"type": "choice", "instructions": "这是什么问题？",
                 "criteria": {"delivery": "物流配送", "refund": "退款", "account": "账户"}},
    "urgent": {"type": "noul", "instructions": "需要马上人工处理。"}
  }
}'
```

一致性检查需要的 `data/build4/test_typed.jsonl` 和 `runs/sft4/preds_test_typed.jsonl` 不在 git 里，要和模型一起拷过去
（也可以用 `python -m ajev.data.build --out data/build4 --typed-repeat 3` 重新生成数据）。
实测参考：transformers 5.17 + CPU fp32 与 A100 bf16 的预测相比，概率最大差 0.008，最高票答案一致 98.5%
（不一致的都是前两名几乎打平的题）。

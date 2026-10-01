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
| TypeSafe Jev | 0.727 |
| Laya（421M） | 0.766 |
| Verdict 2.0（151M） | 0.771 |

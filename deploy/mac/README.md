# AJev Mac 部署包（Gemma 4 12B + LoRA）

在 Apple 芯片的 MacBook（M1–M4）上一键部署 AJev 决策模型。服务接口与 Jev 兼容，并自带一个网页 Playground。

| 项目 | 内容 |
|---|---|
| 模型 | `google/gemma-4-12B-it`（基座，约 24 GB）+ AJev LoRA 适配器 `ajev-lora2`（525 MB，含校准温度） |
| 题型 | 是/否（noul）、单选（choice）、打分（score），中英文都支持 |
| 输出 | 每个选项的概率，以及置信度（1 − 熵 / ln 选项数） |
| 推荐配置 | Apple 芯片，**48 GB 内存**（最少 32 GB），macOS 14 以上，磁盘空闲 30 GB 以上 |

## 三步使用

```bash
tar xzf ajev-mac.tar.gz && cd ajev-mac
./install.sh      # 第 1 步：安装 Python 环境和依赖，下载基座模型（约 24 GB，只需一次）
./run.sh          # 第 2 步：启动服务，自动在浏览器打开 Playground（http://127.0.0.1:8000/）
./verify.sh       # 可选：检查本机推理结果与 GPU 上的正式评测是否一致
```

- `install.sh` 会安装 [uv](https://docs.astral.sh/uv/)（Python 包管理器），在当前目录建 `.venv`，不影响系统的 Python。
  基座模型存放在 `~/.cache/huggingface`。下载中断时重新运行即可继续。
- `run.sh` 第一次加载模型需要 1–3 分钟。按 `Ctrl+C` 停止服务。

## Playground

浏览器打开 `http://127.0.0.1:8000/`：

1. 左边填写**材料**（纯文本或 JSON），添加**问题**（问题名、题型、问题描述、选项或等级）。
   右上角的“载入示例”里有中英文示例：客服工单、发票审核、安全告警。
2. 点“运行”（或按 ⌘+Enter），右边会显示每个问题各选项的概率条、最可能的答案和置信度，以及推理耗时和输入 token 数。
3. 底部可以展开原始请求和响应 JSON，以及对应的 curl 命令，直接复制到自己的程序里用。

## HTTP 接口（与 Jev 兼容）

```bash
curl http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "订单号 88231，耳机一周了还没发货，再不发货我就投诉！",
  "questions": {
    "category": {"type": "choice", "instructions": "这是什么问题？",
                 "criteria": {"delivery": "物流配送", "refund": "退款", "account": "账户"}},
    "urgent":   {"type": "noul", "instructions": "需要马上人工处理。"},
    "anger":    {"type": "score", "instructions": "客户有多生气？", "criteria": ["平静", "不满", "很生气"]}
  }
}'
```

响应：

```json
{"model": "ajev-lora2",
 "answers": {"category": {"choice": "delivery", "probabilities": {"delivery": 0.93, "refund": 0.05, "account": 0.02}, "confidence": 0.71},
             "urgent": {"noul": 0.81},
             "anger": {"score": 1.6, "legend": ["平静", "不满", "很生气"], "probabilities": [0.02, 0.36, 0.62], "confidence": 0.4}},
 "usage": {"input_tokens": 912, "output_tokens": 0},
 "latency_ms": 2850.3}
```

（上面的数字只是格式示意。）其他接口：`GET /health` 返回模型名、设备、校准温度。

## 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `PORT=9000 ./run.sh` | 8000 | 换端口 |
| `./run.sh --host 0.0.0.0` | 127.0.0.1 | 允许局域网内其他设备访问 |
| `./run.sh --max-state-tokens 8192` | 16384 | 材料超过这个长度（token）会被截断。设小一些更省内存、更快 |
| `./run.sh --batch-tokens 4096` | 8192 | 每次前向最多处理的 token 数。内存不够时调小 |
| `./run.sh --no-merge` | 合并 | 默认把 LoRA 合并进基座权重以加快推理；加这个参数则分开计算（更慢，结果几乎相同） |

## 性能预期（尚未在 M4 上实测）

- 模型以 bf16 精度运行在 Apple GPU（MPS）上，内存占用约 25–30 GB。
- 推理时间主要取决于材料长度：每个问题都要把“材料 + 问题”完整读一遍。
  按 M4 Pro / Max 的算力粗略估计，1,000 token 左右的材料每个问题约 1–3 秒，材料越长越慢。
  同一请求里的多个问题会尽量放在同一批里计算。
- 作为对比，在 Colab 的 RTX PRO 6000（G4）上，单个问题的延迟中位数是 64–137 ms。

## 常见问题

- **内存不够或很慢**：关掉其他占内存的程序；用 `--max-state-tokens 8192 --batch-tokens 4096` 启动。
- **下载失败**：检查网络；如果提示需要登录，运行 `.venv/bin/hf auth login` 后重新 `./install.sh`。
- **`verify.sh` 没通过**：通常是依赖版本不对。`requirements.txt` 固定了 `transformers==5.17.0`、`peft==0.21.0`，
  不要随意升级 transformers，Gemma 4 在不同大版本下的结果差别很大。

## 目录结构

```
ajev-mac/
├── install.sh / run.sh / verify.sh   一键安装、启动、验证
├── server.py                         推理服务（FastAPI）
├── playground.html                   网页 Playground
├── verify.py + verify/               验证脚本，以及 JevBench 题目和 GPU 上的参考结果
├── ajev/                             推理代码（与训练、评测用的同一套提示词和打分逻辑）
├── models/ajev-lora2/                LoRA 适配器 + 校准温度
└── requirements.txt
```

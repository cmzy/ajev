# 排行榜评测数据与训练集的重叠：data/lm3/train.jsonl

| 排行榜评测数据 | 条数 | 整段相同 | 共享 ≥60 字句子 | 命中的训练来源 | 示例 |
|---|---:|---:|---:|---|---|
| HellaSwag (validation) | 10042 | 0 | 1 | hellaswag(1) | [title] preheat the oven to 350 degrees fahrenheit (177 degrees celsius). |
| WinoGrande (dev) | 1267 | 0 | 0 |  |  |
| GSM8K (test) | 1319 | 0 | 0 |  |  |
| RAGTruth (test) | 2700 | 0 | 0 |  |  |
| ContractNLI (test) | 2091 | 0 | 646 | contractnli(1802), bev_skills(306), bev_hard(272), bev_default(85) | in witness whereof, the parties hereto have executed this agreement as of the da |
| Humicroedit (test) | 2960 | 0 | 0 |  |  |
| iSarcasmEval A-En (test) | 1423 | 0 | 0 |  |  |
| ACOS (test) | 1399 | 0 | 0 |  |  |
| New Yorker matching (test) | 528 | 0 | 0 |  |  |
| ANLI (test r1) | 1000 | 0 | 0 |  |  |
| ANLI (test r2) | 1000 | 0 | 0 |  |  |
| ANLI (test r3) | 1200 | 0 | 0 |  |  |
| BANKING77 (test) | 3076 | 0 | 0 |  |  |
| CLINC150+OOS (test) | 5500 | 0 | 0 |  |  |
| When2Call MCQ (test) | 3652 | 0 | 0 |  |  |
| PhishNChips | 加载失败：ValueError Config name is missing.
Please pick one among the available configs: ['emails',  | | | | |
| NLI4CT (tasksource test) | 加载失败：ValueError Unknown split "test". Should be one of ['train', 'validation']. | | | | |
| MMLU (test) | 14042 | 0 | 0 |  |  |
| MMLU-Pro (test) | 12032 | 0 | 0 |  |  |

# tools/ —— 数据工具

| 文件 | 作用 |
|---|---|
| `make_sample_data.py` | 从全量 `data/processed/` **确定性抽样**出学生包用的小样本 → `data/sample/`（seed 固定、保持负样本比例） |

```bash
python tools/make_sample_data.py                 # 生成 data/sample（800/300/1500/600/50）
python tools/make_sample_data.py --from_sample   # 不重新抽样，只校验样本并打印「下一步怎么跑」
python tools/make_sample_data.py --scale 0.5     # 更小的规模
```

**为什么需要它**：全量 processed 约 24MB、原始 zip 100MB+ 且有竞赛协议限制。
学生拿到课件包后如果第一步就是「去找数据下载」，很容易卡住直接放弃。
所以 `sample/` 保留一份确定性小样本，**几秒就能跑通全流程**；正式实验默认读 `data/processed/` 全量（已随包发）。

**抽样为什么是确定性的**：先按 `(id, sent_id)` 排序再 `shuffle(seed=42)`。
不排序直接 shuffle 的结果依赖原始文件行序 —— 换个版本的数据文件就抽到别的样本了，
PPT 上的数字也就不可复现。

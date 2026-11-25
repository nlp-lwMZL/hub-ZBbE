# rule/ —— L0 规则基线（零训练）

**一句话**：用「触发词词典 + 正则 + 股票简称词典」抽事件，不带任何模型。
它是**最诚实的对照组**：告诉我们「不用学模型能拿到多少」，后面 BERT 涨的点才有意义。

```bash
python rule/run_rule.py                 # 跑 dev 930 句，出指标 + 失败分析
python rule/run_rule.py --mock          # 用 data/_mock 的 20 条手造数据自检（几秒）
python rule/run_rule.py --min_trigger_count 3
```

| 产物 | 说明 |
|---|---|
| `outputs/metrics/disc/rule.json` | 指标 + `failure_analysis`（错误分类 + 典型例子） |
| `outputs/disc/pred_rule.jsonl` | 预测（含 id + sent_id，交给统一评测） |
| `outputs/logs/rule_eval.json` | 评测日志（用了多少触发词/词典、指标、失败分布） |
| `outputs/logs/rule_console.log` | 本次运行的完整控制台输出 |

**教学重点（实测数字见 `NOTES_discriminative.md`）**
1. 触发词从**训练集自动统计** + 一小张人工同义词表；匹配时**最长优先**（否则「质押」会抢走「解除质押」）。
2. 论元按角色名语义归类，用正则抽金额/日期/比例/数量；公司名用训练集里的「股票简称」建词典。
3. 失败分析把错误拆成互斥几类：`no_trigger` / `wrong_type` / `wrong_trigger` / `arg_missing` / `arg_spurious` / `exact`。
4. **规则的论元精确率很低** —— 它分不清「哪个金额属于哪个角色」，这正是后面必须上模型的原因。

# eval/ —— 汇总、图表、诊断工具

> 这一层**不做抽取**，只负责「把各方法的结果放在一起比」，以及一些诊断/分析脚本。

| 文件 | 作用 | 产物 |
|---|---|---|
| `compare_methods.py` | 读 `outputs/metrics/disc/*.json` 生成**对比表**（主表 + 消融，含溯源字段；缺的方法显式标"未跑"，绝不填估算值） | `outputs/metrics/disc/summary.md` |
| `compare_to_numbers.py` | **新旧数值对表**：把本次重跑结果逐项和历史台账（`_NUMBERS.md` 那组值）对比，**偏差 >0.5pp 自动标 ⚠** —— 这是"重构没改变数值行为"的验收工具 | 控制台表 + `outputs/logs/run_diff.out` |
| `make_figures.py` | 生成 6 张 PPT 用图（白底/中文字体/150dpi） | `outputs/figures/fig_*.png` |
| `check_bio_gold.py` | 量化**金标自身**的非法 BIO（重叠论元的地板值） | `outputs/metrics/disc/bio_gold_baseline.json` |
| `postprocess_trigger_copy.py` | 后处理规则：某角色与触发词"文本完全相等"时回填（纯 CPU，不重训） | `<base>_rule.json` |
| `plot_compare.py` | 把 LLM 链路的多个实验画成对比图（精度/延迟/token） | `outputs/figures/llm_*.png` |
| `run_experiments.py` | **一键复现**全部判别式实验（串行、单卡、带日志与耗时汇总）。**已改 fail-fast：任一步非 0 退出就立刻 `[FAIL]` 并中止后续**，避免"pipeline 崩了却继续跑下一步"掩盖故障 | `outputs/logs/run_<name>.out`、`run_summary.json` |

```bash
python eval/compare_methods.py --with_n200     # 汇总表（主表 + 消融 + n=200 列）
python eval/compare_to_numbers.py              # 与历史台账逐项对表（>0.5pp 标 ⚠）
python eval/make_figures.py                    # 6 张图
python eval/check_bio_gold.py                  # 金标非法地板值
python eval/run_experiments.py --group main    # 串行跑主线（跑前会检查显存；fail-fast）
python eval/plot_compare.py                    # LLM 链路对比图
```

**教学重点**
1. **汇总只从 JSON 读**，不手抄数字 —— 否则一定会出现「图注写 3 epoch、实际跑了 5 epoch」。
2. `compare_methods.py` 会**自动生成关键对比**（如「BERT 预训练带来多少点」），避免结论文字与实际数字脱节。
3. 缺结果就显式写「未跑」，**绝不补 0 或估算值**。
4. **方法清单里的名字必须和指标文件名一字不差**：本项目真实踩过——4 层消融实际文件叫
   `bert4L_linear_256`（linear 解码器，和主线 `bert_linear_256` 对照才干净），清单里却写了
   `bert4L_crf_256`，导致**整行在汇总表里静默消失**，只能靠人工发现。
   → 改清单后务必 `grep` 一遍 `outputs/metrics/disc/` 确认真有对应 `.json`。

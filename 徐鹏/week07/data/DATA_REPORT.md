# Week7「事件抽取」数据集报告：DuEE-Fin（金融公告 13 类）

> 结论先行：**week7 用 DuEE-Fin（句子级 13 类）**，域是上市公司公告/财经新闻，句子短（抽取子集中位 161 字），
> 13 个事件类型 × 92 个角色槽位（61 个唯一角色），带字符级 trigger/论元 span，处理后 **24.3MB**，可直接进课件包。
> 唯一需要老师拍板的是**许可**（2021 语言与智能技术竞赛参赛者协议：非商业科研/教学可用，但我们课程是收费课程，见 §12）。

---

## 1. 推荐结论（TL;DR）

| 用途 | 推荐文件 | 规模 |
|---|---|---|
| L0 规则（触发词召回 + 句子过滤） | `processed/train_detect.jsonl` / `dev_detect.jsonl` | 32,794 / 5,302 句，含 77.9% 负样本 |
| L1 传统 ML / L2 BERT-BIO(+CRF) / L4 LoRA SFT | `processed/train_extract.jsonl` / `dev_extract.jsonl` | 5,747 / 930 句（单事件），30,940 / 5,052 个 span |
| L3 LLM zero-shot | `dev_extract.jsonl` 分层抽 100 条（seed=42） | 成本可控 |
| L5 系统组合（篇章→句子→事件聚合） | `processed/dev_docs.jsonl` | 985 篇公告，1,493 个事件 |
| 事件类型/角色先验（prompt、schema 约束、规则枚举） | `processed/event_schema.json` | 13 类 / 92 个角色槽位（61 个唯一角色） |

---

## 2. 交付物清单

`data/`

| 文件 | 行数/条数 | 大小 | SHA256 前 16 位 |
|---|---|---|---|
| `processed/event_schema.json` | 13 类 + `__enums__` | 2.6 KB | `f782cb771f91fb6b` |
| `processed/train_extract.jsonl` | 5,747 | 5.24 MB | `08e04fecf5a36bd4` |
| `processed/dev_extract.jsonl` | 930 | 0.86 MB | `cc0d6a2dedc37ad3` |
| `processed/train_detect.jsonl` | 32,794 | 13.96 MB | `6fdbe9ae1a2e5797` |
| `processed/dev_detect.jsonl` | 5,302 | 2.24 MB | `83dfc4731ab04f0c` |
| `processed/dev_docs.jsonl` | 985 | 2.04 MB | `af65efdf152ff6b2` |
| **`sample/`（快速冒烟子集）** | **合计 3,250 条**（800/300/1500/600/50 + schema） | ≈ 3 MB | — |
| `sample/sample_stats.json` | 1 | — | 抽样元信息（seed=42，全量 vs 抽样行数，供 `--from_sample` 校验） |
| `figures/*.png` | 6 张 | 0.44 MB | — |
| **processed 合计** | — | **24.34 MB**（≤30MB 达标） | — |
| `raw/`（原始下载，供复现） | — | 37.1 MB | — |
| `stats/`（统计与校验 JSON） | 3 个 | — | — |
| `scripts/`（数据集构建与验收） | 5 个 `.py` + `README.md` | — | 见 §11 |
| `_mock/`（**冒烟用，6 个文件**） | 20 条句级 + 45 条检测 + 13 类 schema | ≈ 30 KB | 手造数据，给各方法 `--mock` 跑通流程；**指标无意义、别删** |
| `README.md` | — | — | 字段契约 + 规模速查 + 下载渠道/重建步骤 |

> **课件包随包发 `processed/`（完整派生数据，约 27 MB）与 `sample/`（3,250 条快速冒烟子集）**；原始发布文件 `raw/`（37 MB）**不发**——派生数据仍受原协议约束（非商业科研/教学）。
> `_mock/` 体积只有几十 KB 且是 `--mock` 冒烟的依赖，随代码一起分发。

`raw/` 内容：`DuEE_Fin_mirror/`（sentence train/dev、doc train/dev、`conf/DuEE-Fin/event_schema.json`、
`duee_dev.json/License.docx`、`duee_schema/duee事件类型及其角色.pdf`）+ `DuEE_hf/`（候选对比用，未参与下游）。

**数据下载渠道 + 重建步骤** 见 → **`data/README.md` §6**（官方千言 luge.ai 搜 DuEE-Fin，或 GitHub 镜像 `maopademiao/duee_fin`）。

---

## 3. 数据来源与许可

- **主数据**：DuEE-Fin 原始发布格式（13 类金融事件，篇章级 + 句子级）。官方渠道是千言（luge.ai），需注册登录，
  2026-11-16 本机 HEAD 请求 15s 超时不可达；改用 GitHub 公开镜像 **`maopademiao/duee_fin`**（main 分支，
  `DuEE-pytorch/data/DuEE-Fin/`）取到完整原始文件（含训练/验证划分与 schema），详见 §11 复现命令。
- **许可原文**：镜像内 `DuEE-pytorch/data/duee_dev.json/License.docx`（与 `duee_train.json/License.docx` 相同），
  即《2021 语言与智能技术竞赛参赛者协议》。关键条款（中英对照原文摘录）：
  - “参赛者仅有权出于**非商业科研或学术目的**使用数据集，不得出于任何商业目的使用数据集。”
    （*The Participant is entitled to use the Datasets only for the purposes of non-commercial research or teaching
    and shall not use the Datasets for any commercial purpose.*）
  - “我方对数据集中的内容……不享有任何知识产权，也不就上述内容做出任何保证。”（不承诺不侵权）
  - “在任何时候，我方均有权取消或终止授权……并有权要求参赛者删除数据集。”
  - “参赛者不得使用、修改、复制、公开传播、发行……前述信息”（未经书面约定）
- **判断**：作为**课堂教学/科研**用途明确被允许；但“八斗学院”是付费课程，且原协议**没有给再分发（打包发给学生）授权**。
  见 §12 风险 1 的处置建议。

---

## 4. 候选对比

| 候选 | 可得性 | 许可 | 规模 | 与投研场景 | 结论 |
|---|---|---|---|---|---|
| **DuEE-Fin**（GitHub 镜像） | 可（公开 zip 直下，实测 22.7s / 104MB） | 2021 竞赛参赛者协议：非商业科研/教学 | 篇章 train 7,047 / dev 1,174；句子级 32,794 / 5,302 | ★★★★★ 上市公司公告 13 类（质押/回购/减持/增持/收购/破产/上市/融资/中标/亏损/高管变动/被约谈/解除质押） | **主数据** |
| HF `nlhappy/DuEE`（=DuEE 1.0 句子级，早期验证过） | 可（HF 直连 200，parquet 2.2MB） | HF 页面标 MIT（上传者声明，非官方授权书） | 11,603 / 1,453 句，65 类 | ★★☆ 通用域；金融相关仅“财经/交易”9 类 996 事件 | 不用（一句带过：域泛、金融子集样本薄） |
| `FinanceMTEB/DuEE-fin` | 可 | 未标注 | 19,906 条语料（query/corpus/score） | ★★☆ 只有语料，无事件标注 | 仅语料备选 |
| 原版千言 DuEE-Fin（luge.ai） | 需登录，本机 15s 超时 | 同上 | 同映射像 | ★★★★★ | 未走通，用镜像替代 |

---

## 5. Schema 与 span 口径

### 5.1 统一 JSONL 字段

```json
{"id": "…", "sent_id": "…", "text": "中教控股收购海南学校",
 "event": {"type": "企业收购",
           "trigger": {"text": "收购", "start": 4, "end": 6},
           "args": [{"role": "收购方", "text": "中教控股", "start": 0, "end": 4}]}}
```

- `extract` 每行恰好 1 个事件（只取 `event_list` 长度 = 1 的句子）。
- `detect`：`{"id","sent_id","text","has_event","types","n_events"}`，全量句子（含负样本）。
- `dev_docs`：`{"id","title","text","events":[{"type","trigger":{…},"args":[…]}]}`，篇章级，`start/end` 在**整篇 text** 内。
- `event_schema.json`：`{"事件类型": ["角色1", ...], ...}`；`__enums__` 存放原 schema 的枚举：
  `{"公司上市.环节": ["筹备上市","暂停上市","正式上市","终止上市"]}`（键含 `.`，与事件类型区分；13 类中只有这 1 个枚举角色）。

### 5.2 换算依据（重要）

- 单位：**字符级 offset**，Python `str` 下标（中文每字 1 个下标），**左闭右开**，
  即 `text[start:end] == span.text`、`end = start + len(span.text)`。不涉及 token 级换算（无分词）。
- 生成方式：**不采信**原始 `trigger_start_index` / `argument_start_index`，一律在所属句子/篇章内用 `text.find()` 重新定位。
- 原始 index 审计（`scripts/audit_raw_index.py` → `stats/audit_raw.json`）：
  - train：9,440 事件 / 52,863 个 span；有 index 的 52,395 个**全部与 text 自洽**，
    且**与 `text.find()` 结果 0 例不一致** → 证明原 index 是“按首次出现回填”的，不是人工标注位置；
  - 无 index 的 468 个论元**全部是 `环节` 角色**（枚举值，如“筹备上市”，原文没有这串字）；
  - dev：1,523 事件 / 8,629 个 span，有 index 8,554 个全部自洽，无 index 75 个（同为 `环节`）。
- 对齐率（处理后，`scripts/verify_processed.py`）：

| 数据 | trigger 定位 | 论元定位 | 说明 |
|---|---|---|---|
| `train_extract` | 5,747 / 5,747 = **100%** | 25,193 / 25,540 = **98.64%** | 丢弃 347 个，全部是 `环节` |
| `dev_extract` | 930 / 930 = **100%** | 4,122 / 4,178 = **98.66%** | 丢弃 56 个，全部是 `环节` |
| `dev_docs` | 1,493 / 1,538 事件 = **97.07%**（丢 45 个事件） | 7,658 / 7,760 = **98.69%** | 丢 102 个，79 个是 `环节`，另 23 个为跨句/归一化论元 |
| 校验总计 | — | **45,143 个 span 重建 0 错误** | 逐行 `json.loads` + `text[start:end]==span.text` |

- 局限：同一字符串在句内多次出现时**取第一次出现**。`extract` 子集 30.6%（9,478 / 30,940）的 span 属于这种情况；
  原始全量句子中为 34.3%（18,123 / 52,863）。这是本数据集的已知精度上限，报告/PPT 里必须写明。

---

## 6. 关键统计数字

### 6.1 句子级（sentence/train|dev.json）

| 指标 | train | dev |
|---|---|---|
| 句子总数（`*_detect`） | 32,794 | 5,302 |
| 有事件句子（正样本） | 7,249（22.1%） | 1,179（22.2%） |
| 无事件句子（负样本） | 25,545（77.9%） | 4,123（77.8%） |
| 恰好 1 个事件 | 5,747（17.5%） | 930（17.5%） |
| ≥2 个事件（被排除） | 1,502（4.6%） | 249（4.7%） |
| 句长 p50 / p90 / p99 / max（字） | 102 / 227 / 363 / 1,206 | — |
| **extract 子集**句长 p50 / p90 / p99 / max | **161 / 265 / 393 / 797** | — |
| extract 超 510 字（BERT 512 会截断） | 11 句 | 2 句 |

- 触发器长度：p50 = 2 字，p99 = 4 字，max = 7 字（规则法可直接用事件类型→触发词表）。
- 每事件论元数：mean 4.38，p50 4，p90 7，p99 9，max 18。

### 6.2 篇章级（doc train/dev.json）

| 指标 | train | dev |
|---|---|---|
| 公告篇数 | 7,047 | 1,174 |
| 有事件篇数 | 6,150（87.3%） | 1,023（87.1%） |
| 无事件篇数 | 897 | 151 |
| 事件总数 | 9,513 | 1,538 |
| 每篇事件数 | 1 个 4,110（58.3%）、2 个 1,369、3 个 388、4 个 150、5 个 59、≥6 个 74 | — |
| 篇长 p50 / p90 / p99 / max / mean（字） | 506 / 661 / 2,696 / 3,502 / 500.3 | 输出 985 篇：500 / 650 / 2,682 / 3,485 / 487.2 |

> `dev_docs` 的偏差：doc dev 有事件 1,023 篇，其中 **38 篇所有事件的 trigger 在正文中都定位不到**（标题式/归一化 trigger，
> 如 trigger=“减持”但正文残缺，或 trigger=“正式上市”正文只写“上市”），整篇丢弃；另有 45 个事件被丢 → 保留 **985 篇 / 1,493 事件**。

### 6.3 事件类型分布（`train_extract` 单事件句子数）

| 类型 | 数 | 类型 | 数 | 类型 | 数 |
|---|---|---|---|---|---|
| 股份回购 | 979 | 企业融资 | 423 | 解除质押 | 255 |
| 企业收购 | 746 | 质押 | 398 | 企业破产 | 177 |
| 中标 | 706 | 公司上市 | 369 | 股东增持 | 162 |
| 亏损 | 648 | 高管变动 | 288 | 被约谈 | 53 |
| 股东减持 | 543 | — | — | — | — |

dev 侧同序：股份回购 178、中标 116、企业收购 116、亏损 106、股东减持 83、公司上市 58、质押 57、企业融资 53、
高管变动 48、解除质押 40、企业破产 33、股东增持 31、被约谈 11。**13 类在 train/dev 全覆盖，无缺类**。

### 6.4 角色（评卷要点）

- 61 个唯一角色（schema 92 个“类型×角色”槽位，如 `质押方`、`回购方`、`交易金额` 跨类型复用）。
- Top10（train_extract 出现次数）：披露时间 2,056、交易金额 1,308、每股交易价格 1,091、事件时间 926、中标公司 879、
  回购股份数量 866、回购方 848、回购完成时间 740、减持方 734、中标标的 714。
- 时间类角色（披露时间/事件时间/各类完成时间）+ 金额/数量类角色占比高 → 适合讲“公司/金额/日期/比例”四类槽位抽取。

### 6.5 重叠 / 重复 / 泄漏

| 项 | 数字 | 影响 |
|---|---|---|
| trigger span 与某论元 span 完全相同 | train 238 / dev 38 个事件（约 4.1%） | BIO 标注会冲突，需定优先级（如 trigger 优先） |
| 同一事件内任意两个 span 重叠 | train 568 / dev 97 个事件（约 9.9% / 10.4%） | 同上 |
| train 内重复句子文本 | 2,452 / 32,794（7.5%，多为“原标题：xxx”标题句重复） | 随机切分会有重复样本 |
| dev 内重复句子文本 | 287 / 5,302（5.4%） | — |
| train/dev 句子文本交集 | 201 句（占 dev 3.8%） | 句子级指标会轻微虚高，报告中注明 |
| 文档 id 在 train/dev 交集 | **0** | 篇章级划分干净（官方按文档划分） |
| `dev_docs` 中 title 原文出现在正文里 | 488 / 1,174（41.6%） | 篇章切句会切出重复标题句，需去重 |

---

## 7. 5 条跨类型示例（均来自 `train_extract.jsonl`，span 已按 §5.2 校验）

1. **公司上市**（`5f929eb7…`）：`[[*ST凯迪]]：公司股票可能被终止[[上市]]`
   trigger=`上市`(15,17)；args：上市公司=`*ST凯迪`(0,5)、环节=`终止上市`(13,17)——`环节` 恰好与 trigger 重叠。
2. **企业收购**（`9103d9fa…`）：`[[中教控股]]收购海南学校`
   trigger=`收购`(4,6)；args：收购方=`中教控股`(0,4)（标题式短句，论元不全，适合讲“论元缺失”）。
3. **亏损**（`0b809f8c…`）：`[[昆百大Ａ]]Q1亏损1.63亿元`
   trigger=`亏损`(6,8)；args：公司名称=`昆百大Ａ`(0,4)、净亏损=`1.63亿元`(8,14)。
4. **高管变动**（`c6fe4454…`）：`[[万钢]]担任暨南大学董事长`
   trigger=`担任`(2,4)；args：高管姓名=`万钢`(0,2)、变动类型=`担任`(2,4)（**与 trigger 同 span**）、变动后职位=`董事长`(8,11)。
5. **被约谈**（`a19317ea…`）：`[[南方能源监管局]]监管约谈广西水利电力建设集团有限公司`
   trigger=`约谈`(9,11)；args：约谈机构=`南方能源监管局`(0,7)、公司名称=`广西水利电力建设集团有限公司`(11,25)。

---

## 8. 数据坑（做实验前必须知道）

1. **原始 offset 不可信**：自带 `argument_start_index` 实为按 `text.find()` 首次出现回填（§5.2 审计：0 例与 find 不一致），
   且全量句子 34.3% 的 span 字符串在句内出现多次 → 个别论元可能指到错误位置。**处理方案**：交付文件统一用 `find()` 重算、
   首次出现为准；评测 gold 也用本文档 span，保证内部自洽（不要与论文数字直接对比）。
2. **论元归一化导致 span 缺失**：`环节` 是枚举角色（如“筹备上市”），原文没有这串字 → 1.36%/1.34% 论元被丢弃（train 347 / dev 56）；
   另有 `2290万` vs `2290万股` 这类“标注值 ≠ 原文子串”的情况（doc 级 23 个非 `环节` 丢弃项）。做 L1/L2 时这些槽位只能靠 schema 枚举补。
3. **单事件 vs 多事件**：`extract` 只保留单事件句（4.6% 的多事件句被排除）。L2/L4 做 span 抽取可以用；
   但 **L5 端到端系统必须用 `detect` + `dev_docs`**，否则讲不了“一篇公告多个事件”。
4. **负样本占比 77.9%**：句子级检测正负比约 1:3.5，直接训练会全预测负类；`detect` 文件保留全量负样本是为了量化 FP，
   课堂训练建议平衡采样（如负样本下采样到与正样本 1:1，seed 固定）。
5. **title 与正文重复**：41.6% 的 dev 文档 title 原样出现在正文首行（“原标题：…”），切句后会出现重复句；做 doc→句子切分时要按文本去重。
6. **train/dev 句子级有 201 句交集**（doc id 无交集，官方按文档划分）：句子级指标会略偏高，**报告实验数字时说明“按句子划分、存在 0.6% 文档内句子交叉”**。
7. **域与时代**：数据是 2019–2020 年的公告/新闻，标题式短句多（如“中教控股收购海南学校”），与真实长篇公告句式有差距；
   课堂讲“规则 vs BERT vs LLM”完全够用，产品化需用真实公告补样本。

---

## 9. 「哪个子集给哪条实验」结论

| 实验层 | 数据文件 | 用法 | 规模/备注 |
|---|---|---|---|
| L0 规则 | `train_detect` / `dev_detect` + `event_schema` | 触发词表（类型→常见触发词，p50 2 字）+ 正则/关键词召回 + 句子过滤；用负样本量化误报 | 32,794 / 5,302 |
| L1 传统 ML | `train_extract` / `dev_extract` | 触发词分类（13 类触发词）+ 角色分类 + 规则拼装 | 5,747 / 930 |
| L2 BERT-BIO(+CRF) | `train_extract` / `dev_extract` | 用 fast tokenizer 的 offset mapping 把字符 span 映射成 BIO；`max_len=512`（仅 11+2 句超长） | 30,940 span |
| L3 LLM zero-shot | `dev_extract` 分层抽 100 条（seed=42） | prompt 里给 13 类 + 角色清单 + enum；只跑 100 条控成本 | 约 100 条 |
| L4 LoRA SFT | `train_extract` → `dev_extract` | 1–2 epoch（Qwen3-0.6B/LoRA 够用），输出统一 JSON、按 span 对齐评测 | 5,747 → 930 |
| L5 系统组合 | `dev_docs`（+ `detect` 做句子过滤） | 公告切句 → 句子检测 → span 抽取 → 同文档事件聚合；用篇级 P/R/F1 评估 | 985 篇 / 1,493 事件 |

> 小样本冒烟：`train_extract` 按类型分层抽 500 条（seed=42）可在 1 分钟内跑完一条 L2/L4 链路；
> 完整实验用全量，报告里注明是子集还是全量。

---

## 10. 图表（`figures/`，中文标签、白底、150dpi，可直接贴 PPT）

| 文件 | 内容 | 关键数字 |
|---|---|---|
| `fig_event_type_distribution.png` | 13 类事件分布（单事件子集） | 979（股份回购）→ 53（被约谈） |
| `fig_docs_events_per_doc.png` | 每篇公告事件数分布 | 1 个 58.3%、2 个 19.4%、0 个 12.7% |
| `fig_role_top30.png` | 角色出现次数 Top30 | 披露时间 2,056、交易金额 1,308 |
| `fig_sentence_length.png` | 句子长度分布（含负样本） | p50 102、p90 227、p99 363 |
| `fig_doc_length.png` | 篇章长度分布 | p50 506、p90 661、p99 2,696 |
| `fig_args_per_event.png` | 每事件论元数分布 | 1–18 个，均值 4.38（5 个最多 1,173 句） |

---

## 11. 复现命令（在项目根目录执行，`python` = 你的解释器）

> 下面脚本都在 `data/scripts/` 下，**从项目根目录跑**（它们内部按 `Path(__file__).parents` 定位路径，不依赖 cwd 是 `data/`）。

```powershell
# 0) 目录（在项目根目录执行）
New-Item -ItemType Directory -Force data/raw,data/processed,data/figures,data/stats | Out-Null

# 1) 下载 DuEE-Fin 镜像（公开 GitHub 仓库 zip；本次 2026-11-16 从 main 分支取）
Invoke-WebRequest -Uri "https://codeload.github.com/maopademiao/duee_fin/zip/refs/heads/main" `
  -OutFile data/raw\duee_fin.zip -TimeoutSec 600
#    解压其中 6 个文件到 data/raw\DuEE_Fin_mirror\（文件名里的 / 换成 __）：
#      DuEE-pytorch/data/DuEE-Fin/train.json                 → doc 级 train
#      DuEE-pytorch/data/DuEE-Fin/dev.json                   → doc 级 dev
#      DuEE-pytorch/data/DuEE-Fin/sentence/train.json        → 句子级 train
#      DuEE-pytorch/data/DuEE-Fin/sentence/dev.json          → 句子级 dev
#      DuEE-pytorch/conf/DuEE-Fin/event_schema.json          → 13 类 schema
#      DuEE-pytorch/data/duee_dev.json/License.docx          → 参赛者协议（许可凭证）
#    解压后 data/raw\duee_fin.zip 可删除（原始文件已保留）。

# 2) 依次执行：审计 → 构建 → 验收 → 出图
python data/scripts/audit_raw_index.py     # → data/stats/audit_raw.json
python data/scripts/build_dataset.py       # → data/processed/*.jsonl + event_schema.json + data/stats/build_stats.json
python data/scripts/verify_processed.py    # → data/stats/verify_stats.json（0 错误才算过）
python data/scripts/make_figures.py        # → data/figures/*.png

# 3) 抽样出学生包小样本（3,250 条 → data/sample/）
python tools/make_sample_data.py
```

- 脚本：`data/scripts/build_dataset.py`（构建）、`verify_processed.py`（验收）、`make_figures.py`（图）、
  `audit_raw_index.py`（原始 offset 审计）、`stats_duee10_fin.py`（HF DuEE 候选统计，仅对比用）。
  每个脚本干什么、按什么顺序跑，见 `data/scripts/README.md`。
- 本次验收：**`data/stats/verify_stats.json` 错误数 0**，45,143 个 span 全部满足 `text[start:end] == span.text`。

> **只想拿数据、不想跑构建？** 直接看 `data/README.md §6`：官方渠道（千言 luge.ai 搜 DuEE-Fin）与
> 公开镜像（GitHub `maopademiao/duee_fin`）两种来源、以及从原始文件重建 `processed/` 的两条命令。

---

## 12. 风险与建议（需老师拍板 / 下游注意）

1. **许可风险（最高优先级）**：数据许可限“非商业科研或教学”，且未授权再分发。本课程为收费课程、课件要打包发学生。
   建议按“教学引用”处理：① 课件包只放 **processed 的小样本/派生标注**（或让学生自己用 §11 脚本下载），
   不在公开渠道分发原始数据；② PPT/报告注明数据来源与协议；③ 若要长期产品化，换可商用数据源或联系发布方授权。
2. **offset 歧义**：30.6% span 字符在句内多次出现，取首次出现；评测口径统一用本文档 gold，不要与外部论文数字直接对比。
3. **标注缺失**：论元定位率 98.6%（`环节` 等归一化值无 span），跨句论元被丢；评测 F1 的“理论上限”不足 100%，
   报告里要写明丢弃量与分母。
4. **单事件子集偏差**：L1/L2/L4 只用 `extract`（单事件），不能代表篇章级多事件场景；L5 必须用 `detect`+`dev_docs`。
5. **指标虚高**：句子级 train/dev 有 201 句交集（0.6%），且 7.5% 的 train 句是重复标题句；实验报告注明切分口径。
6. **数据时代**：公告来自 2019–2020 年，与今天的产品/话术有差异；课堂上适合作为“方法对比”的固定 benchmark，不适合作为产品评测集。

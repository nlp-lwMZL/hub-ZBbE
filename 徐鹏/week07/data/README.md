# 数据说明（DuEE-Fin 金融公告事件抽取）

> 本目录只放**数据与数据说明**，不放代码。所有数据处理逻辑见 `common/event_data.py`，
> 数据规模/标签体系的汇总见 `python common/event_data.py`（会打印 13 类 / 92 槽位 / 61 去重角色）。

---

## 1. 数据来源

| 项 | 内容 |
|---|---|
| 数据集 | **DuEE-Fin**（百度「千言」/ 飞桨大赛的金融公告事件抽取数据集） |
| 内容 | A 股上市公司**公告原文** + 标注的「事件类型 / 触发词 / 论元」 |
| 规模 | 13 个事件类型、92 个「类型-角色」槽位、61 个去重角色名 |
| 本机来源 | GitHub 镜像 `maopademiao/duee_fin`（原始下载地址） |
| 本地原始文件 | `data/raw/`（zip / 镜像 json，**不参与训练**，仅备查） |

### 许可与分发（重要）

- DuEE-Fin 来自公开竞赛，原始协议限**非商业科研 / 教学**用途。
- **课件包分发派生数据**：完整 `data/processed/`（随包）+ `data/sample/`（冒烟子集）；**不分发原始发布文件 `raw/`**。
- 学生/第三方若要全量数据，请自行到原始渠道获取并遵守其协议。
- 本仓库的代码（`方法目录/`、`service/`）采用课程内部教学用途，与数据许可无关。

---

## 2. 目录结构

```
data/
├── raw/            # 原始下载文件（zip / GitHub 镜像），仅备查，代码不读
├── processed/      # ★ 完整派生数据（随包发；训练/评测真正用这份）
│   ├── event_schema.json        # 13 事件类型 → 角色列表（含 __enums__ 枚举型角色）
│   ├── train_extract.jsonl      # 5747 句，每句恰好 1 个事件
│   ├── dev_extract.jsonl        # 930 句，每句恰好 1 个事件（主评测集）
│   ├── train_detect.jsonl       # 32794 句（含大量负样本，全量句子）
│   ├── dev_detect.jsonl         # 5302 句（含负样本）
│   ├── dev_extract_clean.jsonl  # 去掉与 train 文本重复后的事件句
│   ├── dev_detect_clean.jsonl   # 去掉与 train 文本重复后的检测句（4815）
│   └── dev_docs.jsonl           # 985 篇篇章级公告（用于系统演示/端到端）
├── sample/         # 快速冒烟子集（确定性抽样 seed=42；几秒跑通全流程）
│   ├── event_schema.json  train_extract.jsonl  dev_extract.jsonl
│   ├── train_detect.jsonl dev_detect.jsonl     dev_docs.jsonl
│   └── sample_stats.json
├── _mock/          # **手造冒烟数据（6 个文件）**，给各方法 `--mock` 用；指标无意义，别删（见 §2.2）
├── scripts/        # 数据集构建与验收脚本（build_dataset / verify_processed / audit_raw_index / stats_duee10_fin …）
│   └── README.md   #   每个脚本干什么、按什么顺序跑（重建 processed 时看这里）
├── stats/          # 审计与验收数字（原始 index 审计、processed 统计、验收结果）
├── figures/        # 数据分布图 6 张（中文标签、150dpi，PPT 用）
├── DATA_REPORT.md  # 数据集选型、统计、许可、数据坑（数字来源之一）
└── README.md       # 本文件：字段契约 + 数据规模速查 + 下载渠道/重建步骤
```

> **随包发什么**：`processed/`（**完整训练/评测数据，约 27 MB**）+ `sample/`（3,250 条**快速冒烟子集**，几秒跑完一遍流程）+ `_mock/`（手造冒烟）。
> **不发** `raw/`（原始发布文件，37 MB）。见到 `sample/` 别误会成「唯一的训练数据」——正式实验一律用 `processed/`（脚本默认就是它）。

### 2.1 `data/sample/` 里每个文件是干什么的（**快速冒烟子集**，正式实验读 `processed/`）

| 文件 | 行数 | 格式 | 谁在用 | 说明 |
|---|---|---|---|---|
| `event_schema.json` | 1 个 JSON | `{"事件类型": [角色, ...], "__enums__": {...}}` | **所有方法** | 13 类 + 92 个「类型×角色」槽位 + 枚举值；标签体系、角色白名单都从它生成 |
| `train_extract.jsonl` | **800** | 见 §3.2 | BERT / BiLSTM / LLM 的训练 | 句级、每句恰好 1 个事件；全量 5,747 |
| `dev_extract.jsonl` | **300** | 见 §3.2 | **所有方法的评测**（主评测集） | 全量 930；学生包用 300 跑出的分数**不能直接和 PPT 上的全量数字比** |
| `train_detect.jsonl` | **1,500** | 见 §3.3 | 两段式的检测器训练 | 含负样本（**77.9% 无事件**）；全量 32,794 |
| `dev_detect.jsonl` | **600** | 见 §3.3 | 检测器评测 | 全量 5,302（clean 版 4,815） |
| `dev_docs.jsonl` | **50** | 见 §3.4 | `service/` 篇章级 demo | 全量 985 篇；切句 → 逐句抽 → 事件流 |
| `sample_stats.json` | 1 个 JSON | 抽样元信息 | 对账 / 复现 | 固定 `seed=42`，记录每个文件的全量行数 vs 抽样行数；`make_sample_data.py --from_sample` 用它校验 |

> 一句话：`sample/` 是「几秒跑通流程 + 看分布」的最小集；**要对上 PPT 的数字就用 `processed/`**（已随包发，脚本默认读它）。

### 2.2 `data/_mock/` 是干什么的（**别删**）

**手造的 20 条句级样本 + 45 条检测样本**，只服务一件事：`--mock` 冒烟。

| 文件 | 行数 | 谁在用 |
|---|---|---|
| `event_schema.json` | 13 类（与全量同构） | 所有 `--mock` 调用 |
| `train_extract.jsonl` / `dev_extract.jsonl` | 各 20 条（13 类型各 1 条 + 7 条边界用例） | `rule/`、`bilstm_crf/`、`bert/`、`bert_pipeline/` |
| `train_detect.jsonl` / `dev_detect.jsonl` | 各 45 条（正负混合） | `bert_pipeline/` 的检测器 |
| `dev_docs.jsonl` | 2 篇 | `service/demo_extract.py --mock` |

- **用途**：把全流程在**纯 CPU、几分钟内**跑通（改完代码先验证不崩），不依赖全量数据、不占 GPU。
- **⚠ 指标毫无意义**：只有 20 条，且是人工编的。汇总脚本会跳过带 `mock` 标记的结果；
  **任何 mock 跑出来的 F1 都不能写进报告/PPT**。
- 加下划线的命名就是提示「内部冒烟数据」，**不是临时文件**，清理时不要删（删了 `--mock` 全部失效）。

**学生包默认用 `data/sample/`**（约 3 MB，跑通全流程够用）：
```bash
python tools/make_sample_data.py --from_sample                 # 校验样本 + 打印统计
python bert/train.py --data_dir data/sample --device cpu --epochs 1 --bert_layers 4
```
**复现 PPT 上的完整数字**：把全量 processed 放回 `data/processed/`，脚本会自动优先用全量
（`--data_dir` 留空时默认读 `data/processed/`）。

---

## 3. 字段契约（**改字段前先改这里和 `common/event_data.py`**）

### 3.1 `event_schema.json`
```json
{
  "股份回购": ["回购方", "披露时间", "回购股份数量", "每股交易价格",
              "占公司总股本比例", "交易金额", "回购完成时间"],
  "__enums__": { "公司上市.环节": ["筹备上市", "暂停上市", "正式上市", "终止上市"] }
}
```
- 键 = 事件类型；值 = 该类型的角色列表（顺序无意义，代码里按名字索引）。
- `__enums__` 是元信息，**遍历事件类型时必须跳过**（`event_data.load_schema()` 已处理）。

### 3.2 `*_extract.jsonl`（句子级，一句恰好 1 个事件）
```json
{"id": "文档id", "sent_id": "句子id", "text": "原文",
 "event": {"type": "公司上市",
           "trigger": {"text": "IPO", "start": 43, "end": 46},
           "args": [{"role": "上市公司", "text": "理想汽车", "start": 19, "end": 23}]}}
```
- `start/end` 为**字符级左闭右开**偏移，且满足 `text[start:end] == 声明的 text`（已全量校验 0 错）。
- `id` 是**文档 id，不唯一**（见 §4 坑 1）。

### 3.3 `*_detect.jsonl`（句子级，含负样本 —— 用于两段式的检测器）
```json
{"id": "...", "sent_id": "...", "text": "...", "has_event": true,
 "types": ["公司上市"], "n_events": 1}
```
- 负样本 `has_event=false, types=[], n_events=0`，训练集里负样本约占 **77.9%**。

### 3.4 `dev_docs.jsonl`（篇章级，系统演示用）
```json
{"id": "文档id", "title": "公告标题", "text": "整篇公告原文",
 "events": [{"type": "...", "trigger": {...}, "args": [...]}]}
```
- 985 篇，events 合计 1493 个；`trigger/args` 的 offset 是**相对整篇 text** 的。

---

## 4. 数据坑（做实验前必须知道，PPT 上会讲）

1. **`id` 是文档 id，不是样本 id（不唯一）**
   dev_extract 930 行只有 823 个唯一 `id`，dev_detect 5302 行只有 1174 个。
   评测时若只按 `id` 对齐，同文档的多句预测会**互相覆盖**（实测 gold-vs-gold 只有 0.93 F1）。
   所以：预测文件必须带 `id` + `sent_id`，评测端按 `f"{id}|{sent_id}"` 对齐
   （见 `common/evaluate_events.py` 的 `row_key`）。

2. **offset 是「回填」出来的，不是原标注自带**
   原数据的论元只给文本片段，`start/end` 是在原文里 `find` 回填的；
   若同一片段在句中多次出现，回填取**首次出现**位置 —— 极少数情况下位置与人工标注不完全一致。
   正因如此，本项目**所有校验都用 `text` 字段比对**、`start/end` 只用于打印/高亮。

3. **论元归一化会丢少量样本**
   转换阶段对无法在原文中定位的论元做了丢弃处理（约 **1.4%** 的论元），
   保证「标注的每个论元都能在原文找到」这一硬约束 —— 否则训练时 BIO 会对不上。
   代价是极少数 event 的论元比原始标注少，做上限分析时要记得这一项。

4. **`dev_detect` 与 `train_detect` 有约 9.2% 文本重复（真实泄漏）**
   原始数据的 dev 划分里混进了训练集句子（5302 → clean 版 4815，少了 487 条 ≈ 9.2%）。
   所以检测指标要**同时报全量和 clean 版**：全量会虚高，clean 版才代表真实泛化。
   （week6 的文本分类也踩过同样的坑，可以前后呼应。）

---

## 5. 数据规模速查

| 文件 | 行数 | 说明 |
|---|---|---|
| train_extract | 5747 | 句级，唯一 `(id,sent_id)` |
| dev_extract | 930 | 主评测集 |
| train_detect | 32794 | 负样本 77.9% |
| dev_detect | 5302 | 负样本 77.9%；clean 版 4815 |
| dev_docs | 985 | 篇章级，events 共 1493 |
| sample/* | 800 / 300 / 1500 / 600 / 50 | 学生包小样本（seed=42） |

平均句长：extract 164 字（p99=393，最长 797）；detect 114 字。
→ 这解释了为什么 `max_len=128` 会截断大量样本、`max_len=256` 仍有约 11% 截断
（本项目「max_len 128 vs 256」消融就是量化这件事的代价）。

---

## 6. 完整数据从哪来（下载渠道 + 重建步骤）

### 6.1 官方渠道（推荐先看协议）

| 项 | 内容 |
|---|---|
| 数据集 | **DuEE-Fin**（百度「千言」/ 2021 语言与智能技术竞赛 金融事件抽取数据集） |
| 官方发布 | 千言数据平台 **https://www.luge.ai/** —— 搜 `DuEE-Fin`（需注册登录后下载） |
| 任务说明 | 2021 语言与智能技术竞赛（千言）金融事件抽取赛道页面 |
| 许可 | 原始包内 `License.docx`：《2021 语言与智能技术竞赛参赛者协议》——**仅限非商业科研 / 教学**，未授权再分发 |

> 本课程因此**只分发派生数据**（`data/processed/` + `data/sample/`），**不分发原始发布文件**；PPT 与文档均注明来源与协议，仅限课程学习使用。

### 6.2 本项目实际使用的公开镜像（免登录，2026-11 实测可用）

官方站需登录、本机曾 15s 超时不可达；实际数据取自 GitHub 公开镜像（与原版同源）：

- 仓库：`https://github.com/maopademiao/duee_fin`
- 下载 + 解压（PowerShell，在项目根目录执行）：

```powershell
Invoke-WebRequest -Uri "https://codeload.github.com/maopademiao/duee_fin/zip/refs/heads/main" -OutFile duee_fin.zip
Expand-Archive duee_fin.zip -DestinationPath data/raw/_tmp_duee_fin
```

- 需要的 5 个文件（本项目的加工脚本只读这几个）：

| 原始路径（镜像内） | 放到 `data/raw/DuEE_Fin_mirror/` 下的文件名 |
|---|---|
| `DuEE-pytorch/data/DuEE-Fin/train.json` | `DuEE-pytorch__data__DuEE-Fin__train.json` |
| `DuEE-pytorch/data/DuEE-Fin/dev.json` | `DuEE-pytorch__data__DuEE-Fin__dev.json` |
| `DuEE-pytorch/data/DuEE-Fin/sentence/train.json` | `DuEE-pytorch__data__DuEE-Fin__sentence__train.json` |
| `DuEE-pytorch/data/DuEE-Fin/sentence/dev.json` | `DuEE-pytorch__data__DuEE-Fin__sentence__dev.json` |
| `DuEE-pytorch/conf/DuEE-Fin/event_schema.json` | `DuEE-pytorch__conf__DuEE-Fin__event_schema.json` |

```powershell
# 例：把解压出来的文件按上表改名拷进 raw 目录（其余文件不用）
New-Item -ItemType Directory -Force data/raw/DuEE_Fin_mirror | Out-Null
$src = "data/raw/_tmp_duee_fin/duee_fin-main/DuEE-pytorch"
Copy-Item "$方法目录/data/DuEE-Fin/train.json"          "data/raw/DuEE_Fin_mirror/DuEE-pytorch__data__DuEE-Fin__train.json"
Copy-Item "$方法目录/data/DuEE-Fin/dev.json"            "data/raw/DuEE_Fin_mirror/DuEE-pytorch__data__DuEE-Fin__dev.json"
Copy-Item "$方法目录/data/DuEE-Fin/sentence/train.json" "data/raw/DuEE_Fin_mirror/DuEE-pytorch__data__DuEE-Fin__sentence__train.json"
Copy-Item "$方法目录/data/DuEE-Fin/sentence/dev.json"   "data/raw/DuEE_Fin_mirror/DuEE-pytorch__data__DuEE-Fin__sentence__dev.json"
Copy-Item "$方法目录/conf/DuEE-Fin/event_schema.json"   "data/raw/DuEE_Fin_mirror/DuEE-pytorch__conf__DuEE-Fin__event_schema.json"
```

### 6.3 从原始文件重建 `processed/`（两条命令，在项目根目录执行）

```bash
python data/scripts/build_dataset.py            # 原始 json → processed 的四类文件 + schema（约 30 秒）
python data/scripts/verify_processed.py         # 全量校验：45,143 个 span 必须 text[start:end]==声明文本，应 0 错
```

> 两个脚本随包分发（`data/scripts/`）。改字段前先改 §3 契约与 `common/event_data.py`，再重跑上面两条。
> 生成小样本：`python tools/make_sample_data.py`（固定 seed=42，覆盖 `data/sample/`）。

### 6.4 常见问题

| 问题 | 处理 |
|---|---|
| 千言站登录/下载失败 | 用 §6.2 的 GitHub 镜像；两边同源 |
| 想复现 PPT 全量数字 | 按 §6.3 重建 `data/processed/`，脚本 `--data_dir` 留空默认读全量 |
| 换自己的公告数据 | 产出同契约的 `*_extract.jsonl` 即可，方法与评测不用改；`event_schema.json` 换成自己的类型/角色表 |
| 能不能分发原始发布文件 | 不能：协议限非商业科研/教学。课件包发的是**派生数据**（`processed/` + `sample/`），`raw/` 不外发；派生数据同样仅限课程学习使用 |

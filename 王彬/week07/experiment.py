"""CPU information extraction experiments. Run `python my_train/experiment.py --help`."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
import urllib.request

# 路径约定：CODE_HOME始终指向脚本目录；BASE决定本次数据和结果放在哪里。
# 全量实验由外部设置MY_TRAIN_RUN_DIR指向full_run，密钥和累计账本仍共用CODE_HOME。
# 本脚本训练的是字符级BiLSTM
CODE_HOME = Path(__file__).resolve().parent
BASE = Path(os.environ.get("MY_TRAIN_RUN_DIR", str(CODE_HOME))).resolve()
TEACHER = CODE_HOME.parent / "code"
sys.path.insert(0, str(TEACHER / "common"))


# 配置读取：本机已有环境变量优先，.env.local只补缺项；不要打印密钥。
def load_local_config():
    """Read local secrets without printing or serializing them; existing env takes precedence."""
    path = CODE_HOME / ".env.local"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('\"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


# 读取JSONL：一行一个样本，返回列表；空行不作为样本。
def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# 保存普通JSON，用于配置、指标和进度；保留中文，自动创建父目录。
def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# 保存逐行JSON，适合预测、数据和原始回复等可逐条核对的产物。
def jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# 按文件路径导入老师的规则或解析模块，复用实现，避免复制出不同口径。
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 按事件类型比例抽样，并尽量保留各类型；固定随机种子使样本选择可复现。
# 适用于本项目n足够覆盖所有类型的调用；不是文档分组划分器。
def stratified(rows, n, seed):
    """Proportional allocation, with minimum coverage of available event types."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: (r["id"], r["sent_id"])):
        groups[row["event"]["type"]].append(row)
    n = min(n, len(rows))
    targets = {t: min(len(rs), max(1, int(n * len(rs) / len(rows)))) for t, rs in groups.items()}
    while sum(targets.values()) < n:
        options = [t for t in groups if targets[t] < len(groups[t])]
        t = max(options, key=lambda k: (n * len(groups[k]) / len(rows) - targets[k], k))
        targets[t] += 1
    while sum(targets.values()) > n:
        t = max((t for t in groups if targets[t] > 1), key=lambda k: targets[k])
        targets[t] -= 1
    selected = []
    for t in sorted(groups):
        rng.shuffle(groups[t])
        selected.extend(groups[t][:targets[t]])
    rng.shuffle(selected)
    return selected


# 注意：此函数是早期小样本准备流程，固定800/130/200，且同文本只留首条。
# 全量5747条的分组划分由full_run.py准备；这里看到800不代表全量训练只用了800。
# 已有manifest时直接保留数据，避免重跑命令悄悄更换实验样本。
def prepare(args):
    dest = BASE / "data"
    if (dest / "manifest.json").exists():
        print("Existing fixed data retained; see my_train/data/manifest.json")
        return
    original = TEACHER / "data" / "processed"
    train_all = read(original / "train_extract.jsonl")
    # Identical text is a single sampling unit. Conflicting duplicates retain first annotation.
    unique = {}
    for row in train_all:
        unique.setdefault(row["text"], row)
    rows = list(unique.values())
    doc_ids = sorted({r["id"] for r in rows})
    random.Random(42).shuffle(doc_ids)
    val_ids = set(doc_ids[:math.ceil(len(doc_ids) * .2)])
    train = stratified([r for r in rows if r["id"] not in val_ids], 800, 42)
    val = stratified([r for r in rows if r["id"] in val_ids], 130, 43)
    # Use clean dev and remove overlap against ALL original training texts/docs.
    train_texts = {r["text"] for r in train_all}
    train_docs = {r["id"] for r in train_all}
    test_unique = {}
    for row in read(original / "dev_extract_clean.jsonl"):
        if row["text"] not in train_texts and row["id"] not in train_docs:
            test_unique.setdefault(row["text"], row)
    test = stratified(list(test_unique.values()), 200, 44)
    assert [len(train), len(val), len(test)] == [800, 130, 200]
    for a, b in [(train, val), (train, test), (val, test)]:
        assert not {r["text"] for r in a} & {r["text"] for r in b}
        assert not {r["id"] for r in a} & {r["id"] for r in b}
    schema = json.loads((original / "event_schema.json").read_text(encoding="utf-8"))
    save(dest / "event_schema.json", schema)
    for name, items in [("train", train), ("val", val), ("test", test)]:
        jsonl(dest / f"{name}.jsonl", items)
    save(dest / "manifest.json", {
        "source": "teacher processed DuEE-Fin derived data; test from dev_extract_clean",
        "sampling": "deduplicate text; document-disjoint train/val; proportional type sampling; seeds 42/43/44",
        "selection_policy": "best checkpoint by validation argument F1; test never used to choose epoch",
        "train_original_rows": len(train_all), "train_unique_texts": len(rows),
        "duplicate_policy": "retain first annotation for identical text; no semantic relabeling",
        "splits": {name: {"n": len(items), "types": dict(Counter(r["event"]["type"] for r in items)),
                           "sha256": hashlib.sha256((dest / f"{name}.jsonl").read_bytes()).hexdigest()}
                   for name, items in [("train", train), ("val", val), ("test", test)]},
        "scope": "small CPU teaching experiment, positive single-event sentences only; not teacher dev930 benchmark",
    })
    print("Fixed split: train=800, val=130, test=200; no document/text overlap.")


# 按固定顺序返回拟合集、验证集、测试集；实际规模取决于BASE下的数据。
def data():
    return tuple(read(BASE / "data" / f"{s}.jsonl") for s in ("train", "val", "test"))


# 统一调用老师的评测函数：触发词、角色与论元文本、完整事件分别计分。
# F1是集合匹配指标，不是模型输出的概率；预测为空的样本也保留。
def metric(tag, gold, pred, meta):
    from evaluate_events import evaluate
    result = evaluate(gold, pred, verbose=False)
    result["experiment"] = tag
    result["meta"] = meta
    jsonl(BASE / "outputs" / "predictions" / f"{tag}.jsonl", pred)
    save(BASE / "outputs" / "metrics" / f"{tag}.json", result)
    print(tag, "T/A/E", *(result[k]["f1"] for k in ("trigger", "argument", "event_exact")), flush=True)
    return result


# 规则对照不做梯度训练：从拟合集构造词典，再比较人工同义词及匹配优先级。
def rule(args):
    from event_data import load_schema, make_pred_row
    rules = module("teacher_rules", TEACHER / "rule" / "run_rule.py")
    train, _, test = data()
    schema = load_schema(BASE / "data" / "event_schema.json")
    auto = rules.build_trigger_dict(train, min_count=2)
    companies = rules.build_company_dict(train)
    original_find = rules.find_trigger
    for tag, manual, longest in [("A_rule", True, True), ("A_no_manual", False, True),
                                  ("A_short_first", True, False)]:
        rules.find_trigger = original_find
        pairs = rules.merge_trigger_dict(auto, rules.MANUAL_TRIGGERS if manual else {})
        if not longest:
            # Teacher find_trigger explicitly maximizes length, independent of list order.
            # Change the selection criterion itself for a genuine shortest-first ablation.
            def shortest(text, pairs):
                hits = [(len(word), text.find(word), etype, word) for word, etype in pairs if word in text]
                if not hits:
                    return None
                _, pos, etype, word = min(hits)
                return etype, word, pos, pos + len(word)
            rules.find_trigger = shortest
        t0 = time.perf_counter()
        preds = []
        for row in test:
            ev = rules.extract_one(row["text"], schema, pairs, companies)
            preds.append(make_pred_row(row, [ev] if ev else []))
        elapsed = time.perf_counter() - t0
        metric(tag, test, preds, {"train_n": len(train), "manual_synonyms": manual,
                                 "longest_first": longest, "inference_s": elapsed,
                                 "ms_per_sentence": elapsed * 1000 / len(test), "train_s": 0})


# CPU监督训练入口：准备BIO数据 → BiLSTM编码 → Linear或CRF → 验证选点 → 测试。
# 没有大模型聊天角色，因此这里不使用“仅assistant计算loss”的SFT策略。
def train(args):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from crf import CRFClassifier
    from event_data import (CharDataset, collate_char, make_vocab, load_schema, build_joint_schema,
                            decode_bio_spans, spans_to_event, make_pred_row, illegal_bio_stats,
                            build_bio_constraint_mask, build_start_constraint_mask)
    from evaluate_events import evaluate
    # 固定CPU线程和随机种子，开启确定性计算，减少复现差异；不同软件环境仍可能不同。
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    train_rows, val_rows, test_rows = data()
    if 0 < args.train_n < len(train_rows):
        train_rows = stratified(train_rows, args.train_n, 42)
    # 字符字表只从拟合集建立，避免读入验证/测试字符信息；未知字符使用公共模块的映射。
    vocab = make_vocab(train_rows)
    schema = load_schema(BASE / "data" / "event_schema.json")
    # 联合标签把触发词、事件类型和论元角色编码进同一BIO序列。
    # B表示span开头，I表示同类span后续字符，O表示不属于标注span；不是逐位置消融。
    scheme = build_joint_schema(schema)

    # 随机初始化的小型编码器：字符id → 64维embedding → 双向LSTM。
    # 每方向隐藏维64，拼接后每个字符输出128维；没有加载预训练BERT权重。
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(vocab), 64, padding_idx=0)
            self.dropout = nn.Dropout(args.dropout)
            self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)

        def forward(self, ids, mask, token_type_ids=None):
            # Match teacher encoder for direct pedagogical comparison.
            # 这里未用pack_padded_sequence，mask未传入LSTM内部；反向LSTM可能受批内padding影响。
            #             # padding虽不计loss，但不等于完全不影响编码，这是实现的一个限制。
            hidden, _ = self.lstm(self.dropout(self.emb(ids)))
            return hidden

    # CharDataset负责字符截断及span到BIO转换，max_len单位为字符。
    # collate_char按批次动态补齐，产生input_ids、attention_mask、labels和原始row_idx。
    datasets = [CharDataset(rows, vocab, scheme, args.max_len) for rows in (train_rows, val_rows, test_rows)]
    # 只打乱训练集；验证和测试顺序固定。row_idx用于把批内输出对应回原样本。
    loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=(i == 0),
                          collate_fn=collate_char, num_workers=0,
                          generator=torch.Generator().manual_seed(args.seed))
               for i, ds in enumerate(datasets)]
    # 公共分类器先把128维特征投影为各标签的发射分数。
    # Linear使用逐字符交叉熵，labels=-100的padding不计loss；CRF使用整条路径负对数似然，
    # 由attention_mask排除padding。CRF解码用维特比，Linear逐位置取argmax。
    model = CRFClassifier(Encoder(), 128, len(scheme["labels"]), dropout=args.dropout,
                          use_crf=args.decoder == "crf")
    # AdamW更新所有可训练参数；weight_decay是权重衰减，不是dropout。
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # 预热步数按总更新步数的比例计算；学习率线性升到目标值后保持恒定。
    # warmup=0时仍保留至少1步的分母，调度器倍率立即为1，相当于不预热。
    warmup = max(1, int(len(loaders[0]) * args.epochs * args.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1., (step + 1) / warmup))
    # 保存超参数、环境和数据/源码哈希便于溯源；修改注释后源码哈希会变化，历史记录保留原值。
    config = vars(args).copy()
    config.pop("func")
    config.update({"embedding": 64, "hidden_per_direction": 64, "num_labels": len(scheme["labels"]),
                   "device": "cpu", "python": platform.python_version(), "torch": torch.__version__,
                   "params": sum(p.numel() for p in model.parameters()), "vocab_size": len(vocab),
                   "checkpoint_selection": "validation argument F1, tie by lower validation loss"})
    config["argv"] = sys.argv
    config["data_sha256"] = {name: hashlib.sha256((BASE / "data" / f"{name}.jsonl").read_bytes()).hexdigest()
                              for name in ("train", "val", "test")}
    config["source_sha256"] = {str(path.relative_to(CODE_HOME.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(__file__), TEACHER / "common/crf.py", TEACHER / "common/event_data.py",
                                             TEACHER / "common/evaluate_events.py")}
    folder = BASE / "outputs" / "runs" / args.tag
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "history.json").exists():
        raise FileExistsError(f"Completed run exists: {folder}; choose another tag")
    save(folder / "config.json", config)

    # 推理关闭dropout并禁用梯度；把标签id还原成BIO，再还原span和事件。
    def predict(loader, rows):
        model.eval()
        preds, tags_all = [], []
        start = time.perf_counter()
        with torch.no_grad():
            for batch in loader:
                paths = model.predict(batch["input_ids"], batch["attention_mask"])
                for index, path in zip(batch["row_idx"].tolist(), paths):
                    tags = [scheme["id2label"][t] for t in path]
                    row = rows[index]
                    event = spans_to_event(decode_bio_spans(tags), row["text"])
                    preds.append(make_pred_row(row, [event] if event else []))
                    tags_all.append(tags)
        return preds, illegal_bio_stats(tags_all), time.perf_counter() - start

    # 验证不更新参数。val_loss是各batch的loss均值，不是按字符数重新加权后的全局均值。
    def validation():
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for batch in loaders[1]:
                _, loss = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
                val_loss += loss.item()
        preds, _, _ = predict(loaders[1], val_rows)
        return val_loss / len(loaders[1]), evaluate(val_rows, preds, verbose=False)

    history = []
    best, start, steps = (-1., float("-inf")), time.perf_counter(), 0
    print("START", args.tag, json.dumps(config, ensure_ascii=False), flush=True)
    with (folder / "steps.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            # 每轮恢复训练模式以启用dropout；验证中的eval模式不会自动恢复。
            model.train()
            total, count, e0 = 0., 0, time.perf_counter()
            for batch in loaders[0]:
                _, loss = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss")
                # 每个batch先清旧梯度，再反向传播；本实现没有跨batch梯度累积。
                opt.zero_grad()
                loss.backward()
                # 裁剪总梯度范数至5，缓解梯度爆炸；记录的norm是裁剪前数值。
                norm = nn.utils.clip_grad_norm_(model.parameters(), 5.)
                # 先更新模型参数，再推进学习率调度器；每个batch算一个optimizer step。
                opt.step()
                sched.step()
                total += loss.item()
                count += 1
                steps += 1
                log.write(json.dumps({"epoch": epoch, "step": steps, "loss": loss.item(),
                                      "lr": opt.param_groups[0]["lr"], "grad_norm_before_clip": float(norm),
                                      "elapsed_s": time.perf_counter() - start}) + "\n")
            log.flush()
            val_loss, val_metric = validation()
            rec = {"epoch": epoch, "train_loss": total / count, "val_loss": val_loss,
                   "val_trigger_f1": val_metric["trigger"]["f1"],
                   "val_argument_f1": val_metric["argument"]["f1"],
                   "val_event_f1": val_metric["event_exact"]["f1"],
                   "epoch_s": time.perf_counter() - e0}
            history.append(rec)
            print(json.dumps(rec), flush=True)
            # 按验证论元F1选最佳checkpoint；同分时val_loss越低越好。
            #             # 不用测试集选epoch，也不直接以严格事件F1作为训练目标。
            score = (rec["val_argument_f1"], -val_loss)
            if score > best:
                best = score
                torch.save({"state_dict": model.state_dict(), "vocab": vocab,
                            "labels": scheme["labels"], "config": config, "epoch": epoch}, folder / "best.pt")
            save(folder / "progress.json", {"epochs": history, "finished": False})
    elapsed = time.perf_counter() - start
    # 训练结束后加载验证最佳权重，最后一轮不一定是最好的一轮。
    ckpt = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    preds, illegal, infer_s = predict(loaders[2], test_rows)
    meta = {**config, "train_n": len(train_rows), "val_n": len(val_rows), "test_n": len(test_rows),
            "best_epoch": ckpt["epoch"], "train_s": elapsed, "inference_s": infer_s,
            "ms_per_sentence": infer_s * 1000 / len(test_rows), "illegal_bio": illegal,
            "test_truncated": datasets[2].n_truncated}
    metric(args.tag, test_rows, preds, meta)
    # 同一CRF权重增加BIO合法起始/转移约束，再评测一次；没有重新训练。
    # _hard的train_s=0表示本次额外解码不训练，不代表基础CRF训练免费。
    if args.decoder == "crf":
        model.crf.constraint = build_bio_constraint_mask(scheme)
        model.crf.start_constraint = build_start_constraint_mask(scheme)
        preds, illegal, infer_s = predict(loaders[2], test_rows)
        metric(args.tag + "_hard", test_rows, preds,
               {**meta, "decode_hard_constraint": True, "train_s": 0,
                "source_checkpoint": str(folder / "best.pt"), "illegal_bio": illegal,
                "inference_s": infer_s, "ms_per_sentence": infer_s * 1000 / len(test_rows)})
    save(folder / "history.json", {"config": config, "epochs": history, "finished": True,
                                  "train_s": elapsed, "best_epoch": ckpt["epoch"]})
    save(folder / "progress.json", {"epochs": history, "finished": True,
                                   "train_s": elapsed, "best_epoch": ckpt["epoch"]})


# 构造API提示：zero给任务和schema；few加入示例对话；optimized加强边界与角色要求。
# 示例中的assistant是提示上下文，这里没有反向传播或assistant loss mask训练。
def messages(mode, text, schema, examples):
    types = "\n".join(t + "：" + "、".join(roles) for t, roles in schema.items() if not t.startswith("__"))
    system = ('你是中文金融事件抽取助手。从原文抽取一个事件，只输出JSON。类型及角色如下：\n' + types +
              '\n格式：{"events":[{"type":"事件类型","trigger":{"text":"触发词原文"},'
              '"arguments":[{"role":"角色","argument":"论元原文"}]}]}。没有事件输出{"events":[]}。')
    if mode == "optimized":
        system += ('\n所有触发词和论元必须逐字复制原文，不改写、不推断未出现的信息；'
                   '只使用该类型允许的角色。没有依据的角色省略。区别质押与解除质押、增持与减持。'
                   '金额、日期、数量保留原文边界和限定词，不自行换算单位。相同角色可有不同论元，'
                   '相同角色相同文本不要重复。只输出一个JSON对象，不要解释或Markdown围栏。')
    result = [{"role": "system", "content": system}]
    if mode == "few":
        for row in examples:
            result.extend([{"role": "user", "content": row["text"]},
                           {"role": "assistant", "content": json.dumps(target(row), ensure_ascii=False)}])
    result.append({"role": "user", "content": text})
    return result


# 把老师样本的event/args字段改写成API示例要求的events/arguments结构。
def target(row):
    event = row["event"]
    return {"events": [{"type": event["type"], "trigger": {"text": event["trigger"]["text"]},
                        "arguments": [{"role": a["role"], "argument": a["text"]} for a in event["args"]]}]}


# API仅远程推理，不更新本机或服务端模型权重，也不需要本机显卡。
# 调用会计费；preview只保存提示，实际请求仍受下方累计预算控制。
def api(args):
    # Standard-library HTTP avoids requiring an SDK. Only OpenAI-compatible chat endpoints.
    train_rows, _, test = data()
    schema = json.loads((BASE / "data/event_schema.json").read_text(encoding="utf-8"))
    # 只从拟合集为六种类型各选一个最短示例；不是从测试答案选择示例。
    examples = []
    for t in ("解除质押", "质押", "股东减持", "股东增持", "高管变动", "企业收购"):
        examples.append(min((r for r in train_rows if r["event"]["type"] == t), key=lambda r: len(r["text"])))
    schema_wrapped = {"event_types": {k: v for k, v in schema.items() if not k.startswith("__")}}
    prompt_parser = module("my_prompt_parser", TEACHER / "llm_zero_shot/event_prompt.py")
    test = test[:args.n]
    payloads = [messages(args.mode, r["text"], schema, examples) for r in test]
    prompt_file = BASE / "outputs/prompts" / f"{args.mode}.json"
    save(prompt_file, {"mode": args.mode, "example_ids": [[r["id"], r["sent_id"]] for r in examples] if args.mode == "few" else [],
                       "first_sample_messages": payloads[0], "n": len(test)})
    if args.preview:
        print("Prompt saved; no API calls:", prompt_file)
        return
    load_local_config()
    key, model, url = (os.getenv(k, "") for k in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"))
    if not all((key, model, url)):
        raise RuntimeError("Set LLM_API_KEY, LLM_MODEL and LLM_BASE_URL locally first.")
    tag = {"zero": "D_api_zero", "few": "E_api_few", "optimized": "F_api_optimized"}[args.mode] + "_" + args.label
    raw_path = BASE / "outputs/raw" / f"{tag}.jsonl"
    if raw_path.exists() and not args.resume:
        raise FileExistsError(f"Run label already exists: {raw_path}")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    # 所有实验共用累计费用账本，避免更换BASE后重新获得19元预算。
    # 价格常量只对应当时实验的估算口径，不能视为未来API的实时价格。
    ledger_path = CODE_HOME / "outputs/api_budget.jsonl"
    spent = sum(r.get("cost_upper_cny", 0) for r in read(ledger_path)) if ledger_path.exists() else 0.
    budget = 19.0  # User-raised cumulative budget for ALL calls, including connection probes.
    # Official peak price as checked 2026-11-26: $0.30/M input, $1.20/M output.
    # Budgeting conversion 10 CNY/USD is a deliberate safety buffer, not an exchange-rate claim.
    def cost_upper(prompt_tokens, completion_tokens):
        return 10.0 * (.30 * prompt_tokens + 1.20 * completion_tokens) / 1_000_000

    def ledger(record):
        record["run_root"] = str(BASE.relative_to(CODE_HOME))
        with ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    from event_data import make_pred_row
    preds, usage, total = [], Counter(), 0.
    # 断点续跑只复用已记录的连续前缀，并校验样本顺序和模型名，避免重复付费。
    previous = read(raw_path) if raw_path.exists() else []
    for row, raw_record in zip(test, previous):
        assert (row["id"], row["sent_id"]) == (raw_record["id"], raw_record["sent_id"])
        assert raw_record["requested_model"] == model
        events, _ = prompt_parser.parse_model_output(raw_record["raw"], row["text"], schema_wrapped)
        preds.append(make_pred_row(row, events))
        usage.update({k: int(v) for k, v in (raw_record.get("usage") or {}).items() if isinstance(v, int)})
        total += raw_record.get("elapsed_s", 0)
    assert len(previous) <= len(test)
    completed_prefix = len(previous)
    wall_start = time.perf_counter()

    # 单次HTTP请求：输出上限640是生成token数，与BiLSTM的128/256字符截断不同。
    # 出错只返回异常类型，不输出可能含敏感信息的请求内容；不自动重试。
    def request_one(msgs):
            body = {"model": model, "messages": msgs, "max_tokens": 640, "temperature": 0, "stream": False}
            if "deepseek.com" in url:
                body["thinking"] = {"type": "disabled"}
            if args.json_mode:
                body["response_format"] = {"type": "json_object"}
            request = urllib.request.Request(url.rstrip("/") + "/chat/completions",
                                            json.dumps(body).encode("utf-8"),
                                            {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    obj = json.load(response)
                return obj, time.perf_counter() - started, None
            except Exception as exc:
                return {}, time.perf_counter() - started, type(exc).__name__

    # 并发数由args.workers决定；主线程先预留整批费用，再发送请求，随后按usage冲正。
    # 若请求失败或缺失usage则保留预留金额。这是保守估算，不是平台实际账单。
    # 此机制是单进程控制，不应同时启动多个共享账本的API进程。
    # Four in-flight requests; budget reservation and ledger are managed by this single thread.
    # Reserve BEFORE dispatch, so even a process crash leaves a conservative charge in the ledger.
    with raw_path.open("a" if args.resume else "w", encoding="utf-8") as output, ThreadPoolExecutor(max_workers=args.workers) as executor:
        for start in range(completed_prefix, len(test), args.workers):
            chunk = payloads[start:start + args.workers]
            reserves = [cost_upper(4 * len(json.dumps(msgs, ensure_ascii=False).encode("utf-8")) + 2048, 640) for msgs in chunk]
            if spent + sum(reserves) > budget:
                save(BASE / "outputs/api_budget_stop.json", {"run": tag, "completed": len(preds),
                     "spent_upper_cny": spent, "next_batch_reserve_cny": sum(reserves), "limit_cny": budget})
                raise RuntimeError("Stopped before batch: cumulative 19 CNY budget guard.")
            for j, reserve in enumerate(reserves):
                ledger({"run": tag, "sample_index": start + j, "status": "reserved_before_dispatch",
                        "cost_upper_cny": reserve, "time": time.time()})
            spent += sum(reserves)
            responses = list(executor.map(request_one, chunk))
            errors = []
            for j, (obj, elapsed, error) in enumerate(responses):
                row, reserve = test[start + j], reserves[j]
                u = obj.get("usage", {})
                if not error and "prompt_tokens" in u and "completion_tokens" in u:
                    cost = cost_upper(u["prompt_tokens"], u["completion_tokens"])
                    ledger({"run": tag, "sample_index": start + j, "status": "actual_usage_adjustment",
                            "cost_upper_cny": cost - reserve, "cost_request_upper_cny": cost,
                            "prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
                            "returned_model": obj.get("model"), "time": time.time()})
                    spent += cost - reserve
                    usage.update({k: int(v) for k, v in u.items() if isinstance(v, int)})
                else:
                    errors.append(error or "missing_usage")
                raw = (obj.get("choices") or [{"message": {}}])[0]["message"].get("content") or ""
                events, info = prompt_parser.parse_model_output(raw, row["text"], schema_wrapped)
                preds.append(make_pred_row(row, events))
                total += elapsed
                output.write(json.dumps({"id": row["id"], "sent_id": row["sent_id"], "raw": raw,
                                         "info": info, "usage": u, "elapsed_s": elapsed, "error": error,
                                         "requested_model": model, "returned_model": obj.get("model")}, ensure_ascii=False) + "\n")
            output.flush()
            print(tag, len(preds), "/", len(test), "budget_upper_cny", round(spent, 4), flush=True)
            if errors:
                raise RuntimeError("Stopped after recording batch; no automatic retries: " + ",".join(errors))
    metric(tag, test, preds, {"model": model, "mode": args.mode, "api_calls": len(test),
                             "usage": dict(usage), "inference_s": total, "train_s": 0,
                             "json_mode": args.json_mode, "concurrency": args.workers,
                             "resumed_prefix": completed_prefix,
                             "wall_s": time.perf_counter() - wall_start,
                             "ms_per_sentence": total * 1000 / len(test), "n": len(test)})


# 读取已保存指标生成终端表和JSON，不训练、不调用API；整合README由其他报告脚本生成。
def summarize(args):
    files = sorted((BASE / "outputs/metrics").glob("*.json"))
    records = []
    lines = ["# 我的实验结果", "", "同一份固定测试集；API若只跑前几条，须按相同子集投影后比较。", "",
             "| 实验 | n | 触发词F1 | 论元F1 | 事件F1 | 训练秒 | 推理ms/句 | 非法BIO句占比 |", "|---|---|---|---|---|---|---|---|"]
    for p in files:
        result = json.loads(p.read_text(encoding="utf-8"))
        records.append(result)
        m = result["meta"]
        vals = [p.stem, str(result["num_samples"])] + [f'{result[k]["f1"]:.4f}' for k in ("trigger", "argument", "event_exact")]
        vals += [f'{m["train_s"]:.1f}' if "train_s" in m else "—",
                 f'{m["ms_per_sentence"]:.2f}' if "ms_per_sentence" in m else "—",
                 str(m.get("illegal_bio", {}).get("illegal_seq_rate", "—"))]
        lines.append("| " + " | ".join(vals) + " |")
    save(BASE / "outputs/method_summary.json", {"methods": records})
    print("\n".join(lines))


# 命令分派：prepare准备旧小样本；rule跑规则；train训练BiLSTM；api调用接口。
# 注意train默认8轮，已完成的全量实验由调度脚本显式传入30轮。
# API的confirm_calls参数当前仅被解析，未参与放行判断，不能把它当作付费保护开关。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command, fn in [("prepare", prepare), ("rule", rule), ("summarize", summarize)]:
        sub.add_parser(command).set_defaults(func=fn)
    p = sub.add_parser("config-status")
    def config_status(args):
        load_local_config()
        print(json.dumps({"key_configured": bool(os.getenv("LLM_API_KEY")),
                          "model": os.getenv("LLM_MODEL", ""),
                          "base_url": os.getenv("LLM_BASE_URL", ""),
                          "budget_cny": 19}, ensure_ascii=False))
    p.set_defaults(func=config_status)
    p = sub.add_parser("train")
    p.add_argument("--tag", required=True)
    p.add_argument("--decoder", choices=["linear", "crf"], default="linear")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--train_n", type=int, default=-1, help="-1 uses all fit rows in the chosen data directory")
    p.add_argument("--dropout", type=float, default=.3)
    p.add_argument("--warmup", type=float, default=.1)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.set_defaults(func=train)
    p = sub.add_parser("api")
    p.add_argument("--mode", choices=["zero", "few", "optimized"], default="zero")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--n", type=int, choices=range(1, 10001), default=200)
    p.add_argument("--label", default="main")
    p.add_argument("--json_mode", action="store_true")
    p.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    p.add_argument("--resume", action="store_true", help="retain already recorded prefix replies without repeating paid calls")
    p.add_argument("--confirm_calls", action="store_true")
    p.set_defaults(func=api)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

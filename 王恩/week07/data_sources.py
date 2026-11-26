"""适配 DuEE-Fin 和 DuEE 1.0 到课件的单事件格式。"""

import hashlib
import json
import random
import unicodedata
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile, ZipFile

from http_client import open_url


ROOT = Path(__file__).resolve().parent
COURSE = ROOT.parents[1] / "courseware" / "week07" / "code"
DUEE_URL = "https://dataset-bj.cdn.bcebos.com/qianyan/DuEE_1_0.zip"
DUEE_SHA256 = "15c45b64cdbc2da61386993db2567fb4004ad76a490a00d0e880b9d16ad599dd"
DUEE_ARCHIVE = ROOT / "data" / "DuEE_1_0.zip"


def read_jsonl(lines):
    return [json.loads(line) for line in lines if line.strip()]


def open_duee_archive(path):
    if not path.exists() and path == DUEE_ARCHIVE:
        with open_url(DUEE_URL, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != DUEE_SHA256:
            raise ValueError("DuEE 1.0 下载文件的 SHA-256 与官方存档快照不符")
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary.write(data)
        Path(temporary.name).replace(path)
    try:
        return ZipFile(path)
    except BadZipFile as exc:
        raise ValueError(f"DuEE 1.0 压缩包损坏：{path}") from exc


def span(value, start):
    return {"text": value, "start": start, "end": start + len(value)}


def load_general(archive_path):
    with open_duee_archive(archive_path) as archive:
        schemas = read_jsonl(archive.read("DuEE_1_0/event_schema.json").decode("utf-8-sig").splitlines())
        schema = {item["event_type"]: [r["role"] for r in item["role_list"]] for item in schemas}
        splits = []
        excluded = {}
        for split in ("train", "dev"):
            raw = read_jsonl(archive.read(f"DuEE_1_0/{split}.json").decode("utf-8-sig").splitlines())
            rows = []
            for row in raw:
                if len(row["event_list"]) != 1:
                    continue
                event = row["event_list"][0]
                arguments = [
                    dict(span(a["argument"], a["argument_start_index"]), role=a["role"])
                    for a in event["arguments"]
                ]
                rows.append({
                    "uid": row["id"],
                    "text": row["text"],
                    "event": {
                        "type": event["event_type"],
                        "trigger": span(event["trigger"], event["trigger_start_index"]),
                        "args": arguments,
                    },
                })
            excluded[split] = len(raw) - len(rows)
            splits.append(rows)
    return schema, *splits, excluded


def load_financial():
    source = COURSE / "data" / "processed"
    schema = json.loads((source / "event_schema.json").read_text(encoding="utf-8"))
    schema.pop("__enums__", None)
    splits = []
    for split in ("train", "dev"):
        with (source / f"{split}_extract.jsonl").open(encoding="utf-8") as stream:
            rows = read_jsonl(stream)
        for row in rows:
            row["uid"] = f'{row["id"]}|{row["sent_id"]}'
        splits.append(rows)
    return schema, *splits, {"train": 0, "dev": 0}


def normalized_text(value):
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def validate_rows(rows, schema):
    seen = set()
    for row in rows:
        uid, text, event = row["uid"], row["text"], row["event"]
        if uid in seen:
            raise ValueError(f"重复样本 ID {uid}")
        seen.add(uid)
        roles = schema[event["type"]]
        for part in [event["trigger"], *event["args"]]:
            start, end = part["start"], part["end"]
            if not 0 <= start < end <= len(text) or text[start:end] != part["text"]:
                raise ValueError(f"样本 {uid} 的标注片段与原文不一致")
            if "role" in part and part["role"] not in roles:
                raise ValueError(f"样本 {uid} 包含未定义角色 {part['role']}")


def fingerprint(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sample(rows, limit, seed):
    if not limit or len(rows) <= limit:
        return rows
    return [rows[i] for i in sorted(random.Random(seed).sample(range(len(rows)), limit))]


def load_dataset(name, archive_path, train_limit, eval_limit, seed):
    if name == "duee":
        schema, train, dev, excluded = load_general(archive_path)
    else:
        schema, train, dev, excluded = load_financial()
    validate_rows(train, schema)
    validate_rows(dev, schema)
    training_text = {normalized_text(row["text"]) for row in train}
    clean_dev = [row for row in dev if normalized_text(row["text"]) not in training_text]
    selected_train = sample(train, train_limit, seed)
    selected_dev = sample(clean_dev, eval_limit, seed)
    if not selected_train or not selected_dev:
        raise ValueError(f"{name} 缺少可用的训练或评测样本")
    manifest = {
        "schema_types": len(schema),
        "schema_roles": sum(map(len, schema.values())),
        "excluded_non_single_event": excluded,
        "train_available": len(train),
        "dev_before_dedup": len(dev),
        "dev_duplicate_with_train": len(dev) - len(clean_dev),
        "dev_available": len(clean_dev),
        "train_used": len(selected_train),
        "dev_used": len(selected_dev),
        "train_sha256": fingerprint(selected_train),
        "dev_sha256": fingerprint(selected_dev),
    }
    return schema, selected_train, selected_dev, manifest

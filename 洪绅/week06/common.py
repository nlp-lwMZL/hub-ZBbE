"""公共部分：标签、评测、数据准备。算法实现分别放在 l0~l4 文件中。

python common.py 生成小数据集；没有原始数据时先下载。原始文件不覆盖。
"""
import csv
import json
import os
from pathlib import Path
import random
import tempfile
import urllib.request
import zipfile

URL = "https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip"
OUT = Path(__file__).resolve().parent / "data/tnews"  # 原始全量数据，保留不覆盖
DATA_DIR = OUT.parent / "tnews_quick"  # 所有算法统一读取这份练习子集
TRAIN_SAMPLES = 3000
DEV_SAMPLES = 150
TEST_SAMPLES = 150

LABEL2NAME = {
    "100": "故事", "101": "文化", "102": "娱乐", "103": "体育", "104": "财经",
    "106": "房产", "107": "汽车", "108": "教育", "109": "科技", "110": "军事",
    "112": "旅游", "113": "国际", "114": "证券", "115": "农业", "116": "电竞",
}


def count_labels(y):
    """统计每个标签出现几次，返回 {标签: 次数}"""
    counts = {}
    for label in y:
        counts[label] = counts.get(label, 0) + 1
    return counts


def accuracy(y_true, y_pred):
    """计算整体正确率，同时用 macro_f1 观察各类表现。"""
    correct = 0
    for t, p in zip(y_true, y_pred):
        if t == p:
            correct += 1
    return correct / len(y_true)


def macro_f1(y_true, y_pred):
    """每类各算一个 F1，再取平均。小类别表现差会直接拉低它。"""
    f1_list = []
    labels = set(y_true) | set(y_pred)
    for label in sorted(labels):
        tp = 0
        fp = 0
        fn = 0
        for t, p in zip(y_true, y_pred):
            if t == label and p == label:
                tp += 1
            elif t != label and p == label:
                fp += 1
            elif t == label and p != label:
                fn += 1
        if tp + fp:
            precision = tp / (tp + fp)
        else:
            precision = 0
        if tp + fn:
            recall = tp / (tp + fn)
        else:
            recall = 0
        if precision + recall:
            f1_list.append(2 * precision * recall / (precision + recall))
        else:
            f1_list.append(0)
    return sum(f1_list) / len(f1_list)


def report(y_true, y_pred):
    """打印每个类别的召回率 + 两个总分，返回 {'accuracy': ..., 'macro_f1': ...}"""
    counts = count_labels(y_true)

    print("类别  样本数  召回率")
    for label in sorted(counts, key=counts.get, reverse=True):
        n = counts[label]
        correct = 0
        for t, p in zip(y_true, y_pred):
            if t == label and p == label:
                correct += 1
        print(LABEL2NAME[label], n, f"{correct / n:.4f}")

    result = {"accuracy": accuracy(y_true, y_pred), "macro_f1": macro_f1(y_true, y_pred)}
    print()
    print(f"accuracy = {result['accuracy']:.4f}")
    print(f"macro_f1 = {result['macro_f1']:.4f}")
    return result


INSTRUCTION = (
    "把下面这条新闻标题分到以下 15 个类别之一："
    "故事、文化、娱乐、体育、财经、房产、汽车、教育、科技、军事、旅游、国际、证券、农业、电竞。"
    "只回答类别名。"
)


def sample_rows(rows, size, seed=42):
    """先为每类保留两条，再随机补足；覆盖小类别，同时大致保留原始不平衡比例。"""
    if len(rows) <= size:
        return rows[:]
    rng = random.Random(seed)
    groups = {}
    for row in rows:
        groups.setdefault(row[0], []).append(row)
    if size < sum(min(2, len(group)) for group in groups.values()):
        raise ValueError("样本数太少，无法保证每类至少两条")
    selected, remaining = [], []
    for group in groups.values():
        rng.shuffle(group)
        selected.extend(group[:2])
        remaining.extend(group[2:])
    selected.extend(rng.sample(remaining, size - len(selected)))
    rng.shuffle(selected)
    return selected


def prepare_quick_data():
    """从原 train/dev/test 分别取子集，不混合数据划分；固定种子，每次得到相同数据。"""
    if not all((OUT / f"{split}.tsv").exists() for split in ("train", "dev", "test")):
        prepare_data()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, size in (("train", TRAIN_SAMPLES), ("dev", DEV_SAMPLES), ("test", TEST_SAMPLES)):
        with (OUT / f"{split}.tsv").open(encoding="utf-8") as f:
            rows = list(csv.reader(f, delimiter="\t"))[1:]
        rows = sample_rows(rows, size)
        save_tsv(rows, DATA_DIR / f"{split}.tsv")
        print(f"{split}: {len(rows)} 条，{len(count_labels([r[0] for r in rows]))} 类")


def read_jsonl(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if d["sentence"].strip():
            rows.append((d["label"], d["sentence"].strip()))
    return rows


def save_tsv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["label", "text"])
        w.writerows(rows)


def count_by_name(rows):
    """统计每个中文类别的样本数，返回 [(类别名, 数量)]，按数量从大到小"""
    counts = {}
    for label, text in rows:
        name = LABEL2NAME[label]
        counts[name] = counts.get(name, 0) + 1
    items = list(counts.items())
    items.sort(key=get_count, reverse=True)
    return items


def get_count(item):
    """按样本数量排序。"""
    return item[1]


def prepare_data():
    Path(OUT).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "tnews.zip")
        print("下载中 ...")
        urllib.request.urlretrieve(URL, zip_path)
        zipfile.ZipFile(zip_path).extractall(tmp)
        train = read_jsonl(os.path.join(tmp, "train.json"))
        test = read_jsonl(os.path.join(tmp, "dev.json"))  # 官方 test 无标签，这里用官方 dev 做最终评测

    random.seed(42)
    random.shuffle(train)
    dev = train[:3000]  # 从原训练集留出 3000 条作为开发集
    train = train[3000:]

    for name, rows in [("train", train), ("dev", dev), ("test", test)]:
        save_tsv(rows, os.path.join(OUT, name + ".tsv"))
        print(name, len(rows), count_by_name(rows))


if __name__ == "__main__":
    if not (OUT / "train.tsv").exists():
        prepare_data()
    prepare_quick_data()

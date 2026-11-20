"""
从 HuggingFace 下载 TNEWS 数据集并保存为本地 JSON 文件

教学重点：
  1. 标准数据集的获取与本地化存储方式
  2. 类别映射（label_id → 类别码 → 中文名）的建立
  3. 数据集划分结构（train / val / test）

使用方式：
  python download_data.py

依赖：
  pip install datasets
"""

import json
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path(__file__).parent.parent / "data"

# TNEWS 原始类别码 → 中文类别名
LABEL_CODE_TO_NAME = {
    "100": "故事",
    "101": "文化",
    "102": "娱乐",
    "103": "体育",
    "104": "财经",
    "106": "房产",
    "107": "汽车",
    "108": "教育",
    "109": "科技",
    "110": "军事",
    "112": "旅游",
    "113": "国际",
    "114": "证券",
    "115": "农业",
    "116": "电竞",
}


def build_label_map(features):
    """构建三层映射：label_id(int) ↔ 类别码(str) ↔ 中文名(str)"""
    label_names = features["label"].names  # ['100', '101', ...]
    id2code = {i: code for i, code in enumerate(label_names)}
    id2name = {i: LABEL_CODE_TO_NAME[code] for i, code in id2code.items()}
    return {
        "id2code": id2code,
        "id2name": id2name,
        "code2id": {v: k for k, v in id2code.items()},
        "name2id": {v: k for k, v in id2name.items()},
        "num_labels": len(label_names),
    }


def save_split(dataset_split, path: Path, split_name: str):
    records = []
    for item in dataset_split:
        record = {
            "idx": item["idx"],
            "sentence": item["sentence"],
            "label": item["label"],  # int，-1 表示 test 集无标签
        }
        records.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  {split_name}: {len(records)} 条 → {path}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("正在下载 TNEWS ...")
    ds = load_dataset("clue", "tnews")

    label_map = build_label_map(ds["train"].features)
    label_map_path = DATA_DIR / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"label_map 已保存 → {label_map_path}")
    print(f"类别数：{label_map['num_labels']}")
    for i, name in label_map["id2name"].items():
        print(f"  {i:2d} | {label_map['id2code'][i]} | {name}")

    print("\n正在保存各数据集分割 ...")
    save_split(ds["train"], DATA_DIR / "train.json", "train")
    save_split(ds["validation"], DATA_DIR / "val.json", "val")
    save_split(ds["test"], DATA_DIR / "test.json", "test")

    print("\n下载完成。")


if __name__ == "__main__":
    main()

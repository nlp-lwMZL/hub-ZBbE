"""事件 schema：事件类型 ↔ 槽位（role）定义，以及「由槽位推断事件类型」规则。

真值来源与 build_event_annotation_dataset.py 的 ROLE_SCHEMA 保持一致，
但这里去掉无法做 span 标注的「说明/内容」类兜底槽位。

schema 文件
-----------
`python -m common.schema` 会导出 data/schema/event_schema.json，供人工查看/其它服务复用。
"""

from __future__ import annotations

import json
from pathlib import Path

from .label import ENTITY_ROLES, EXCLUDED_ROLES, TRIGGER_LABEL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "data" / "schema" / "event_schema.json"

# 所有事件都带的公共槽位
COMMON_ROLES: tuple[str, ...] = ("基金名称", "基金主代码", "基金管理人", "公告日期")

# 事件类型 -> 槽位定义。key_roles 为判别该类型最关键的槽位。
EVENT_SCHEMA: dict[str, dict] = {
    "基金经理增聘": {
        "roles": [*COMMON_ROLES, "新任基金经理", "任职日期"],
        "key_roles": ["新任基金经理"],
    },
    "基金经理离任": {
        "roles": [*COMMON_ROLES, "离任基金经理", "离任日期", "离任原因"],
        "key_roles": ["离任基金经理"],
    },
    "基金经理变更": {
        "roles": [*COMMON_ROLES, "原基金经理", "新任基金经理", "生效日期"],
        "key_roles": ["原基金经理", "新任基金经理"],
    },
    "基金经理助理聘任": {
        "roles": [*COMMON_ROLES, "基金经理助理", "聘任日期"],
        "key_roles": ["基金经理助理"],
    },
    "基金经理助理离任": {
        "roles": [*COMMON_ROLES, "离任基金经理助理", "离任日期", "离任原因"],
        "key_roles": ["离任基金经理助理"],
    },
    "基金经理助理变更": {
        "roles": [*COMMON_ROLES, "原基金经理助理", "新任基金经理助理", "生效日期"],
        "key_roles": ["原基金经理助理", "新任基金经理助理"],
    },
    "代为履职": {
        "roles": [*COMMON_ROLES, "原基金经理", "代为履职人", "起始日期"],
        "key_roles": ["代为履职人"],
    },
    "恢复履职": {
        "roles": [*COMMON_ROLES, "恢复履职人", "恢复日期"],
        "key_roles": ["恢复履职人"],
    },
    "其他人事变动": {
        "roles": [*COMMON_ROLES, "变动人员"],
        "key_roles": ["变动人员"],
    },
}

# 推断规则：按顺序匹配，命中 required 全部槽位即判为该类型。
# 顺序很重要——双侧组合、特异槽位优先于单侧槽位。
INFER_RULES: list[tuple[tuple[str, ...], str]] = [
    (("代为履职人",), "代为履职"),
    (("恢复履职人",), "恢复履职"),
    (("原基金经理助理", "新任基金经理助理"), "基金经理助理变更"),
    (("原基金经理", "新任基金经理"), "基金经理变更"),
    (("离任基金经理助理",), "基金经理助理离任"),
    (("原基金经理助理",), "基金经理助理变更"),
    (("新任基金经理助理",), "基金经理助理变更"),
    (("基金经理助理",), "基金经理助理聘任"),
    (("原基金经理",), "基金经理变更"),
    (("新任基金经理",), "基金经理增聘"),
    (("离任基金经理",), "基金经理离任"),
    (("变动人员",), "其他人事变动"),
]

EVENT_TYPES: list[str] = list(EVENT_SCHEMA)


def infer_event_type(base_labels) -> str | None:
    """由抽到的槽位（基标签集合，不含 TRIGGER）推断事件类型。

    先按 INFER_RULES 精确匹配（双侧/特异槽位优先）；全部不中返回 None。
    """
    labels = set(base_labels)
    labels.discard(TRIGGER_LABEL)
    for required, etype in INFER_RULES:
        if all(r in labels for r in required):
            return etype
    return None


def export_schema(path: str | Path | None = None) -> Path:
    """导出可读的 JSON schema。"""
    out = Path(path) if path else SCHEMA_PATH
    payload = {
        "trigger_label": TRIGGER_LABEL,
        "event_types": EVENT_TYPES,
        "roles": list(ENTITY_ROLES),
        "excluded_roles": sorted(EXCLUDED_ROLES),
        "common_roles": list(COMMON_ROLES),
        "schema": {
            etype: {"roles": spec["roles"], "key_roles": spec["key_roles"]}
            for etype, spec in EVENT_SCHEMA.items()
        },
        "infer_rules": [
            {"required": list(req), "type": etype} for req, etype in INFER_RULES
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    p = export_schema()
    print(f"[common.schema] 事件类型 {len(EVENT_TYPES)} 个，槽位 {len(ENTITY_ROLES)} 个")
    print(f"schema 已导出: {p}")

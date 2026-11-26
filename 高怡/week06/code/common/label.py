"""
统一的标签定义，供多分类 / 事件抽取 / BERT 微调等所有任务共用。

公告类型多分类（公告一级类目）
    0 发行运作   1 分红送配   2 定期报告
    3 人事调整   4 基金销售   5 其他公告

人事事件抽取（二级类目，全部归属一级「人事公告」）
    基金经理变更 / 正式增聘基金经理 / 基金经理离任 /
    基金经理助理聘任 / 代为履职 / 恢复履职 / 其他人事变动

实体类型
    fund_name / fund_code / effective_date / resign_date /
    fund_manager / original_fund_manager / acting_manager /
    fund_manager_assistant
"""

from __future__ import annotations

from typing import TypeAlias

LabelId: TypeAlias = int

# ---------------------------------------------------------------- 公告类型多分类

CLS_LABEL2ID: dict[str, LabelId] = {
    "发行运作": 0,
    "分红送配": 1,
    "定期报告": 2,
    "人事调整": 3,
    "基金销售": 4,
    "其他公告": 5,
}
CLS_ID2LABEL: dict[LabelId, str] = {v: k for k, v in CLS_LABEL2ID.items()}
CLS_NUM_LABELS = len(CLS_LABEL2ID)


def cls_label2id(label: str) -> LabelId:
    return CLS_LABEL2ID[label]


def cls_id2label(idx: LabelId) -> str:
    return CLS_ID2LABEL[idx]


# ---------------------------------------------------------------- 人事事件抽取（二级类目）

EVENT_LEVEL1 = "人事公告"

EVENT_LEVEL2_LABEL2ID: dict[str, LabelId] = {
    "基金经理变更": 0,
    "正式增聘基金经理": 1,
    "基金经理离任": 2,
    "基金经理助理聘任": 3,
    "代为履职": 4,
    "恢复履职": 5,
    "其他人事变动": 6,
}
EVENT_LEVEL2_ID2LABEL: dict[LabelId, str] = {
    v: k for k, v in EVENT_LEVEL2_LABEL2ID.items()
}
EVENT_LEVEL2_NUM_LABELS = len(EVENT_LEVEL2_LABEL2ID)


def event_level2_label2id(label: str) -> LabelId:
    return EVENT_LEVEL2_LABEL2ID[label]


def event_level2_id2label(idx: LabelId) -> str:
    return EVENT_LEVEL2_ID2LABEL[idx]


# ---------------------------------------------------------------- 实体类型

ENTITY_TYPES: frozenset[str] = frozenset({
    "fund_name",
    "fund_code",
    "effective_date",
    "resign_date",
    "fund_manager",
    "original_fund_manager",
    "acting_manager",
    "fund_manager_assistant",
})

# NER 用的 BIO 标签序列（含 O，B/I 前缀在训练时按需拼接）
NER_BIO_SCHEMA: list[str] = sorted(
    {t for t in ENTITY_TYPES if not t.startswith("date")}
    | {t.replace("resign_date", "effective_date") for t in ENTITY_TYPES if t == "resign_date"}
)


# ---------------------------------------------------------------- 工具


def assert_labels() -> None:
    """启动时快速校验两套映射的双向一致性。"""
    assert len(CLS_LABEL2ID) == CLS_NUM_LABELS
    assert all(CLS_ID2LABEL[v] == k for k, v in CLS_LABEL2ID.items())
    assert len(EVENT_LEVEL2_LABEL2ID) == EVENT_LEVEL2_NUM_LABELS
    assert all(EVENT_LEVEL2_ID2LABEL[v] == k
               for k, v in EVENT_LEVEL2_LABEL2ID.items())


if __name__ == "__main__":
    assert_labels()
    print("[common.label] 标签校验通过")
    print("公告类型:", CLS_ID2LABEL)
    print("人事事件:", EVENT_LEVEL2_ID2LABEL)
    print("实体类型:", sorted(ENTITY_TYPES))

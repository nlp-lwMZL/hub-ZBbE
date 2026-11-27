"""
统一的标签定义，供多分类 / 事件抽取 / BERT 微调等所有任务共用。

公告类型多分类（公告一级类目）
    0 发行运作   1 分红送配   2 定期报告
    3 人事调整   4 基金销售   5 其他公告

人事事件抽取（二级类目，全部归属一级「人事公告」）
    基金经理变更 / 基金经理增聘 / 基金经理离任 /
    基金经理助理聘任 / 基金经理助理离任 / 基金经理助理变更 /
    代为履职 / 恢复履职 / 其他人事变动

实体类型
    fund_name / fund_code / effective_date / resign_date /
    fund_manager / original_fund_manager / acting_manager /
    fund_manager_assistant

事件抽取（序列标注）
    每个字符打 BIO 标签，基标签 = TRIGGER（触发词）或具体槽位（role），
    标签空间由 ENTITY_ROLES 决定，见 build_bio_labels()。
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
    "基金经理增聘": 1,
    "基金经理离任": 2,
    "基金经理助理聘任": 3,
    "基金经理助理离任": 4,
    "基金经理助理变更": 5,
    "代为履职": 6,
    "恢复履职": 7,
    "其他人事变动": 8,
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


# ---------------------------------------------------------------- 序列标注（事件抽取）标签体系

# 参与序列标注的实体型槽位。
# 「说明/内容」类兜底文本（聘任说明/变动说明/履职说明/变动内容）会覆盖整句、与其它槽位
# 大量重叠，不适合做 span 标注，故不纳入。
ENTITY_ROLES: tuple[str, ...] = (
    "基金名称",
    "基金主代码",
    "基金管理人",
    "新任基金经理",
    "离任基金经理",
    "原基金经理",
    "基金经理助理",
    "原基金经理助理",
    "新任基金经理助理",
    "离任基金经理助理",
    "代为履职人",
    "恢复履职人",
    "变动人员",
    "任职日期",
    "离任日期",
    "生效日期",
    "聘任日期",
    "起始日期",
    "恢复日期",
    "公告日期",
    "离任原因",
)

# 被显式排除、不参与标注的槽位（保留常量便于对照）
EXCLUDED_ROLES: frozenset[str] = frozenset({
    "聘任说明", "变动说明", "履职说明", "变动内容",
})

TRIGGER_LABEL = "TRIGGER"
OUTSIDE_LABEL = "O"

# 序列标注时需要忽略的标签 id（padding 位置）
PAD_LABEL_ID = -100


def build_bio_labels() -> list[str]:
    """构造 BIO 标签表：O + 每个基标签的 B-/I-。"""
    labels = [OUTSIDE_LABEL]
    for base in (TRIGGER_LABEL, *ENTITY_ROLES):
        labels.append(f"B-{base}")
        labels.append(f"I-{base}")
    return labels


BIO_LABELS: list[str] = build_bio_labels()
BIO_LABEL2ID: dict[str, LabelId] = {lab: i for i, lab in enumerate(BIO_LABELS)}
BIO_ID2LABEL: dict[LabelId, str] = {i: lab for lab, i in BIO_LABEL2ID.items()}
NUM_BIO_LABELS: int = len(BIO_LABELS)


def split_bio_label(label: str) -> tuple[str, str]:
    """'B-基金名称' -> ('B', '基金名称'); 'O' -> ('O', '')。"""
    if label == OUTSIDE_LABEL:
        return OUTSIDE_LABEL, ""
    prefix, _, base = label.partition("-")
    return prefix, base


def assert_labels() -> None:
    """启动时快速校验两套映射的双向一致性。"""
    assert len(CLS_LABEL2ID) == CLS_NUM_LABELS
    assert all(CLS_ID2LABEL[v] == k for k, v in CLS_LABEL2ID.items())
    assert len(EVENT_LEVEL2_LABEL2ID) == EVENT_LEVEL2_NUM_LABELS
    assert all(EVENT_LEVEL2_ID2LABEL[v] == k
               for k, v in EVENT_LEVEL2_LABEL2ID.items())
    assert len(BIO_LABELS) == NUM_BIO_LABELS
    assert all(BIO_ID2LABEL[v] == k for k, v in BIO_LABEL2ID.items())
    assert len(BIO_LABELS) == 1 + 2 * (1 + len(ENTITY_ROLES))


if __name__ == "__main__":
    assert_labels()
    print("[common.label] 标签校验通过")
    print("公告类型:", CLS_ID2LABEL)
    print("人事事件:", EVENT_LEVEL2_ID2LABEL)
    print("实体类型:", sorted(ENTITY_TYPES))
    print(f"BIO 标签数: {NUM_BIO_LABELS}（槽位 {len(ENTITY_ROLES)} + trigger）")
    print("槽位:", ENTITY_ROLES)

"""
标签体系与投研口径映射（全项目共用）

教学重点：
  1. 公开数据集是「替代数据」——TNEWS 的 15 个新闻类别，和投研关心的主题不是一回事；
     所以第一步不是训练模型，而是**定义标签口径**：把 15 类重映射成 4 组 + 相关性二分类。
  2. 相关性标签由规则派生（金融/产业 = 相关），**零人工标注**——
     这解释了为什么「换个领域做分类」往往不需要重新标数据，改映射表就行。
  3. 标签口径属于业务定义，写在代码里而不是模型里：换领域只改这张表。

为什么单独成文件：
  规则法、TF-IDF、BERT、LLM 四路方法的输出标签必须完全一致，
  否则同一个 0.53 的准确率在不同脚本里含义都不同，无法横向比较。
"""

# TNEWS 原始 15 类（顺序来自官方 label_map.json 的 name2id，务必保持一致）
LABELS = [
    "故事", "文化", "娱乐", "体育", "财经",
    "房产", "汽车", "教育", "科技", "军事",
    "旅游", "国际", "证券", "农业", "电竞",
]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

# 投研关注的 4 组主题：把 15 个新闻类别收拢成业务看得懂的 4 个桶
GROUP_MAP = {
    "财经": "金融", "证券": "金融", "房产": "金融",
    "科技": "产业", "汽车": "产业", "农业": "产业",
    "国际": "宏观", "军事": "宏观",
    "教育": "噪音", "娱乐": "噪音", "体育": "噪音",
    "电竞": "噪音", "文化": "噪音", "故事": "噪音", "旅游": "噪音",
}
GROUPS = ["金融", "产业", "宏观", "噪音"]

# 相关性：金融 + 产业 才是投研要看的；宏观弱相关；噪音直接丢
# 这里用集合表示，而不是 if/else 链——加一档只改一个集合
RELEVANT_GROUPS = {"金融", "产业"}


def to_group(label_id: int) -> str:
    """15 类 id → 4 组主题"""
    return GROUP_MAP[LABELS[label_id]]


def is_relevant(label_id: int) -> bool:
    """15 类 id → 是否与投研相关（由主题组派生，不需要额外标注）"""
    return to_group(label_id) in RELEVANT_GROUPS


def queue_name(label_id: int, conf: float, conf_threshold: float = 0.8) -> str:
    """
    分流结果落到哪个队列——这是「分类」变成「动作」的地方。

    设计要点：
      - 不相关 → drop（噪音，不进投研视野）
      - 相关但置信度低 → review（转 LLM 兜底 / 人工），宁可多一步也不要把不确定的推给人
      - 相关且高置信 → research（进研究队列）
    阈值 0.8 是默认值，实际要用真实 query 校准（见 service/classify_news.py 的说明）。
    """
    if not is_relevant(label_id):
        return "drop"
    if conf < conf_threshold:
        return "review"
    return "research"

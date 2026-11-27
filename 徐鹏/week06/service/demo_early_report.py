"""
早报面板 Demo：模拟「每天几百条消息进来，先分流再给你看」

教学重点：
  1. 分流做完不是结束——产物要变成**动作**：哪些进研究队列、哪些转人工、哪些直接丢。
  2. 同一个服务，喂一批假数据就能演示产品形态（不用等真实接入）。
  3. 注意「review」这一档：低置信的结果不会被丢掉，而是转 LLM 兜底/人工复核，
     宁可多一步，也不要把不确定的结论推给研究员。

用法：
  python service/demo_early_report.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service.classify_news import ShuntService

# 模拟一早晨收到的消息：混合了财经、证券、产业、噪音，还有两条规则能直接命中的公告
MESSAGES = [
    "贵州茅台：2023 年年度业绩预告，预计营收同比增长 17%",
    "宁德时代拟回购不超过 50 亿元公司股份",
    "多家券商上调半导体板块评级",
    "比亚迪 11 月新能源车销量同比增长 32%",
    "央行开展 5000 亿元 MLF 操作",
    "沪指午后跳水，创业板跌超 2%",
    "某公司发布减持计划，控股股东拟减持 3%",
    "国家统计局：11 月 CPI 同比上涨 0.6%",
    "华为发布新款折叠屏手机",
    "教育部：进一步减轻学生作业负担",
    "国足世预赛客场 1:3 失利",
    "LPL 春季赛决赛今晚开打",
    "《繁花》大结局收视破纪录",
    "春节档电影预售票房破 10 亿",
    "某地降温 10 度，多地发布寒潮预警",
    "旅行社推出春节出境游线路",
    "国际油价连续三日下跌",
    "美联储宣布维持利率不变",
    "某上市公司被出具保留意见审计报告",
    "小产权房交易风险提示",
]


def main():
    svc = ShuntService()
    rows = [svc.classify(m) for m in MESSAGES]

    buckets = {"research": [], "review": [], "drop": []}
    for r in rows:
        buckets[r["queue"]].append(r)

    def show(title, items, note):
        print(f"\n{'='*58}\n{title}（{len(items)} 条）　{note}\n{'='*58}")
        for r in items:
            print(f"  [{r['topic']}/{r['sub_topic']:<3}] conf={r['conf']:.2f} "
                  f"via {r['model']:<4} | {r['title'][:34]}")
        if not items:
            print("  （空）")

    show("① 研究队列 research", buckets["research"], "高置信 + 与投研相关 → 直接推给研究员")
    show("② 待复核 review", buckets["review"], "置信度不足 → 转 LLM 兜底或人工")
    show("③ 丢弃 drop", buckets["drop"], "噪音类 → 不进投研视野")

    total = len(rows)
    print(f"\n合计 {total} 条：research {len(buckets['research'])} / "
          f"review {len(buckets['review'])} / drop {len(buckets['drop'])}")
    print("规则命中：", ", ".join(r["title"][:18] for r in rows if r["model"] == "rule") or "无")


if __name__ == "__main__":
    main()

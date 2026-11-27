"""
长句滑动窗口推理 —— 解决 cdtier 句长 max 2232 超过 max_length 被截断、尾部实体全丢的问题。

为什么需要（老师 PPT 明确讲过）：
  clean 数据 token 长度 p50≈66，但 max 到 2232（train）/ 2857（val）。
  max_length=256 时这些长句被硬截断，后面的实体直接丢失——不是模型差，是根本没喂进去。
  滑动窗口把长句切成带重叠的短窗，每窗独立推理，再把 span 映射回原句坐标并合并。

【坐标系】clean 数据 tokens 逐字，text = "".join(tokens)，所以 **token 下标 == 字符下标**。
  每个窗口是 tokens 的一个 [start, end) 切片，窗口内推理出的 span 坐标是窗口局部的，
  映射回原句：global_start = win_start + local_start。

【窗口切分】window_size = max_length - 2（给 CLS/SEP 留位），overlap 默认 32。
  stride = window_size - overlap。最后一个窗口若不足 window_size，仍保留（贴到右边界），
  保证句尾实体一定被某个完整窗口覆盖。

【span 合并】来自不同窗口的 span 映射到全局后：
  1. 完全相同的 (type, start, end) 去重；
  2. 同类型、高度重叠（IoU ≥ 0.5）的 span 保留较长者——overlap 区里同一实体
     会在相邻窗口各出现一次，且其中一个可能被窗口边界截断成较短版本。

用法（判别式 evaluate.py 集成）：
  windows = make_windows(len(tokens), window_size, overlap)
  for (s, e) in windows:
      用 tokens[s:e] 造一个临时样本 -> 模型推理 -> 局部 span
      spans.append({type, start: s+local.start, end: s+local.end, text: text[s+local.start:s+local.end]})
  pred_by_id[id] = merge_spans(spans)

自检：python common/sliding_window.py
"""
from __future__ import annotations

from typing import Any, Iterable

__all__ = ["make_windows", "merge_spans", "span_iou"]


def make_windows(n: int, window_size: int, overlap: int = 32) -> list[tuple[int, int]]:
    """把长度 n 的序列切成带重叠的窗口 [start, end)。

    - window_size 必须 > overlap，否则 stride <= 0 会死循环；
    - 短句（n <= window_size）直接返回单窗口 [(0, n)]，不做滑窗；
    - 最后一个窗口若剩余不足 window_size，把起点左移到 n-window_size，
      确保句尾被一个「满长」窗口覆盖，而不是一个很短的尾巴窗。
    """
    if n <= 0:
        return []
    if window_size <= 0:
        raise ValueError(f"window_size 必须为正，实际 {window_size}")
    if overlap < 0 or overlap >= window_size:
        raise ValueError(f"overlap 需满足 0 <= overlap < window_size，实际 overlap={overlap} window_size={window_size}")
    if n <= window_size:
        return [(0, n)]
    stride = window_size - overlap
    windows: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + window_size, n)
        windows.append((start, end))
        if end == n:
            break
        next_start = start + stride
        # 剩余不足一个满窗时，最后一个窗口贴右边界（满长），避免短尾窗
        if n - next_start < window_size:
            next_start = n - window_size
        start = next_start
    return windows


def span_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    """同类型 span 的字符级 IoU（假设已先按 type 过滤）。"""
    s1, e1 = int(a["start"]), int(a["end"])
    s2, e2 = int(b["start"]), int(b["end"])
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def merge_spans(spans: Iterable[dict[str, Any]], iou_threshold: float = 0.5) -> list[dict[str, Any]]:
    """把来自多个窗口、已映射到全局坐标的 span 去重合并。

    1) (type, start, end) 完全相同 -> 去重；
    2) 同类型 IoU >= 阈值 -> 保留较长者（窗口边界截断的短版本被丢弃）。

    返回按 (start, end) 排序的列表。
    """
    # 去重
    seen: dict[tuple[str, int, int], dict[str, Any]] = {}
    for s in spans:
        key = (s["type"], int(s["start"]), int(s["end"]))
        if key not in seen:
            seen[key] = dict(s)
    items = list(seen.values())

    # 按类型分组，组内做 IoU 合并（保留长 span）
    by_type: dict[str, list[dict[str, Any]]] = {}
    for s in items:
        by_type.setdefault(s["type"], []).append(s)

    merged: list[dict[str, Any]] = []
    for etype, group in by_type.items():
        group.sort(key=lambda x: (int(x["start"]), -(int(x["end"]) - int(x["start"]))))
        keep: list[dict[str, Any]] = []
        for s in group:
            absorbed = False
            for i, k in enumerate(keep):
                if span_iou(s, k) >= iou_threshold:
                    # 保留较长者；等长保留先出现的
                    if (int(s["end"]) - int(s["start"])) > (int(k["end"]) - int(k["start"])):
                        keep[i] = s
                    absorbed = True
                    break
            if not absorbed:
                keep.append(s)
        merged.extend(keep)

    merged.sort(key=lambda x: (int(x["start"]), int(x["end"])))
    return merged


def _self_test() -> None:
    print("=" * 68)
    print("common/sliding_window 自检")
    print("=" * 68)

    # 窗口切分：短句单窗
    assert make_windows(10, 256, 32) == [(0, 10)]
    # 长句：覆盖完整、无空隙、贴右边界
    ws = make_windows(100, 32, 8)
    print(f"  n=100 window=32 overlap=8 -> {len(ws)} 窗: {ws[:3]}...{ws[-2:]}")
    assert ws[0][0] == 0 and ws[-1][1] == 100
    for (s1, e1), (s2, e2) in zip(ws, ws[1:]):
        assert s2 < e1, "窗口间必须有重叠"        # 重叠
        assert s2 <= e1, "窗口间不能有空隙"
    # 每处字符至少被一个窗口覆盖
    covered = set()
    for s, e in ws:
        covered.update(range(s, e))
    assert covered == set(range(100)), "有字符没被任何窗口覆盖"

    # span 合并：完全重复去重
    spans = [
        {"type": "Attacker", "start": 10, "end": 18, "text": "APT-C-39"},
        {"type": "Attacker", "start": 10, "end": 18, "text": "APT-C-39"},
    ]
    m = merge_spans(spans)
    print(f"  完全重复 2 -> {len(m)}（应=1）")
    assert len(m) == 1

    # overlap 截断：同类型高度重叠，保留长的
    spans = [
        {"type": "Tools", "start": 20, "end": 24, "text": "Houd"},      # 被窗口截断的短版
        {"type": "Tools", "start": 20, "end": 27, "text": "HoudiniRAT"},  # 完整版
    ]
    m = merge_spans(spans)
    print(f"  高度重叠 2 -> {len(m)}，保留: {m[0]['text']}（应=HoudiniRAT）")
    assert len(m) == 1 and m[0]["end"] == 27

    # 不同类型不合并（同位置不同 entity type）
    spans = [
        {"type": "Region", "start": 5, "end": 7, "text": "印度"},
        {"type": "Industry", "start": 5, "end": 7, "text": "印度"},
    ]
    m = merge_spans(spans)
    print(f"  同位置不同类型 -> {len(m)}（应=2）")
    assert len(m) == 2

    # 全局坐标映射示例
    windows = make_windows(100, 32, 8)
    w_start, _ = windows[1]
    local_span = {"type": "Region", "start": 3, "end": 5, "text": "美国"}
    global_span = {**local_span, "start": w_start + local_span["start"],
                   "end": w_start + local_span["end"]}
    print(f"  局部 span(3,5) @窗口起点{w_start} -> 全局({global_span['start']},{global_span['end']})")
    assert global_span["start"] >= 0 and global_span["end"] <= 100

    print("自检通过 ✓")


if __name__ == "__main__":
    _self_test()

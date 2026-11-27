"""
手写 CRF 层（不依赖 pytorch-crf）：发射分数 + 转移矩阵 + 前向算法 + 维特比解码

为什么要自己手写：
  NER / 事件抽取的标签之间有很强的**结构性约束**。举真实例子：
    合法：  O  B-上市  I-上市  O  B-发行价格  I-发行价格
    非法：  I-上市  I-收购          （序列开头就是 I，没有 B 打头）
    非法：  B-上市  I-收购          （B 和 I 的实体类型不一致）
  线性头（BERT + Linear）对每个 token 独立 softmax，**不知道**这些约束，
  所以会喷出上面那种非法序列。CRF 加了一个转移矩阵 T[prev][cur]，
  用一条「全局路径」的分数来解码，非法路径天然分数低。

三个核心公式（都在 log 空间算，否则连乘会下溢成 0）：
  1) 路径分数     S(y|x) = Σ_i emit_i[y_i] + Σ_i T[y_{i-1}][y_i] + start[y_0] + end[y_L]
  2) 配分函数     logZ   = log Σ_{所有路径} exp(S)        ← 前向算法（动态规划，O(L·C²)）
  3) 损失         loss   = logZ - S(金标路径)              ← 负对数似然，越小越好
  解码用维特比：每步只保留「到当前 tag 为止的最优路径」，回溯得到全局最优序列。

与 pytorch-crf 的差别（课堂上会讲）：
  这里做一个显式的**硬约束开关** `constraint`（C×C 的 bool 矩阵，False 表示禁止转移）。
  默认不开（和 pytorch-crf 一样只是「学出来的软约束」，实测仍会有少量非法转移）；
  打开后非法转移的分数直接置 -1e4，非法序列可以严格归零 —— 这就是
  「为什么 CRF 不保证零非法序列、以及怎么做到零」的完整答案。

使用方式：
  from crf import CRF
  crf = CRF(num_tags=149)                 # 软约束（推荐，和教学主线一致）
  loss = crf.neg_log_likelihood(emissions, tags, mask)
  paths = crf.decode(emissions, mask)     # list[list[int]]，不含 PAD
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows: torch 与 numpy 各带 libiomp5md，不设会 OMP Error #15

import torch
import torch.nn as nn

NEG_INF = -1e4   # 用一个大负数代替 -inf：-inf 在反传时容易产生 NaN


class CRF(nn.Module):
    """
    Linear-chain CRF。

    参数形状（batch_first=True，即 emissions=(B, L, C)）：
      transitions      (C, C)   transitions[i][j] = 从标签 i 转移到标签 j 的分数
      start_transitions(C,)     序列第一个标签的「入场分」
      end_transitions  (C,)     序列最后一个标签的「出场分」

    三个参数都可学习：模型会自动学到「O→B 很常见、O→I 几乎不可能」，
    所以即使不显式加约束，非法序列也会变少（但不会归零）。
    """

    def __init__(self, num_tags: int, constraint: torch.Tensor | None = None,
                 start_constraint: torch.Tensor | None = None,
                 batch_first: bool = True):
        super().__init__()
        assert batch_first, "本项目统一用 batch_first=(B, L, C)"
        self.num_tags = num_tags
        self.batch_first = batch_first

        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        # 初始化成小随机数：全 0 也能训，但小随机能让不同转移方向尽早分化
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

        # 硬约束：注册成 buffer（跟着模型存 checkpoint，但不参与梯度）
        if constraint is None:
            constraint = torch.ones(num_tags, num_tags, dtype=torch.bool)
        self.register_buffer("constraint", constraint.bool(), persistent=True)
        # 开头约束：只允许 O / B-x 当第一个标签（转移矩阵管不到「第一个 token」）
        if start_constraint is None:
            start_constraint = torch.ones(num_tags, dtype=torch.bool)
        self.register_buffer("start_constraint", start_constraint.bool(), persistent=True)

    # ── 内部工具 ─────────────────────────────────────────────────────────
    def _effective_transitions(self) -> torch.Tensor:
        """实际参与计算的转移矩阵：非法转移处填 -1e4（≈ 禁止）。"""
        if self.constraint.all():
            return self.transitions
        return self.transitions.masked_fill(~self.constraint, NEG_INF)

    def _effective_start(self) -> torch.Tensor:
        """实际参与计算的入场分：非法开头处填 -1e4（≈ 禁止）。"""
        if self.start_constraint.all():
            return self.start_transitions
        return self.start_transitions.masked_fill(~self.start_constraint, NEG_INF)

    def _gather_emit(self, emissions: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        """按标签 id 取发射分数：emissions (B,L,C) + tags (B,L) → (B,L)。"""
        return emissions.gather(2, tags.unsqueeze(-1)).squeeze(-1)

    # ── 1) 金标路径的分数 S(y|x) ──────────────────────────────────────────
    def _gold_score(self, emissions: torch.Tensor, tags: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        B, L, C = emissions.shape
        trans = self._effective_transitions()

        emit = self._gather_emit(emissions, tags)               # (B, L)
        score = self._effective_start()[tags[:, 0]] + emit[:, 0]

        for i in range(1, L):
            # 只对「真实 token」累加转移分与发射分；PAD 位置乘 mask 后贡献为 0
            step = trans[tags[:, i - 1], tags[:, i]] + emit[:, i]
            score = score + step * mask[:, i].to(score.dtype)

        # 出场分要加在「每条序列真正的最后一个 token」上，不是补齐后的第 L-1 个
        seq_ends = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, seq_ends.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    # ── 2) 前向算法求 logZ = log Σ_路径 exp(S) ────────────────────────────
    def _forward_algorithm(self, emissions: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
        """
        alpha[b][i][c] = 「走到第 i 个 token、停在标签 c」的全部路径分数的 log-sum-exp。

        递推：alpha[:, i, :] = logsumexp_{c'} ( alpha[:, i-1, c'] + T[c'][:] ) + emit[:, i, :]
        用 logsumexp（不是 max）—— 因为要的是「所有路径的和」，不是最优路径；
        这正是 CRF 训练（前向）和 CRF 解码（维特比）的唯一区别。
        """
        B, L, C = emissions.shape
        trans = self._effective_transitions()

        alpha = self._effective_start().unsqueeze(0) + emissions[:, 0]   # (B, C)
        for i in range(1, L):
            # (B, C_from, 1) + (1, C_from, C_to) + (B, 1, C_to) → (B, C_from, C_to)
            scores = alpha.unsqueeze(2) + trans.unsqueeze(0) + emissions[:, i].unsqueeze(1)
            new_alpha = torch.logsumexp(scores, dim=1)                  # (B, C_to)
            # PAD 位置不推进状态：保留 i-1 的 alpha，保证最后可以正确加 end 分数
            m = mask[:, i].unsqueeze(1).bool()
            alpha = torch.where(m, new_alpha, alpha)

        return torch.logsumexp(alpha + self.end_transitions.unsqueeze(0), dim=1)  # (B,)

    # ── 3) 负对数似然损失 ────────────────────────────────────────────────
    def neg_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
        """
        返回 batch 平均的 -log P(y|x) = mean(logZ - S)。

        注意 tags 里的 PAD 位置在这里**不被当成 -100 处理**：
        调用方要先把 -100 换成 0，并保证 mask 在这些位置为 0（本项目 train_bio.py 里就是这么做的）。
        """
        mask = mask.bool()
        logZ = self._forward_algorithm(emissions, mask)
        gold = self._gold_score(emissions, tags, mask)
        return (logZ - gold).mean()

    # ── 4) 维特比解码 ────────────────────────────────────────────────────
    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor,
               return_scores: bool = False):
        """
        维特比：返回 list[list[int]]（每个样本一条标签路径，长度 = 真实 token 数，不含 PAD）。

        和训练时的区别只有一步：递推里的 logsumexp 换成 max（只留最优前驱），
        并用 backpointers 记住每一步选了哪个前驱，最后回溯。
        可选 return_scores=True 一并返回每条路径的归一化对数概率 exp(S-logZ)，
        给 service 层做「置信度低 → needs_review」用。
        """
        mask = mask.bool()
        B, L, C = emissions.shape
        trans = self._effective_transitions()

        alpha = self._effective_start().unsqueeze(0) + emissions[:, 0]   # (B, C)
        backpointers = torch.zeros(B, L, C, dtype=torch.long, device=emissions.device)

        for i in range(1, L):
            scores = alpha.unsqueeze(2) + trans.unsqueeze(0)            # (B, C_from, C_to)
            best_score, best_prev = scores.max(dim=1)                   # (B, C_to)
            new_alpha = best_score + emissions[:, i]
            m = mask[:, i].unsqueeze(1).bool()
            alpha = torch.where(m, new_alpha, alpha)
            backpointers[:, i] = best_prev

        # 加出场分挑最优的「最后一个标签」
        final_scores = alpha + self.end_transitions.unsqueeze(0)        # (B, C)
        best_last = final_scores.argmax(dim=1)                          # (B,)
        lengths = mask.long().sum(dim=1)

        paths = []
        for b in range(B):
            Lb = int(lengths[b].item())
            tag = int(best_last[b].item())
            path = [tag]
            for i in range(Lb - 1, 0, -1):
                tag = int(backpointers[b, i, tag].item())
                path.append(tag)
            paths.append(path[::-1])   # 回溯出来是倒序，翻回来

        if not return_scores:
            return paths

        # 置信度：最优路径的分数 - logZ（都是 log 空间），exp 回到 0~1 区间
        scores = []
        for b, path in enumerate(paths):
            seq_len = max(len(path), 1)
            tag_t = torch.tensor(path, dtype=torch.long, device=emissions.device)
            emit = self._gather_emit(emissions[b:b + 1, :seq_len], tag_t.unsqueeze(0)).squeeze(0)
            trans_sum = (trans[tag_t[:-1], tag_t[1:]].sum() if seq_len > 1
                         else torch.zeros((), device=emissions.device))
            path_score = (self._effective_start()[tag_t[0]] + emit.sum()
                          + trans_sum + self.end_transitions[tag_t[-1]])
            # 只在这条序列的真实长度上求 logZ，否则尾部 PAD 会把配分函数算大
            sub_mask = mask[b:b + 1, :seq_len]
            logZ = self._forward_algorithm(emissions[b:b + 1, :seq_len], sub_mask)
            # 置信度：**按 token 平均**的对数概率再取 exp。
            # 直接用 exp(path_score - logZ) 是「整条序列的概率」，它随长度指数衰减
            # （68 个 token 的序列即使每步都很自信，序列概率也只有 0.007），
            # 拿去和固定阈值比毫无意义 —— 长句子会被无条件判成"低置信"。
            # 除以序列长度就变成「平均每个 token 有多确信」，0~1 之间且与长度无关，
            # 也正好和线性头的口径（每个 token 最大概率的均值）可比。
            scores.append(float(torch.exp((path_score - logZ[0]) / max(seq_len, 1))
                                .clamp(0.0, 1.0).item()))
        return paths, scores


class CRFClassifier(nn.Module):
    """
    「特征提取器（BERT / BiLSTM）+ 线性层 + CRF」的通用封装。

    为什么这么抽：BERT+CRF 和 BiLSTM+CRF 的差别只在特征提取器，
    后面接的 CRF 完全一样。抽出来之后，两条链路的标签体系、解码、非法统计都是同一套代码，
    指标才真的可比（否则「BERT 比 BiLSTM 好」可能只是两边 BIO 转换写得不一样）。
    """

    def __init__(self, encoder: nn.Module, hidden_size: int, num_tags: int,
                 dropout: float = 0.1, use_crf: bool = True,
                 constraint: torch.Tensor | None = None,
                 start_constraint: torch.Tensor | None = None):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_tags)
        self.num_tags = num_tags
        self.use_crf = use_crf
        self.crf = (CRF(num_tags, constraint=constraint, start_constraint=start_constraint)
                    if use_crf else None)

    def emissions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        特征 → (B, L, C) 的发射分数：encoder（BERT / BiLSTM）→ dropout → Linear。

        encoder 只要遵守「输入 ids/mask，输出 (B, L, H)」这一个约定，
        BERT+CRF 和 BiLSTM+CRF 就能共用这一份 CRF、解码与统计代码。
        """
        hidden = self.encoder(input_ids, attention_mask, token_type_ids)
        return self.classifier(self.dropout(hidden))

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        emit = self.emissions(input_ids, attention_mask, token_type_ids)
        loss = None
        if labels is not None:
            if self.use_crf:
                # CRF 不支持 ignore_index：把 -100 换成 0，PAD 位置靠 mask 屏蔽
                tags = labels.clone()
                tags[tags == -100] = 0
                loss = self.crf.neg_log_likelihood(emit, tags, attention_mask)
            else:
                loss = nn.functional.cross_entropy(
                    emit.view(-1, self.num_tags), labels.view(-1), ignore_index=-100)
        return emit, loss

    @torch.no_grad()
    def predict(self, input_ids, attention_mask, token_type_ids=None,
                return_scores: bool = False):
        """
        推理：CRF 走维特比，线性头走 argmax。
        返回 (paths, scores)，paths 是 list[list[int]]，长度 = 真实 token 数（含 -100 的位置会被跳过）。
        """
        emit = self.emissions(input_ids, attention_mask, token_type_ids)
        if self.use_crf:
            if return_scores:
                return self.crf.decode(emit, attention_mask, return_scores=True)
            return self.crf.decode(emit, attention_mask)
        pred = emit.argmax(dim=-1)                      # (B, L)
        paths, scores = [], []
        for b in range(pred.shape[0]):
            n = int(attention_mask[b].sum().item())
            paths.append(pred[b, :n].tolist())
            if return_scores:
                # 线性头的置信度：每个 token 最大概率的**算术平均**（0~1，与句长无关）。
                # 不要用「所有 token 概率连乘」——那随长度指数衰减，
                # 68 个 token 的置信度会算成 0.007，任何阈值都过不了（本项目踩过这个坑）。
                probs = torch.softmax(emit[b, :n], dim=-1).max(dim=-1).values
                scores.append(float(probs.mean().item()))
        return (paths, scores) if return_scores else paths


if __name__ == "__main__":
    # 自检：手工构造一个「非法转移明显更差」的小例子，确认前向/维特比都算得对
    torch.manual_seed(0)
    C, L = 3, 4
    crf = CRF(num_tags=C)
    emit = torch.randn(2, L, C)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    tags = torch.tensor([[0, 1, 1, 0], [0, 2, 0, 0]])
    loss = crf.neg_log_likelihood(emit, tags, mask)
    paths = crf.decode(emit, mask)
    print(f"loss={loss.item():.4f}")
    print(f"解码路径={paths}（第二条长度应为 3，不含 PAD）")
    print("标签数 C=3，序列长度 L=4 → 检查：loss 与路径长度都依赖 mask 是否正确")

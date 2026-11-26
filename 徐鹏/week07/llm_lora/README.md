# llm_lora/ —— LoRA / 全量微调（生成式链路）

**一句话**：用 LoRA 在 Qwen3-0.6B 上做 **SFT**（指令微调），把「读公告 → 输出事件 JSON」学成条件生成。
它回答的问题是：**小模型 + 一次 LoRA，能不能打赢精细设计的判别式模型？**

```bash
python llm_lora/train_sft.py --epochs 1 --num_train 3500        # LoRA（默认），约 48 分钟
python llm_lora/train_sft.py --max_steps 20                     # 冒烟
python llm_lora/train_sft.py --full_ft --lr 2e-5                # 全量微调（8GB 卡会 OOM）
python llm_lora/evaluate_sft.py --n 200                         # 评测 n=200（与 API/判别式同批）
python llm_lora/evaluate_sft.py --n 930                         # 全量 dev
```

| 文件 / 产物 | 说明 |
|---|---|
| `train_sft.py` | SFT 训练（loss masking：只在目标段算 loss；LoRA/全量开关） |
| `evaluate_sft.py` | 自动识别 LoRA/全量 checkpoint，生成 + 统一评测 + 对比表 |
| `model_paths.py` | 定位本地 `Qwen3-0.6B` |
| `outputs/llm/ckpt/sft_adapter/` | LoRA adapter（约 9MB） |
| `outputs/logs/llm_lora_train.json` / `_steps.jsonl` / `_console.log` | 训练日志三类（与判别式同格式） |
| `outputs/logs/llm_lora_eval_eval.json` | 评测日志 |

**教学重点**
1. **loss masking**：prompt 部分 labels 全设 -100，只在「目标 JSON」上算 loss（否则模型在学抄题目）。
2. **只算尾部 logits**：Qwen 词表 151936，全序列算 logits 会爆显存；只对目标段算 → batch 4 + 梯度检查点才能塞进 8GB。
3. **1 epoch 就够**：3500 条 × 1 epoch（233 个优化器步）→ trigger 0.91 / argument 0.70，已超过云端大模型 zero/few-shot。
4. 代价是延迟：生成式 **3.13 s/条** vs 判别式 **0.04 s/句** —— PPT 上的「精度 vs 延迟」天平。

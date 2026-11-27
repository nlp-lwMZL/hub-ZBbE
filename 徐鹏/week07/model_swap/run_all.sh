#!/usr/bin/env bash
# 一键复现 week07「换模型」LLM 两路（Qwen2-0.5B / Qwen2.5-0.5B-Instruct）
#
# 用法（Git Bash）：
#   bash model_swap/run_all.sh
#
# 注意：
#   - 串行执行，不要同时开两个 GPU 任务（8GB 卡会 OOM）；
#   - 全流程约 50~70 分钟，取决于显卡和生成速度；
#   - 判别式/规则的结果已在 outputs/metrics/disc/ 里，本脚本不重训 BERT。
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
QWEN2="D:/workspace/nlp_study/pretrain_models/Qwen2-0.5B"
QWEN25="D:/workspace/nlp_study/pretrain_models/Qwen2.5-0.5B-Instruct"

echo "== 0) 环境自检"
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "== 1) Qwen2-0.5B zero-shot n=60（base 模型，预期失败）"
python llm_zero_shot/llm_extract.py --backend local --mode zero --n 60 \
    --model_path "$QWEN2" --tag qwen2_zero_n60 --gen_batch 8 --max_new_tokens 384

echo "== 2) Qwen2-0.5B LoRA 训练（3500 条 × 1 epoch）"
python llm_lora/train_sft.py --model_path "$QWEN2" \
    --num_train 3500 --epochs 1 --batch_size 3 --grad_accum 5 \
    --max_length 1280 --grad_ckpt --dtype bf16 \
    --output_dir outputs/llm/ckpt_qwen2 --run_name qwen2_lora

echo "== 3) Qwen2-0.5B LoRA 评测 n=200"
python llm_lora/evaluate_sft.py --model_path "$QWEN2" \
    --ckpt_dir outputs/llm/ckpt_qwen2/sft_adapter \
    --n 200 --tag qwen2_lora_n200 --gen_batch 8

echo "== 4) Qwen2.5-0.5B-Instruct zero-shot n=200"
python llm_zero_shot/llm_extract.py --backend local --mode zero --n 200 \
    --model_path "$QWEN25" --tag qwen25_zero_n200 --gen_batch 8 --max_new_tokens 384
# n=60 是 n=200 的前缀子集：直接切预测，省一次推理
python model_swap/derive_subset_eval.py \
    --pred outputs/llm/pred_qwen25_zero_n200.jsonl \
    --gold outputs/llm/gold_n60.jsonl \
    --out outputs/metrics/llm/qwen25_zero_n60.json

echo "== 5) Qwen2.5-0.5B-Instruct LoRA 训练（同配置）"
python llm_lora/train_sft.py --model_path "$QWEN25" \
    --num_train 3500 --epochs 1 --batch_size 3 --grad_accum 5 \
    --max_length 1280 --grad_ckpt --dtype bf16 \
    --output_dir outputs/llm/ckpt_qwen25 --run_name qwen25_lora

echo "== 6) Qwen2.5-0.5B-Instruct LoRA 评测 n=200"
python llm_lora/evaluate_sft.py --model_path "$QWEN25" \
    --ckpt_dir outputs/llm/ckpt_qwen25/sft_adapter \
    --n 200 --tag qwen25_lora_n200 --gen_batch 8

echo "== 7) 汇总对比 + 分类型拆解"
python model_swap/compare_models.py --bootstrap 1000
python model_swap/per_type_analysis.py

echo "完成：outputs/model_swap/REPORT.md"

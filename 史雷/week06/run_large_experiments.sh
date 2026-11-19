#!/bin/bash
# bert-large-chinese 对比实验队列（MPS，训练集 1 万条 × 2 epoch，串行跑）
cd "$(dirname "$0")"
COMMON="--num_train 10000 --epochs 2 --batch_size 32 --max_eval 2000"
run() {  # run <tag> <extra train args...>
  tag=$1; shift
  echo "=== [$tag] train $(date +%T)"
  python bert/train_bert.py $COMMON --tag $tag "$@" > outputs/log_$tag.txt 2>&1 || { echo "[$tag] train failed"; return; }
  python bert/predict_bert.py --ckpt outputs/bert_ckpt_$tag.pt --n 500 --out outputs/bert_pred_$tag.json >> outputs/log_$tag.txt 2>&1
  rm -f outputs/bert_ckpt_$tag.pt     # 单个 1.3 GB，预测写完就删
  echo "=== [$tag] done $(date +%T)"
}
run large_cls  --pool cls
run large_mean --pool mean
run large_max  --pool max
run large_cls_nocw --pool cls --no_class_weight
echo ALL_DONE

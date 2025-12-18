#!/bin/bash
# Causal-HAFE 快速运行脚本

echo "========================================"
echo "  Causal-HAFE 快速测试脚本"
echo "========================================"

# 检查数据集
if [ ! -f "./data/semeval2014/Restaurants_Train.xml" ]; then
    echo "错误: 数据集未找到!"
    echo "请先下载SemEval-2014数据集到 ./data/semeval2014/"
    exit 1
fi

echo ""
echo "开始实验..."
echo ""

# 实验1: Baseline (无因果模块)
echo "================================================"
echo "实验1: Baseline (无因果模块)"
echo "================================================"
python train_causal.py \
    --dataset semeval2014 \
    --model baseline \
    --epochs 20 \
    --eval_every 5

echo ""
echo "================================================"
echo "实验2: Causal-HAFE (标准设置)"
echo "================================================"
python train_causal.py \
    --dataset semeval2014 \
    --model causal_hafe \
    --causal_dim 128 \
    --spurious_dim 64 \
    --num_confounders 5 \
    --lambda_indep 0.1 \
    --lambda_bias 0.5 \
    --lambda_ib 0.01 \
    --epochs 20 \
    --eval_every 5

echo ""
echo "================================================"
echo "实验3: Causal-HAFE + TIE推理"
echo "================================================"
python train_causal.py \
    --dataset semeval2014 \
    --model causal_hafe \
    --causal_dim 128 \
    --spurious_dim 64 \
    --num_confounders 5 \
    --epochs 20 \
    --eval_every 5 \
    --use_tie_inference

echo ""
echo "========================================"
echo "所有实验完成!"
echo "结果已保存到 ./checkpoints/"
echo "========================================"

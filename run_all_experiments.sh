#!/bin/bash

echo "=========================================="
echo "运行HAFE-ABSA对比实验"
echo "=========================================="

# 创建结果目录
mkdir -p results

# SemEval-2014
echo -e "\n【1/4】SemEval-2014 Baseline"
python train.py --dataset semeval2014 --model baseline --epochs 50 | tee results/semeval2014_baseline.log

echo -e "\n【2/4】SemEval-2014 HAFE"
python train.py --dataset semeval2014 --model hafe --epochs 50 | tee results/semeval2014_hafe.log

# SemEval-2016
echo -e "\n【3/4】SemEval-2016 Baseline"
python train.py --dataset semeval2016 --model baseline --epochs 50 | tee results/semeval2016_baseline.log

echo -e "\n【4/4】SemEval-2016 HAFE"
python train.py --dataset semeval2016 --model hafe --epochs 50 | tee results/semeval2016_hafe.log

echo -e "\n=========================================="
echo "所有实验完成！结果保存在 results/ 目录"
echo "=========================================="


python train_causal.py --dataset semeval2014 --model causal_hafe --epochs 10 --eval_every 2
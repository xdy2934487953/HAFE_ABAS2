"""
简化版Causal-HAFE训练脚本
性能优化版，采用简化的架构和更好的训练策略
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
import numpy as np

# 添加src到路径
sys.path.append('./src')

from data_loader import ABSADataset, download_datasets
from graph_builder import ABSAGraphBuilder
from simplified_causal_hafe import SimplifiedCausalHAFE, SimplifiedHAFE_Baseline
from evaluator import ABSAEvaluator
from utils import ExperimentLogger, create_experiment_logger
import time


def train_epoch_simplified(model, graphs, frequency_buckets, dataset, optimizer, device):
    """训练一个epoch (简化版Causal-HAFE)"""
    model.train()
    total_loss = 0
    loss_breakdown = {'total': 0, 'task': 0, 'indep': 0, 'bias': 0, 'ib': 0}
    num_samples = 0

    for graph in graphs:
        if graph['labels'].shape[0] == 0:
            continue

        features = graph['features'].to(device)
        edge_index = graph['edge_index'].to(device)
        edge_types = graph.get('edge_types', None)
        if edge_types is not None:
            edge_types = edge_types.to(device)
        node_types = graph.get('node_types', None)  # 新增！
        if node_types is not None:
            node_types = node_types.to(device)
        positions = graph.get('positions', None)  # 新增！
        if positions is not None:
            positions = positions.to(device)
        aspect_indices = graph['aspect_indices'].to(device)
        labels = graph['labels'].to(device)

        # 获取频率标签
        aspect_words = graph['aspect_words']
        frequency_labels = []
        for aspect_word in aspect_words:
            aspect_key = aspect_word.lower()
            freq_label = frequency_buckets.get(aspect_key, 2)  # 默认中频
            frequency_labels.append(freq_label)
        frequency_labels = torch.tensor(frequency_labels, dtype=torch.long).to(device)

        optimizer.zero_grad()

        try:
            # 前向传播
            logits, dib_losses = model(
                features, edge_index, aspect_indices, edge_types,
                node_types, positions, frequency_labels, return_dib_losses=True
            )

            # 检查nan
            if torch.isnan(logits).any():
                print(f"警告: logits包含nan，跳过该batch")
                continue

            # 计算总损失
            loss, loss_dict = model.compute_total_loss(logits, labels, dib_losses)

            # 检查nan/inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: loss为nan/inf，跳过该batch")
                continue

            loss.backward()

            # 梯度剪切 (更温和)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            total_loss += loss.item()
            for key in loss_breakdown.keys():
                loss_breakdown[key] += loss_dict[key]
            num_samples += 1

        except Exception as e:
            print(f"训练错误: {e}")
            continue

    # 平均损失
    if num_samples == 0:
        print("警告: 没有成功训练任何样本！")
        return float('nan'), loss_breakdown

    avg_loss = total_loss / num_samples
    for key in loss_breakdown.keys():
        loss_breakdown[key] /= num_samples

    return avg_loss, loss_breakdown


def evaluate_simplified(model, graphs, evaluator, aspect_freq, device, use_tie=False, tie_mode='basic'):
    """评估简化版模型"""
    model.eval()
    evaluator.reset()

    with torch.no_grad():
        for graph in graphs:
            if graph['labels'].shape[0] == 0:
                continue

            features = graph['features'].to(device)
            edge_index = graph['edge_index'].to(device)
            edge_types = graph.get('edge_types', None)
            if edge_types is not None:
                edge_types = edge_types.to(device)
            node_types = graph.get('node_types', None)  # 新增！
            if node_types is not None:
                node_types = node_types.to(device)
            positions = graph.get('positions', None)  # 新增！
            if positions is not None:
                positions = positions.to(device)
            aspect_indices = graph['aspect_indices'].to(device)
            labels = graph['labels'].to(device)

            try:
                if use_tie:
                    # 基础推理 (简化版不支持复杂TIE)
                    logits = model(features, edge_index, aspect_indices, edge_types, node_types, positions)
                    preds = torch.argmax(logits, dim=1)
                else:
                    # 标准前向传播
                    logits = model(features, edge_index, aspect_indices, edge_types, node_types, positions)
                    preds = torch.argmax(logits, dim=1)

                aspects = graph['aspect_words']
                evaluator.add_batch(preds, labels, aspects)

            except Exception as e:
                print(f"评估错误: {e}")
                continue

    metrics = evaluator.compute_metrics(aspect_freq)
    return metrics


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建实验日志记录器
    experiment_name = f"{args.model}_{args.dataset}_lr{args.lr}_h{args.hidden_dim}"
    logger = create_experiment_logger(
        experiment_name,
        config=vars(args)
    )
    print(f"实验日志目录: {logger.experiment_dir}")

    start_time = time.time()

    # 1. 检查数据集
    if args.dataset == 'semeval2014':
        data_path = './data/semeval2014/'
        train_file = 'Restaurants_Train.xml'
    elif args.dataset == 'semeval2016':
        data_path = './data/semeval2016/'
        train_file = 'ABSA16_Restaurants_Train_SB1_v2.xml'
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if not os.path.exists(os.path.join(data_path, train_file)):
        print(f"数据集未找到: {args.dataset}")
        download_datasets()
        return

    # 2. 加载数据
    print(f"\n=== 加载数据集: {args.dataset} ===")
    train_dataset = ABSADataset(data_path, dataset_name=args.dataset, phase='train')
    test_dataset = ABSADataset(data_path, dataset_name=args.dataset, phase='test')

    # 计算频率分桶
    frequency_buckets = train_dataset.compute_frequency_buckets(num_buckets=args.num_frequency_buckets)
    print(f"\n频率分桶完成 (共{args.num_frequency_buckets}个桶)")

    # 3. 构建图
    print("\n=== 构建增强图 ===")
    graph_builder = ABSAGraphBuilder(device=device)

    print("处理训练集...")
    train_graphs = []
    skipped = 0
    for i, sample in enumerate(train_dataset):
        try:
            graph = graph_builder.build_graph(sample['text'], sample['aspects'])
            train_graphs.append(graph)
        except Exception as e:
            skipped += 1
            continue

    print("处理测试集...")
    test_graphs = []
    skipped = 0
    for i, sample in enumerate(test_dataset):
        try:
            graph = graph_builder.build_graph(sample['text'], sample['aspects'])
            test_graphs.append(graph)
        except Exception as e:
            skipped += 1
            continue

    print(f"训练图数量: {len(train_graphs)}, 测试图数量: {len(test_graphs)}")

    if len(train_graphs) == 0 or len(test_graphs) == 0:
        print("错误: 没有成功构建任何图！")
        return

    # 4. 初始化模型
    print("\n=== 初始化简化版Causal-HAFE模型 ===")
    if args.model == 'simplified_causal_hafe':
        model = SimplifiedCausalHAFE(
            input_dim=768,
            hidden_dim=args.hidden_dim,
            causal_dim=args.causal_dim,
            spurious_dim=args.spurious_dim,
            num_classes=3,
            device=device,
            dataset_name=args.dataset,
            num_frequency_buckets=args.num_frequency_buckets,
            num_edge_types=4,
            gat_heads=args.gat_heads,
            lambda_indep=args.lambda_indep,
            lambda_bias=args.lambda_bias,
            lambda_ib=args.lambda_ib,
            dropout=args.dropout
        )
        model_name = "Simplified Causal-HAFE"
        print(f"模型: {model_name}")
        print(f"  隐藏维度: {args.hidden_dim}")
        print(f"  因果维度: {args.causal_dim}")
        print(f"  虚假维度: {args.spurious_dim}")
        print(f"  GAT头数: {args.gat_heads}")

    else:  # baseline
        model = SimplifiedHAFE_Baseline(
            input_dim=768,
            hidden_dim=args.hidden_dim,
            num_classes=3,
            num_edge_types=4,
            gat_heads=args.gat_heads,
            dropout=args.dropout
        )
        model_name = "Simplified HAFE Baseline"
        print(f"模型: {model_name}")

    # 将模型移到设备
    model = model.to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 5. 训练设置
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度器 (更激进)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )

    evaluator = ABSAEvaluator()

    best_macro_f1 = 0
    best_metrics = None

    # 6. 训练循环
    print("\n=== 开始训练 ===")
    epoch_start_time = time.time()

    for epoch in range(args.epochs):
        # 训练一个epoch
        if args.model == 'simplified_causal_hafe':
            train_loss, loss_breakdown = train_epoch_simplified(
                model, train_graphs, frequency_buckets, train_dataset, optimizer, device
            )
        else:
            # Baseline训练
            train_loss = train_epoch_baseline(
                model, train_graphs, optimizer, device
            )
            loss_breakdown = None

        # 记录训练日志
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']

        logger.log_train_step(
            epoch=epoch,
            loss_dict=loss_breakdown or {'total': train_loss},
            lr=current_lr,
            time_elapsed=epoch_time
        )

        if (epoch + 1) % args.eval_every == 0:
            # 评估
            test_metrics = evaluate_simplified(
                model, test_graphs, evaluator, train_dataset.aspect_freq, device,
                use_tie=False
            )

            # 记录评估日志
            logger.log_eval_step(epoch, test_metrics)

            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            if loss_breakdown:
                print(f"    - Task: {loss_breakdown['task']:.4f}")
                print(f"    - Indep: {loss_breakdown['indep']:.4f}")
                print(f"    - Bias: {loss_breakdown['bias']:.4f}")
                print(f"    - IB: {loss_breakdown['ib']:.4f}")
            print(f"  Test Acc: {test_metrics['accuracy']*100:.2f}%")
            print(f"  Test Macro-F1: {test_metrics['macro_f1']*100:.2f}%")
            if 'gini' in test_metrics:
                print(f"  Gini: {test_metrics['gini']:.4f}")
                print(f"  DP-Aspect: {test_metrics['dp_aspect']:.4f}")

            # 检查是否为最佳模型
            is_best = test_metrics['macro_f1'] > best_macro_f1
            if is_best:
                best_macro_f1 = test_metrics['macro_f1']
                best_metrics = test_metrics

            # 保存模型checkpoint
            logger.save_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                is_best=is_best
            )

            if is_best:
                print(f"  💾 保存最佳模型 (F1: {best_macro_f1:.4f})")

            # 更新学习率
            scheduler.step(test_metrics['macro_f1'])

        epoch_start_time = time.time()

    # 7. 生成训练报告和可视化
    total_time = time.time() - start_time
    print(f"\n训练总用时: {total_time:.2f}秒 ({total_time/3600:.2f}小时)")

    # 生成训练曲线图
    logger.plot_training_curves()

    # 生成实验报告
    report = logger.generate_report()

    # 打印详细的实验总结
    logger.print_summary()

    # 7. 最终结果
    if best_metrics is None:
        print("训练失败，没有有效的评估结果")
        return

    print("\n" + "="*60)
    print(f"最终结果 - {args.dataset.upper()} - {model_name}")
    print("="*60)
    print(f"Accuracy: {best_metrics['accuracy']*100:.2f}%")
    print(f"Macro-F1: {best_metrics['macro_f1']*100:.2f}%")
    print(f"Micro-F1: {best_metrics['micro_f1']*100:.2f}%")

    if 'gini' in best_metrics:
        print(f"\n公平性指标:")
        print(f"  Variance: {best_metrics['variance']:.4f}")
        print(f"  Gap: {best_metrics['gap']:.4f}")
        print(f"  Gini: {best_metrics['gini']:.4f}")
        print(f"  DP-Aspect: {best_metrics['dp_aspect']:.4f}")
        print(f"  High-Freq F1: {best_metrics['high_freq_f1']:.4f}")
        print(f"  Low-Freq F1: {best_metrics['low_freq_f1']:.4f}")

    # 8. Per-aspect详细结果
    if len(best_metrics['per_aspect_f1']) > 0:
        print(f"\nPer-Aspect F1 (Top 10):")
        sorted_aspects = sorted(best_metrics['per_aspect_f1'].items(),
                               key=lambda x: train_dataset.aspect_freq.get(x[0], 0),
                               reverse=True)
        for aspect, f1 in sorted_aspects[:10]:
            freq = train_dataset.aspect_freq.get(aspect, 0)
            print(f"  {aspect:35s} (freq={freq:4d}): F1={f1:.4f}")

    print("="*60)
    print(f"\n实验结果已保存到: {logger.experiment_dir}")
    print("包含: 训练日志、模型checkpoint、可视化图表、详细报告")


def train_epoch_baseline(model, graphs, optimizer, criterion, device):
    """基线模型训练"""
    model.train()
    total_loss = 0
    num_samples = 0

    for graph in graphs:
        if graph['labels'].shape[0] == 0:
            continue

        features = graph['features'].to(device)
        edge_index = graph['edge_index'].to(device)
        edge_types = graph.get('edge_types', None)
        if edge_types is not None:
            edge_types = edge_types.to(device)
        node_types = graph.get('node_types', None)
        if node_types is not None:
            node_types = node_types.to(device)
        positions = graph.get('positions', None)
        if positions is not None:
            positions = positions.to(device)
        aspect_indices = graph['aspect_indices'].to(device)
        labels = graph['labels'].to(device)

        optimizer.zero_grad()

        try:
            logits = model(features, edge_index, aspect_indices, edge_types, node_types, positions)
            loss = criterion(logits, labels)

            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

                total_loss += loss.item()
                num_samples += 1

        except Exception as e:
            continue

    return total_loss / num_samples if num_samples > 0 else float('nan')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='简化版Causal-HAFE训练脚本')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='semeval2014',
                       choices=['semeval2014', 'semeval2016'],
                       help='选择数据集')

    # 模型参数
    parser.add_argument('--model', type=str, default='simplified_causal_hafe',
                       choices=['simplified_causal_hafe', 'baseline'],
                       help='选择模型')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='隐藏层维度')
    parser.add_argument('--causal_dim', type=int, default=128,
                       help='因果表示维度')
    parser.add_argument('--spurious_dim', type=int, default=64,
                       help='虚假表示维度')
    parser.add_argument('--gat_heads', type=int, default=4,
                       help='GAT注意力头数')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout比例')

    # DIB损失权重 (优化后的权重)
    parser.add_argument('--lambda_indep', type=float, default=0.1,
                       help='解耦约束权重')
    parser.add_argument('--lambda_bias', type=float, default=0.5,
                       help='偏差拟合权重')
    parser.add_argument('--lambda_ib', type=float, default=0.01,
                       help='信息瓶颈权重')

    # 频率分桶
    parser.add_argument('--num_frequency_buckets', type=int, default=5,
                       help='频率分桶数量')

    # 训练参数 (优化后的超参数)
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                       help='权重衰减')
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数')
    parser.add_argument('--eval_every', type=int, default=2,
                       help='每隔多少轮评估一次')

    args = parser.parse_args()

    # 打印配置
    print("="*60)
    print("简化版Causal-HAFE训练配置:")
    print("="*60)
    print(f"  数据集: {args.dataset}")
    print(f"  模型: {args.model}")
    print(f"  隐藏维度: {args.hidden_dim}")
    print(f"  因果维度: {args.causal_dim}")
    print(f"  虚假维度: {args.spurious_dim}")
    print(f"  GAT头数: {args.gat_heads}")
    print(f"  学习率: {args.lr}")
    print(f"  训练轮数: {args.epochs}")
    print("="*60)

    main(args)

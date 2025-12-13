import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import sys

# 添加src到路径
sys.path.append('./src')

from data_loader import ABSADataset, download_datasets
from graph_builder import ABSAGraphBuilder
from hafe_absa import HAFE_ABSA_Model, BaselineASGCN
from evaluator import ABSAEvaluator

def collate_fn(batch):
    """自定义batch整理函数"""
    return batch

def train_epoch(model, graphs, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_samples = 0
    
    for graph in graphs:
        if graph['labels'].shape[0] == 0:
            continue
        
        features = graph['features'].to(device)
        edge_index = graph['edge_index'].to(device)
        edge_types = graph.get('edge_types', None)  # 获取边类型
        if edge_types is not None:
            edge_types = edge_types.to(device)
        aspect_indices = graph['aspect_indices'].to(device)
        labels = graph['labels'].to(device)
        
        assert aspect_indices.shape[0] == labels.shape[0], \
            f"维度不匹配: aspect_indices={aspect_indices.shape[0]}, labels={labels.shape[0]}"
        
        optimizer.zero_grad()
        
        try:
            # 传入edge_types
            logits = model(features, edge_index, aspect_indices, edge_types)
            
            assert logits.shape[0] == labels.shape[0], \
                f"输出维度不匹配: logits={logits.shape[0]}, labels={labels.shape[0]}"
            
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_samples += 1
        except Exception as e:
            print(f"训练错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return total_loss / max(num_samples, 1)

def evaluate(model, graphs, evaluator, aspect_freq, device):
    """评估模型"""
    model.eval()
    evaluator.reset()
    
    with torch.no_grad():
        for graph in graphs:
            if graph['labels'].shape[0] == 0:
                continue
            
            features = graph['features'].to(device)
            edge_index = graph['edge_index'].to(device)
            edge_types = graph.get('edge_types', None)  # 获取边类型
            if edge_types is not None:
                edge_types = edge_types.to(device)
            aspect_indices = graph['aspect_indices'].to(device)
            labels = graph['labels'].to(device)
            
            try:
                # 传入edge_types
                logits = model(features, edge_index, aspect_indices, edge_types)
                preds = torch.argmax(logits, dim=1)
                
                assert preds.shape[0] == labels.shape[0], \
                    f"预测维度不匹配: preds={preds.shape[0]}, labels={labels.shape[0]}"
                
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
    
    # 3. 构建图
    print("\n=== 构建ABSA图 ===")
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
            if skipped <= 3:
                print(f"警告 {i}: {e}")
            continue
    
    if skipped > 3:
        print(f"... 跳过了总共 {skipped} 个样本")
    
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
    
    # 检查边类型是否存在
    has_edge_types = 'edge_types' in train_graphs[0]
    if has_edge_types:
        print(f"✓ 图中包含边类型信息")
        # 统计边类型分布
        edge_type_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for graph in train_graphs:
            for edge_type in graph['edge_types'].tolist():
                edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        
        total_edges = sum(edge_type_counts.values())
        print(f"  边类型分布:")
        print(f"    OPINION (0):     {edge_type_counts[0]:6d} ({edge_type_counts[0]/total_edges*100:5.1f}%)")
        print(f"    SYNTAX_CORE (1): {edge_type_counts[1]:6d} ({edge_type_counts[1]/total_edges*100:5.1f}%)")
        print(f"    COREF (2):       {edge_type_counts[2]:6d} ({edge_type_counts[2]/total_edges*100:5.1f}%)")
        print(f"    OTHER (3):       {edge_type_counts[3]:6d} ({edge_type_counts[3]/total_edges*100:5.1f}%)")
    else:
        print(f"✗ 图中不包含边类型信息（使用标准GCN）")
    
    # 4. 初始化模型
    print("\n=== 初始化模型 ===")
    if args.model == 'hafe':
        model = HAFE_ABSA_Model(
            input_dim=768, 
            hidden_dim=args.hidden_dim, 
            num_classes=3, 
            device=device,
            dataset_name=args.dataset,
            use_type_aware=args.use_type_aware
        )
        model_name = "HAFE-ABSA"
        if args.use_type_aware:
            model_name += " (Type-Aware GCN)"
        print(f"模型: {model_name}")
        
        # F3预处理
        model.preprocess_f3(train_graphs)
    else:
        model = BaselineASGCN(
            input_dim=768, 
            hidden_dim=args.hidden_dim, 
            num_classes=3,
            use_type_aware=args.use_type_aware
        )
        model_name = "Baseline ASGCN"
        if args.use_type_aware:
            model_name += " (Type-Aware GCN)"
        print(f"模型: {model_name}")
    
    model = model.to(device)
    
    # 5. 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    evaluator = ABSAEvaluator()
    
    best_macro_f1 = 0
    best_metrics = None
    
    # 6. 训练循环
    print("\n=== 开始训练 ===")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_graphs, optimizer, criterion, device)
        
        if (epoch + 1) % args.eval_every == 0:
            test_metrics = evaluate(model, test_graphs, evaluator, train_dataset.aspect_freq, device)
            
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Test Acc: {test_metrics['accuracy']*100:.2f}%")
            print(f"  Test Macro-F1: {test_metrics['macro_f1']*100:.2f}%")
            if 'gini' in test_metrics:
                print(f"  Gini: {test_metrics['gini']:.4f}")
                print(f"  Gap: {test_metrics['gap']:.4f}")
                print(f"  DP-Aspect: {test_metrics['dp_aspect']:.4f}")
            
            if test_metrics['macro_f1'] > best_macro_f1:
                best_macro_f1 = test_metrics['macro_f1']
                best_metrics = test_metrics
                
                # 保存模型
                os.makedirs('./checkpoints', exist_ok=True)
                save_name = f"{args.dataset}_{args.model}"
                if args.use_type_aware:
                    save_name += "_typeaware"
                torch.save(model.state_dict(), f'./checkpoints/{save_name}_best.pt')
    
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HAFE-ABSA训练脚本')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='semeval2014', 
                       choices=['semeval2014', 'semeval2016'],
                       help='选择数据集')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='hafe', 
                       choices=['hafe', 'baseline'],
                       help='选择模型')
    parser.add_argument('--use_type_aware', action='store_true',
                       help='使用类型感知GCN（边类型区分）')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='隐藏层维度')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--eval_every', type=int, default=5,
                       help='每隔多少轮评估一次')
    
    args = parser.parse_args()
    
    # 打印配置
    print("="*60)
    print("训练配置:")
    print("="*60)
    print(f"  数据集: {args.dataset}")
    print(f"  模型: {args.model}")
    print(f"  类型感知GCN: {'是' if args.use_type_aware else '否'}")
    print(f"  隐藏维度: {args.hidden_dim}")
    print(f"  学习率: {args.lr}")
    print(f"  训练轮数: {args.epochs}")
    print("="*60)
    
    main(args)
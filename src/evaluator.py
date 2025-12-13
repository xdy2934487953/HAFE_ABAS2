import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict

class ABSAEvaluator:
    """ABSA评估器：计算准确性和公平性指标"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.all_aspects = []
    
    def add_batch(self, preds, labels, aspects):
        """
        Args:
            preds: [batch_size] 预测标签
            labels: [batch_size] 真实标签
            aspects: [batch_size] aspect词列表
        """
        self.all_preds.extend(preds.cpu().tolist())
        self.all_labels.extend(labels.cpu().tolist())
        self.all_aspects.extend(aspects)
    
    def compute_metrics(self, aspect_freq=None):
        """计算所有指标"""
        if len(self.all_preds) == 0:
            return {
                'accuracy': 0.0,
                'macro_f1': 0.0,
                'micro_f1': 0.0,
                'per_aspect_f1': {}
            }
        
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        # 1. 整体准确性指标
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
        micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
        
        # 2. Per-aspect性能
        aspect_f1_dict = self._compute_per_aspect_f1(preds, labels)
        
        # 3. 公平性指标
        if aspect_freq and len(aspect_f1_dict) > 0:
            fairness_metrics = self._compute_fairness_metrics(aspect_f1_dict, aspect_freq)
        else:
            fairness_metrics = {}
        
        return {
            'accuracy': acc,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'per_aspect_f1': aspect_f1_dict,
            **fairness_metrics
        }
    
    def _compute_per_aspect_f1(self, preds, labels):
        """计算每个aspect的F1"""
        aspect_results = defaultdict(lambda: {'preds': [], 'labels': []})
        
        for pred, label, aspect in zip(preds, labels, self.all_aspects):
            aspect_lower = aspect.lower()
            aspect_results[aspect_lower]['preds'].append(pred)
            aspect_results[aspect_lower]['labels'].append(label)
        
        aspect_f1_dict = {}
        for aspect, data in aspect_results.items():
            if len(data['preds']) >= 2:  # 至少需要2个样本
                try:
                    f1 = f1_score(data['labels'], data['preds'], average='macro', zero_division=0)
                    aspect_f1_dict[aspect] = f1
                except:
                    pass
        
        return aspect_f1_dict
    
    def _compute_fairness_metrics(self, aspect_f1_dict, aspect_freq):
        """计算公平性指标"""
        if len(aspect_f1_dict) < 2:
            return {}
        
        f1_values = np.array(list(aspect_f1_dict.values()))
        aspects = list(aspect_f1_dict.keys())
        
        # Variance
        variance = np.var(f1_values)
        
        # Min-Max Gap
        gap = np.max(f1_values) - np.min(f1_values)
        
        # Gini Coefficient
        sorted_f1 = np.sort(f1_values)
        n = len(sorted_f1)
        cumsum = np.cumsum(sorted_f1)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_f1)) / (n * np.sum(sorted_f1) + 1e-8) - (n + 1) / n
        
        # Demographic Parity (高频 vs 低频)
        freq_list = [aspect_freq.get(asp, 0) for asp in aspects]
        
        if len(freq_list) > 0:
            q75 = np.percentile(freq_list, 75)
            q25 = np.percentile(freq_list, 25)
            
            high_freq_f1 = [f1 for asp, f1 in aspect_f1_dict.items() 
                           if aspect_freq.get(asp, 0) >= q75]
            low_freq_f1 = [f1 for asp, f1 in aspect_f1_dict.items() 
                          if aspect_freq.get(asp, 0) <= q25]
            
            if len(high_freq_f1) > 0 and len(low_freq_f1) > 0:
                dp_aspect = abs(np.mean(high_freq_f1) - np.mean(low_freq_f1))
            else:
                dp_aspect = 0.0
        else:
            high_freq_f1 = []
            low_freq_f1 = []
            dp_aspect = 0.0
        
        return {
            'variance': variance,
            'gap': gap,
            'gini': gini,
            'dp_aspect': dp_aspect,
            'high_freq_f1': np.mean(high_freq_f1) if len(high_freq_f1) > 0 else 0,
            'low_freq_f1': np.mean(low_freq_f1) if len(low_freq_f1) > 0 else 0
        }
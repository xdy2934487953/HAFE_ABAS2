import os
import xml.etree.ElementTree as ET
import torch
import numpy as np
from torch.utils.data import Dataset

class ABSADataset(Dataset):
    """SemEval ABSA数据集加载器"""
    
    def __init__(self, data_path, dataset_name='semeval2014', phase='train'):
        """
        Args:
            data_path: 数据集根目录
            dataset_name: 'semeval2014' 或 'semeval2016'
            phase: 'train' 或 'test'
        """
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.phase = phase
        
        # 加载数据
        if dataset_name == 'semeval2014':
            self.samples = self._load_semeval2014_xml()
        elif dataset_name == 'semeval2016':
            self.samples = self._load_semeval2016_xml()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # 统计aspect频率
        self.aspect_freq = self._compute_aspect_frequency()
        
        print(f"加载 {dataset_name} {phase} 数据: {len(self.samples)} 条样本")
        print(f"Aspect频率分布 (Top 10):")
        sorted_freq = sorted(self.aspect_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        for aspect, freq in sorted_freq:
            print(f"  {aspect:30s}: {freq:4d}")
    
    def _load_semeval2014_xml(self):
        """解析SemEval-2014 XML文件"""
        if self.phase == 'train':
            xml_file = os.path.join(self.data_path, 'Restaurants_Train.xml')
        else:
            xml_file = os.path.join(self.data_path, 'Restaurants_Test.xml')
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        samples = []
        for sentence in root.findall('.//sentence'):
            text = sentence.find('text').text
            
            aspects = []
            aspect_terms = sentence.find('aspectTerms')
            if aspect_terms is not None:
                for opinion in aspect_terms.findall('aspectTerm'):
                    aspect = {
                        'term': opinion.get('term'),
                        'polarity': opinion.get('polarity'),
                        'from': int(opinion.get('from')),
                        'to': int(opinion.get('to'))
                    }
                    aspects.append(aspect)
            
            if len(aspects) > 0:
                samples.append({
                    'text': text,
                    'aspects': aspects
                })
        
        return samples
    
    def _load_semeval2016_xml(self):
        """解析SemEval-2016 XML文件（新增）"""
        if self.phase == 'train':
            xml_file = os.path.join(self.data_path, 'ABSA16_Restaurants_Train_SB1_v2.xml')
        else:
            xml_file = os.path.join(self.data_path, 'EN_REST_SB1_TEST.xml.gold')
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        samples = []
        
        # SemEval-2016的结构不同，使用Opinion标签
        for review in root.findall('.//Review'):
            for sentence_elem in review.findall('.//sentence'):
                text = sentence_elem.find('text').text
                
                if text is None:
                    continue
                
                aspects = []
                opinions = sentence_elem.find('Opinions')
                
                if opinions is not None:
                    for opinion in opinions.findall('Opinion'):
                        # SemEval-2016使用category而非term
                        category = opinion.get('category')
                        polarity = opinion.get('polarity')
                        target = opinion.get('target')
                        
                        if target and target != 'NULL':
                            # 如果有明确的target词
                            aspect_term = target
                            # 在文本中查找位置
                            try:
                                from_idx = text.lower().index(target.lower())
                                to_idx = from_idx + len(target)
                            except ValueError:
                                # 找不到则跳过
                                continue
                        else:
                            # 没有明确target，使用category作为aspect
                            aspect_term = category.split('#')[0].lower()
                            # 尝试在文本中找到相关词
                            from_idx = 0
                            to_idx = len(aspect_term)
                        
                        aspect = {
                            'term': aspect_term,
                            'category': category,  # 保留category信息
                            'polarity': polarity,
                            'from': from_idx,
                            'to': to_idx
                        }
                        aspects.append(aspect)
                
                if len(aspects) > 0:
                    samples.append({
                        'text': text,
                        'aspects': aspects
                    })
        
        return samples
    
    def _compute_aspect_frequency(self):
        """统计aspect词的频率"""
        freq = {}
        for sample in self.samples:
            for aspect in sample['aspects']:
                # 对于2016，如果有category就用category，否则用term
                if 'category' in aspect and self.dataset_name == 'semeval2016':
                    key = aspect['category']
                else:
                    key = aspect['term'].lower()
                freq[key] = freq.get(key, 0) + 1
        return freq

    def compute_frequency_buckets(self, num_buckets=5):
        """
        为每个样本的aspect分配频率分桶

        Args:
            num_buckets: 分桶数量 (默认5: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%)

        Returns:
            frequency_buckets: dict mapping aspect_key -> bucket_id (0 到 num_buckets-1)
        """
        # 获取所有频率值
        freq_values = list(self.aspect_freq.values())

        if len(freq_values) == 0:
            return {}

        # 计算百分位数作为分桶边界
        percentiles = np.linspace(0, 100, num_buckets + 1)
        boundaries = np.percentile(freq_values, percentiles)

        # 为每个aspect分配桶
        frequency_buckets = {}
        for aspect_key, freq in self.aspect_freq.items():
            # 使用digitize分配到桶
            bucket_id = np.digitize(freq, boundaries[1:-1])
            bucket_id = np.clip(bucket_id, 0, num_buckets - 1)
            frequency_buckets[aspect_key] = int(bucket_id)

        return frequency_buckets

    def get_aspect_key(self, aspect):
        """
        获取aspect的key (用于查找频率)

        Args:
            aspect: aspect字典

        Returns:
            key: aspect的标识符
        """
        if 'category' in aspect and self.dataset_name == 'semeval2016':
            return aspect['category']
        else:
            return aspect['term'].lower()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def download_datasets():
    """下载数据集说明"""
    print("=" * 60)
    print("数据集下载说明")
    print("=" * 60)
    
    print("\n【SemEval-2014 Restaurant】")
    print("1. 访问: https://alt.qcri.org/semeval2014/task4/")
    print("2. 下载:")
    print("   - Restaurants_Train.xml")
    print("   - Restaurants_Test.xml")
    print("3. 放置到: ./data/semeval2014/")
    
    print("\n【SemEval-2016 Restaurant】(新增)")
    print("1. 访问: http://alt.qcri.org/semeval2016/task5/")
    print("2. 下载:")
    print("   - ABSA16_Restaurants_Train_SB1_v2.xml")
    print("   - EN_REST_SB1_TEST.xml.gold")
    print("3. 放置到: ./data/semeval2016/")
    
    print("\n【快速下载命令】")
    print("mkdir -p data/semeval2014 data/semeval2016")
    print("# 然后手动下载文件到对应目录")
    print("=" * 60)
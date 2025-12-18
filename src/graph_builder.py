import torch
import stanza
import numpy as np
from transformers import BertTokenizer, BertModel

# ===== 新增：边类型定义 =====
class EdgeType:
    """边类型枚举"""
    OPINION = 0      # Aspect-Opinion边（最重要）
    SYNTAX_CORE = 1  # 核心句法边（nsubj, dobj, amod等）
    COREF = 2        # Aspect协同边
    OTHER = 3        # 其他边（连接词、冠词等）

# 核心句法关系列表
CORE_DEP_RELS = {
    'nsubj', 'nsubjpass',  # 主语
    'dobj', 'iobj',        # 宾语
    'amod', 'advmod',      # 修饰语
    'acomp', 'xcomp',      # 补语
    'compound',            # 复合词
}

# 情感词典（简化版，实际应该加载完整词典）
SENTIMENT_WORDS = {
    # Positive
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
    'delicious', 'perfect', 'beautiful', 'nice', 'lovely', 'awesome',
    'outstanding', 'superb', 'fabulous', 'terrific', 'brilliant',
    'delightful', 'pleasant', 'tasty', 'fresh', 'friendly', 'helpful',
    'cozy', 'elegant', 'romantic', 'comfortable', 'clean', 'quick',
    
    # Negative
    'bad', 'terrible', 'awful', 'horrible', 'poor', 'disappointing',
    'disgusting', 'nasty', 'unpleasant', 'rude', 'slow', 'dirty',
    'cold', 'overpriced', 'expensive', 'small', 'bland', 'mediocre',
    'disappointing', 'subpar', 'inadequate', 'inferior', 'lousy',
    'pathetic', 'dreadful', 'appalling', 'atrocious', 'abysmal'
}

class ABSAGraphBuilder:
    """ABSA图构建器（改进版：支持边类型）"""
    
    def __init__(self, device='cuda'):
        self.device = device

        # 设置Stanza资源目录到当前用户目录（解决Windows权限问题）
        import os
        self.stanza_dir = os.path.join(os.path.expanduser('~'), 'stanza_resources')
        os.makedirs(self.stanza_dir, exist_ok=True)

        # 初始化依存解析器
        print("初始化Stanza依存解析器...")
        print(f"Stanza资源目录: {self.stanza_dir}")
        try:
            self.nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse',
                                      use_gpu=(device=='cuda'), verbose=False,
                                      dir=self.stanza_dir)
        except:
            print("下载Stanza英文模型...")
            stanza.download('en', model_dir=self.stanza_dir, verbose=True)
            print("模型下载完成！")
            self.nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse',
                                      use_gpu=(device=='cuda'), verbose=False,
                                      dir=self.stanza_dir)
        
        # 初始化BERT
        print("初始化BERT...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.bert.eval()
        
    def build_graph(self, text, aspects):
        """
        构建单个句子的ABSA图（带边类型）
        
        Returns:
            graph: dict包含
                - features: [num_nodes, 768] BERT特征
                - edge_index: [2, num_edges] 边索引
                - edge_types: [num_edges] 边类型 ← 新增！
                - aspect_indices: [num_aspects] aspect节点的索引
                - labels: [num_aspects] 情感标签
        """
        # 1. 依存解析
        doc = self.nlp(text)
        words = [word.text for sent in doc.sentences for word in sent.words]
        
        if len(words) == 0:
            raise ValueError("句子解析后没有词")
        
        # 2. 构建依存边（同时记录依存关系）
        edges = []
        dep_rels = []  # 记录每条边的依存关系
        
        for sent in doc.sentences:
            for word in sent.words:
                if word.head > 0:  # 不是根节点
                    head_idx = word.head - 1
                    word_idx = word.id - 1
                    if head_idx < len(words) and word_idx < len(words):
                        edges.append([word_idx, head_idx])
                        edges.append([head_idx, word_idx])  # 双向边
                        
                        # 记录依存关系
                        dep_rels.append(word.deprel)
                        dep_rels.append(word.deprel)  # 双向都记录
        
        # 3. BERT特征提取
        features = self._extract_bert_features(text, words)
        
        # 4. 为每个aspect找到对应的节点索引
        aspect_indices = []
        labels = []
        aspect_words_list = []
        
        for aspect in aspects:
            aspect_term = aspect['term'].lower()
            aspect_from = aspect['from']
            aspect_to = aspect['to']
            
            # 找到与aspect最匹配的word索引
            best_match_idx = None
            best_overlap = 0
            
            char_pos = 0
            for i, word in enumerate(words):
                word_start = text.lower().find(word.lower(), char_pos)
                word_end = word_start + len(word)
                
                overlap_start = max(word_start, aspect_from)
                overlap_end = min(word_end, aspect_to)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match_idx = i
                
                char_pos = word_end
            
            if best_match_idx is None:
                aspect_tokens = aspect_term.split()
                for i, word in enumerate(words):
                    if word.lower() in aspect_tokens or aspect_tokens[0] in word.lower():
                        best_match_idx = i
                        break
            
            if best_match_idx is None:
                best_match_idx = 0
            
            aspect_indices.append(best_match_idx)
            aspect_words_list.append(words[best_match_idx])
            
            polarity_map = {'positive': 0, 'negative': 1, 'neutral': 2, 'conflict': 2}
            labels.append(polarity_map.get(aspect['polarity'], 2))
        
        # 5. 添加aspect间协同边
        coref_edge_start = len(edges)  # 记录协同边的起始位置
        for i in range(len(aspect_indices)):
            for j in range(i+1, len(aspect_indices)):
                if aspect_indices[i] != aspect_indices[j]:
                    edges.append([aspect_indices[i], aspect_indices[j]])
                    edges.append([aspect_indices[j], aspect_indices[i]])
                    dep_rels.append('coref')  # 标记为协同关系
                    dep_rels.append('coref')
        
        # ===== 新增：6. 识别边的类型 =====
        edge_types = self._identify_edge_types(
            edges, 
            dep_rels, 
            words, 
            aspect_indices,
            coref_edge_start
        )
        
        # 转换为tensor
        if len(edges) > 0:
            edge_index = torch.LongTensor(edges).t()
            edge_types = torch.LongTensor(edge_types)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_types = torch.zeros(0, dtype=torch.long)
        
        assert len(aspect_indices) == len(labels), \
            f"Aspect数量不匹配: {len(aspect_indices)} vs {len(labels)}"
        
        return {
            'features': features,
            'edge_index': edge_index,
            'edge_types': edge_types,  # 新增！
            'aspect_indices': torch.LongTensor(aspect_indices),
            'labels': torch.LongTensor(labels),
            'text': text,
            'words': words,
            'aspect_words': aspect_words_list
        }
    
    def _identify_edge_types(self, edges, dep_rels, words, aspect_indices, coref_start):
        """
        识别每条边的类型
        
        Args:
            edges: 边列表 [[src, tgt], ...]
            dep_rels: 依存关系列表 ['nsubj', 'amod', ...]
            words: 词列表
            aspect_indices: aspect节点索引列表
            coref_start: 协同边的起始位置
        
        Returns:
            edge_types: 边类型列表
        """
        edge_types = []
        
        for idx, (edge, dep_rel) in enumerate(zip(edges, dep_rels)):
            src, tgt = edge
            
            # 类型1: Aspect协同边（最容易识别）
            if idx >= coref_start:
                edge_types.append(EdgeType.COREF)
                continue
            
            # 类型2: Aspect-Opinion边（最重要）
            if self._is_aspect_opinion_edge(src, tgt, words, aspect_indices):
                edge_types.append(EdgeType.OPINION)
                continue
            
            # 类型3: 核心句法边
            if dep_rel in CORE_DEP_RELS:
                edge_types.append(EdgeType.SYNTAX_CORE)
                continue
            
            # 类型4: 其他边
            edge_types.append(EdgeType.OTHER)
        
        return edge_types
    
    def _is_aspect_opinion_edge(self, src, tgt, words, aspect_indices):
        """
        判断一条边是否是Aspect-Opinion边
        
        策略：
        1. 一端是aspect节点
        2. 另一端是情感词
        """
        # 检查是否有一端是aspect
        src_is_aspect = src in aspect_indices
        tgt_is_aspect = tgt in aspect_indices
        
        if not (src_is_aspect or tgt_is_aspect):
            return False
        
        # 检查另一端是否是opinion词
        opinion_idx = tgt if src_is_aspect else src
        
        if opinion_idx >= len(words):
            return False
        
        opinion_word = words[opinion_idx].lower()
        
        # 简单检查：是否在情感词典中
        return opinion_word in SENTIMENT_WORDS
    
    def _extract_bert_features(self, text, words):
        """提取BERT特征（与之前相同）"""
        encoded = self.tokenizer(text, return_tensors='pt', padding=True, 
                                truncation=True, max_length=128)
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state.squeeze(0)
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        
        word_features = []
        
        for word in words:
            word_lower = word.lower()
            word_tokens = self.tokenizer.tokenize(word_lower)
            
            token_indices = []
            for i, token in enumerate(tokens):
                token_clean = token.replace('##', '')
                if token_clean in word_lower or word_lower in token_clean:
                    token_indices.append(i)
            
            if len(token_indices) > 0:
                word_feat = embeddings[token_indices].mean(dim=0)
            else:
                word_feat = embeddings[1:-1].mean(dim=0) if embeddings.shape[0] > 2 else embeddings[0]
            
            word_features.append(word_feat)
        
        return torch.stack(word_features)
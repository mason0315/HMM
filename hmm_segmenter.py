"""
HMM中文分词与命名实体识别系统
使用人民日报1998年语料库进行训练和测试
"""

import os
import re
import pickle
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict
import glob


class HMM_Segmenter:
    """基于HMM的中文分词器"""
    
    def __init__(self):
        self.states = ['B', 'M', 'E', 'S']  # BMES标签
        self.state2id = {s: i for i, s in enumerate(self.states)}
        
        # 初始状态概率 π
        self.pi = defaultdict(float)
        
        # 状态转移概率矩阵 A
        self.A = defaultdict(lambda: defaultdict(float))
        
        # 发射概率矩阵 B
        self.B = defaultdict(lambda: defaultdict(float))
        
        # 统计计数（用于平滑）
        self.pi_count = defaultdict(int)
        self.A_count = defaultdict(lambda: defaultdict(int))
        self.B_count = defaultdict(lambda: defaultdict(int))
        
    def word_to_bmes(self, word: str) -> List[str]:
        """将一个词转换为BMES标签序列"""
        if len(word) == 1:
            return ['S']
        return ['B'] + ['M'] * (len(word) - 2) + ['E']
    
    def sentence_to_bmes(self, words: List[str]) -> Tuple[List[str], List[str]]:
        """将分好词的句子转为字符+标签对"""
        chars, labels = [], []
        for word in words:
            tags = self.word_to_bmes(word)
            for char, tag in zip(word, tags):
                chars.append(char)
                labels.append(tag)
        return chars, labels
    
    def parse_corpus_line(self, line: str) -> List[Tuple[str, str]]:
        """解析语料库的一行，返回(词, 词性)列表"""
        # 去除行首的编号和行尾空白
        line = re.sub(r'^\S+/m\s+', '', line.strip())
        
        # 匹配 词/词性 的模式
        pattern = r'\[?([^\s\]/]+)/([^\s\]]+)\]?'
        matches = re.findall(pattern, line)
        
        result = []
        for word, pos in matches:
            # 处理方括号包围的复合词
            if word.startswith('['):
                word = word[1:]
            result.append((word, pos))
        
        return result
    
    def load_corpus(self, file_paths: List[str]) -> List[Tuple[List[str], List[str]]]:
        """加载语料库并转换为BMES格式"""
        corpus = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                continue
                
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析词和词性
                    word_pos_list = self.parse_corpus_line(line)
                    
                    if not word_pos_list:
                        continue
                    
                    # 提取词列表
                    words = [wp[0] for wp in word_pos_list]
                    
                    # 转换为BMES格式
                    chars, labels = self.sentence_to_bmes(words)
                    
                    if chars and labels:
                        corpus.append((chars, labels))
        
        return corpus
    
    def train(self, corpus: List[Tuple[List[str], List[str]]]):
        """训练HMM模型"""
        print(f"开始训练，语料库大小: {len(corpus)} 条句子")
        
        # 统计频率
        for chars, tags in corpus:
            if not tags:
                continue
                
            # 统计初始概率
            self.pi_count[tags[0]] += 1
            
            # 统计转移与发射
            for i, (char, tag) in enumerate(zip(chars, tags)):
                self.B_count[tag][char] += 1
                if i > 0:
                    self.A_count[tags[i-1]][tag] += 1
        
        # 计算初始概率（加1平滑）
        total_pi = sum(self.pi_count.values()) + len(self.states)
        for s in self.states:
            self.pi[s] = (self.pi_count[s] + 1) / total_pi
        
        # 计算转移概率（加1平滑）
        for s in self.states:
            total = sum(self.A_count[s].values()) + len(self.states)
            for t in self.states:
                self.A[s][t] = (self.A_count[s][t] + 1) / total
        
        # 计算发射概率
        for s in self.states:
            total = sum(self.B_count[s].values()) + 1
            for char in self.B_count[s]:
                self.B[s][char] = self.B_count[s][char] / total
        
        print("训练完成!")
    
    def viterbi(self, sentence: str) -> List[str]:
        """Viterbi解码，输出BMES标签序列"""
        N = len(self.states)
        T = len(sentence)
        
        if T == 0:
            return []
        
        # 动态规划表
        dp = np.full((N, T), -np.inf)
        path = np.zeros((N, T), dtype=int)
        
        # 初始化
        for i, s in enumerate(self.states):
            emit = self.B[s].get(sentence[0], 1e-10)
            dp[i][0] = np.log(self.pi[s]) + np.log(emit)
        
        # 递推
        for t in range(1, T):
            for j, cur in enumerate(self.states):
                emit = self.B[cur].get(sentence[t], 1e-10)
                for i, prev in enumerate(self.states):
                    score = dp[i][t-1] + np.log(self.A[prev][cur]) + np.log(emit)
                    if score > dp[j][t]:
                        dp[j][t] = score
                        path[j][t] = i
        
        # 回溯
        best_end = np.argmax(dp[:, -1])
        tags = [best_end]
        for t in range(T-1, 0, -1):
            tags.insert(0, path[tags[0]][t])
        
        return [self.states[i] for i in tags]
    
    def segment(self, sentence: str) -> List[str]:
        """分词"""
        tags = self.viterbi(sentence)
        words, word = [], ""
        
        for char, tag in zip(sentence, tags):
            word += char
            if tag in ('E', 'S'):
                words.append(word)
                word = ""
        
        if word:
            words.append(word)
        
        return words
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'pi': dict(self.pi),
            'A': {k: dict(v) for k, v in self.A.items()},
            'B': {k: dict(v) for k, v in self.B.items()},
            'pi_count': dict(self.pi_count),
            'A_count': {k: dict(v) for k, v in self.A_count.items()},
            'B_count': {k: dict(v) for k, v in self.B_count.items()}
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pi = defaultdict(float, model_data['pi'])
        self.A = defaultdict(lambda: defaultdict(float), 
                            {k: defaultdict(float, v) for k, v in model_data['A'].items()})
        self.B = defaultdict(lambda: defaultdict(float),
                            {k: defaultdict(float, v) for k, v in model_data['B'].items()})
        print(f"模型已从 {filepath} 加载")


class NER_Recognizer:
    """命名实体识别器（基于规则+词典）"""
    
    def __init__(self):
        # 实体词典
        self.person_names = set()
        self.locations = set()
        self.organizations = set()
        
        # 姓氏列表
        self.surnames = set(['王', '李', '张', '刘', '陈', '杨', '黄', '赵', '吴', '周',
                            '徐', '孙', '马', '朱', '胡', '郭', '何', '林', '罗', '高',
                            '郑', '梁', '谢', '宋', '唐', '许', '韩', '冯', '邓', '曹',
                            '彭', '曾', '肖', '田', '董', '袁', '潘', '于', '蒋', '蔡',
                            '余', '杜', '叶', '程', '苏', '魏', '吕', '丁', '任', '沈',
                            '姚', '卢', '姜', '崔', '钟', '谭', '陆', '汪', '范', '金',
                            '石', '廖', '贾', '夏', '韦', '傅', '方', '白', '邹', '孟',
                            '熊', '秦', '邱', '江', '尹', '薛', '闫', '段', '雷', '侯',
                            '龙', '史', '陶', '黎', '贺', '顾', '毛', '郝', '龚', '邵',
                            '万', '钱', '严', '覃', '武', '戴', '莫', '孔', '向', '汤'])
    
    def train_from_corpus(self, corpus: List[List[Tuple[str, str]]]):
        """从语料库中训练实体词典"""
        for sentence in corpus:
            for word, pos in sentence:
                if pos == 'nr':  # 人名
                    self.person_names.add(word)
                elif pos == 'ns':  # 地名
                    self.locations.add(word)
                elif pos == 'nt':  # 机构名
                    self.organizations.add(word)
        
        print(f"提取实体: 人名{len(self.person_names)}个, 地名{len(self.locations)}个, 机构名{len(self.organizations)}个")
    
    def recognize(self, words: List[str]) -> List[Tuple[str, str]]:
        """识别命名实体"""
        entities = []
        i = 0
        
        while i < len(words):
            word = words[i]
            
            # 检查是否在词典中
            if word in self.person_names or self._is_person_name(word):
                entities.append((word, 'PER'))
            elif word in self.locations:
                entities.append((word, 'LOC'))
            elif word in self.organizations:
                entities.append((word, 'ORG'))
            elif self._is_location(word):
                entities.append((word, 'LOC'))
            
            i += 1
        
        return entities
    
    def _is_person_name(self, word: str) -> bool:
        """判断是否为人名（简单规则）"""
        if len(word) < 2 or len(word) > 4:
            return False
        return word[0] in self.surnames
    
    def _is_location(self, word: str) -> bool:
        """判断是否为地名（简单规则）"""
        location_suffix = ['省', '市', '县', '区', '镇', '乡', '村', '路', '街', '国', '州']
        return any(word.endswith(suffix) for suffix in location_suffix)


def evaluate_model(model: HMM_Segmenter, test_corpus: List[Tuple[List[str], List[str]]], 
                   max_samples: int = 500) -> Dict:
    """评估模型性能"""
    from sklearn.metrics import classification_report, accuracy_score
    
    all_true, all_pred = [], []
    
    test_samples = test_corpus[:max_samples]
    
    for chars, true_tags in test_samples:
        pred_tags = model.viterbi(''.join(chars))
        all_true.extend(true_tags)
        all_pred.extend(pred_tags)
    
    # 计算准确率
    accuracy = accuracy_score(all_true, all_pred)
    
    # 生成分类报告
    report = classification_report(all_true, all_pred, 
                                   labels=['B', 'M', 'E', 'S'],
                                   output_dict=True)
    
    return {
        'accuracy': accuracy,
        'report': report,
        'total_samples': len(test_samples)
    }


def main():
    """主函数"""
    print("=" * 60)
    print("HMM中文分词与命名实体识别系统")
    print("=" * 60)
    
    # 数据路径
    data_dir = r"d:\一些东西\自然语言处理\HMM\199801\199801"
    
    # 获取所有语料文件
    corpus_files = [
        os.path.join(data_dir, '199801.txt'),
        os.path.join(data_dir, '199802.txt'),
        os.path.join(data_dir, '199803.txt'),
        os.path.join(data_dir, '199804.txt'),
        os.path.join(data_dir, '199805.txt'),
        os.path.join(data_dir, '199806.txt'),
    ]
    
    # 检查文件
    available_files = [f for f in corpus_files if os.path.exists(f)]
    print(f"\n找到 {len(available_files)} 个语料文件:")
    for f in available_files:
        print(f"  - {os.path.basename(f)}")
    
    # 初始化模型
    hmm_model = HMM_Segmenter()
    ner_recognizer = NER_Recognizer()
    
    # 加载语料库
    print("\n正在加载语料库...")
    corpus = hmm_model.load_corpus(available_files)
    print(f"成功加载 {len(corpus)} 条句子")
    
    if len(corpus) == 0:
        print("错误: 未能加载任何语料!")
        return
    
    # 划分训练集和测试集 (8:2)
    split_idx = int(len(corpus) * 0.8)
    train_corpus = corpus[:split_idx]
    test_corpus = corpus[split_idx:]
    
    print(f"训练集: {len(train_corpus)} 条")
    print(f"测试集: {len(test_corpus)} 条")
    
    # 训练HMM模型
    print("\n" + "=" * 60)
    print("训练HMM模型...")
    hmm_model.train(train_corpus)
    
    # 保存模型
    model_path = r"d:\一些东西\自然语言处理\HMM\hmm_model.pkl"
    hmm_model.save_model(model_path)
    
    # 测试分词
    print("\n" + "=" * 60)
    print("分词测试")
    print("=" * 60)
    
    test_sentences = [
        "中国人民解放军在北京举行阅兵式",
        "江泽民同志发表新年讲话",
        "中国经济保持稳定发展",
        "我们充满信心地迈向新世纪",
        "北京市举行新年音乐会"
    ]
    
    for sent in test_sentences:
        result = hmm_model.segment(sent)
        tags = hmm_model.viterbi(sent)
        print(f"\n原文: {sent}")
        print(f"分词: {' / '.join(result)}")
        print(f"标签: {tags}")
    
    # 评估模型
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)
    
    eval_results = evaluate_model(hmm_model, test_corpus, max_samples=500)
    
    print(f"\n总体准确率: {eval_results['accuracy']:.4f}")
    print(f"评估样本数: {eval_results['total_samples']}")
    
    print("\n各类别性能:")
    report = eval_results['report']
    print(f"{'标签':<6} {'精确率':<10} {'召回率':<10} {'F1值':<10} {'支持度':<10}")
    print("-" * 50)
    for label in ['B', 'M', 'E', 'S']:
        if label in report:
            r = report[label]
            print(f"{label:<6} {r['precision']:<10.4f} {r['recall']:<10.4f} {r['f1-score']:<10.4f} {int(r['support']):<10}")
    
    # 宏平均
    print("-" * 50)
    print(f"{'宏平均':<6} {report['macro avg']['precision']:<10.4f} {report['macro avg']['recall']:<10.4f} {report['macro avg']['f1-score']:<10.4f}")
    print(f"{'加权平均':<6} {report['weighted avg']['precision']:<10.4f} {report['weighted avg']['recall']:<10.4f} {report['weighted avg']['f1-score']:<10.4f}")
    
    # NER识别
    print("\n" + "=" * 60)
    print("命名实体识别")
    print("=" * 60)
    
    # 从训练语料构建NER词典
    ner_corpus = []
    for file_path in available_files[:3]:  # 使用前3个文件构建词典
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                word_pos_list = hmm_model.parse_corpus_line(line)
                if word_pos_list:
                    ner_corpus.append(word_pos_list)
    
    ner_recognizer.train_from_corpus(ner_corpus)
    
    # 测试NER
    ner_test_sentences = [
        "江泽民在北京发表重要讲话",
        "中国人民解放军举行阅兵式",
        "上海市经济发展迅速",
        "李岚清访问欧洲各国"
    ]
    
    for sent in ner_test_sentences:
        words = hmm_model.segment(sent)
        entities = ner_recognizer.recognize(words)
        print(f"\n原文: {sent}")
        print(f"分词: {' / '.join(words)}")
        print(f"实体: {entities}")
    
    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

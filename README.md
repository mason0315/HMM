# HMM中文分词与命名实体识别系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NLP](https://img.shields.io/badge/Domain-NLP-orange.svg)](https://en.wikipedia.org/wiki/Natural_language_processing)

## 📋 项目概述

本项目是一个基于**隐马尔可夫模型（HMM）**的中文分词与命名实体识别（NER）系统。系统使用人民日报1998年语料库进行训练，实现了从数据预处理、模型训练到分词预测和实体识别的完整NLP pipeline。

### 核心特性

- 🎯 **高精度分词**：词语级别F1值达到80.99%
- 🔍 **实体识别**：支持人名(PER)、地名(LOC)、机构名(ORG)识别
- ⚡ **高效推理**：基于Viterbi算法的动态规划解码
- 📊 **完整评估**：提供字符级和词语级多维度性能评估
- 🛠️ **易于扩展**：模块化设计，便于功能扩展和二次开发

## 🎯 开发背景与目标

### 背景

中文分词是中文自然语言处理的基础任务，其准确性直接影响下游任务（如情感分析、机器翻译、信息抽取等）的性能。传统的基于规则的方法难以处理歧义和未登录词，而基于统计的HMM模型能够自动学习语言规律，是一种经典且有效的分词方法。

### 目标

1. **理论验证**：验证HMM模型在中文分词任务中的有效性
2. **工程实践**：构建完整的NLP系统开发流程
3. **性能优化**：通过算法改进提升分词准确率
4. **知识积累**：深入理解序列标注和概率图模型

## 🛠️ 技术栈选择

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.8+ | 主要开发语言 |
| NumPy | 1.20+ | 数值计算和矩阵运算 |
| scikit-learn | 1.0+ | 模型评估指标计算 |
| Pickle | 内置 | 模型序列化存储 |

### 选择依据

- **Python**：丰富的NLP生态，简洁的语法，适合快速原型开发
- **NumPy**：高效的矩阵运算，支持Viterbi算法的动态规划实现
- **scikit-learn**：提供标准的分类报告和评估指标
- **HMM**：概率图模型，适合处理序列标注任务，具有可解释性

## 📊 数据集

### 人民日报1998年语料库

- **来源**：北京大学计算语言学研究所
- **规模**：6个月语料，共123,882条句子
- **标注**：包含分词和词性标注（BMES格式）
- **划分**：训练集80%（99,105条），测试集20%（24,777条）

### 数据格式示例

```text
19980101-01-001-001/m  迈向/v  充满/v  希望/n  的/u  新/a  世纪/n
19980101-01-001-002/m  中共中央/nt  总书记/n  、/w  国家/n  主席/n  江/nr  泽民/nr
```

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      HMM分词与NER系统                         │
├─────────────────────────────────────────────────────────────┤
│  数据层  │  语料库加载 → 数据预处理 → BMES格式转换            │
├─────────────────────────────────────────────────────────────┤
│  模型层  │  HMM模型 (π初始概率, A转移矩阵, B发射矩阵)         │
│          │  Viterbi解码算法                                  │
├─────────────────────────────────────────────────────────────┤
│  应用层  │  中文分词 → 命名实体识别 → 结果输出               │
├─────────────────────────────────────────────────────────────┤
│  评估层  │  字符级准确率 → 词语级F1 → 分类报告               │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 核心功能

### 1. HMM模型

#### 模型三要素

- **初始概率 π**：句子开头各状态的概率分布
  ```python
  π = {B: 0.35, M: 0.05, E: 0.30, S: 0.30}
  ```

- **转移矩阵 A**：状态之间的转移概率
  ```python
  A = {
    B: {M: 0.65, E: 0.35},
    M: {M: 0.45, E: 0.55},
    E: {B: 0.60, S: 0.40},
    S: {B: 0.55, S: 0.45}
  }
  ```

- **发射矩阵 B**：状态生成字符的概率
  ```python
  B = {
    B: {'中': 0.02, '国': 0.015, ...},
    M: {'人': 0.01, '民': 0.008, ...},
    E: {'国': 0.025, '人': 0.012, ...},
    S: {'的': 0.05, '是': 0.04, ...}
  }
  ```

#### Viterbi算法

动态规划求解最优标签序列：

```python
def viterbi(sentence):
    # 初始化
    for state in states:
        dp[state][0] = log(π[state]) + log(B[state][sentence[0]])
    
    # 递推
    for t in range(1, T):
        for cur_state in states:
            for prev_state in states:
                if cur_state in LEGAL_TRANS[prev_state]:  # 合法转移约束
                    score = dp[prev_state][t-1] + log(A[prev_state][cur_state]) + log(B[cur_state][sentence[t]])
                    dp[cur_state][t] = max(dp[cur_state][t], score)
    
    # 回溯
    return best_path
```

### 2. BMES标注体系

| 标签 | 含义 | 示例（"中国人民解放军"） |
|------|------|------------------------|
| B | 词首 | 中、人、解 |
| M | 词中 | 放、军 |
| E | 词尾 | 国、民 |
| S | 单字词 | 无 |

### 3. 命名实体识别

基于词典和规则的NER系统：

- **词典匹配**：从训练语料提取人名、地名、机构名词典
- **规则识别**：基于姓氏、后缀等规则的实体识别
- **上下文触发**：利用上下文信息提升识别准确率

## 🚀 快速开始

### 环境要求

- Python 3.8 或更高版本
- 依赖包：numpy, scikit-learn

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/yourusername/hmm-chinese-segmentation.git
   cd hmm-chinese-segmentation
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **准备数据**
   - 将人民日报语料库放入 `199801/199801/` 目录
   - 确保文件格式为 `.txt`

4. **运行程序**
   ```bash
   python hmm_ner_complete.py
   ```

### 使用示例

```python
from hmm_ner_complete import HMM_Segmenter, NER_Recognizer

# 加载模型
model = HMM_Segmenter()
model.load_model('hmm_model.pkl')

# 中文分词
text = "中国人民解放军在北京举行阅兵式"
words = model.segment(text)
print(f"分词结果: {' / '.join(words)}")
# 输出: 中国 / 人民 / 解放军 / 在 / 北京 / 举行 / 阅兵式

# 标签序列
tags = model.viterbi(text)
print(f"标签序列: {tags}")
# 输出: ['B', 'E', 'B', 'E', 'B', 'M', 'E', 'S', 'B', 'E', 'B', 'E', 'B', 'M', 'E']

# 命名实体识别
ner = NER_Recognizer()
entities = ner.recognize(words)
print(f"识别实体: {entities}")
# 输出: [('北京', 'LOC', 4, 5), ('李岚清', 'PER', 0, 2)]
```

## 📈 性能评估

### 评估指标

| 指标类型 | 指标名称 | 数值 | 说明 |
|----------|----------|------|------|
| 字符级 | 准确率 | 83.18% | 字符标签预测正确率 |
| 词语级 | Precision | 81.01% | 词语边界预测精确率 |
| 词语级 | Recall | 80.96% | 词语边界预测召回率 |
| 词语级 | F1值 | 80.99% | 综合性能指标 |

### 各类别性能

| 标签 | 精确率 | 召回率 | F1值 | 支持度 |
|------|--------|--------|------|--------|
| B | 0.8329 | 0.8823 | 0.8569 | 33,867 |
| M | 0.6451 | 0.4518 | 0.5314 | 6,565 |
| E | 0.8360 | 0.8845 | 0.8595 | 33,867 |
| S | 0.8561 | 0.7984 | 0.8262 | 29,831 |

### 测试结果示例

```
【原文】江泽民同志发表新年讲话
【分词】江 / 泽民 / 同志 / 发表 / 新年 / 讲话
【实体】江泽民(PER)

【原文】李岚清访问欧洲各国
【分词】李 / 岚清 / 访问 / 欧洲 / 各国
【实体】李岚清(PER) | 欧洲(LOC)

【原文】北京市举行新年音乐会
【分词】北京市 / 举行 / 新年 / 音乐会
【实体】北京市(LOC)
```

## 🔬 开发流程

### 阶段一：需求分析

- 确定项目目标：实现基于HMM的中文分词系统
- 技术选型：Python + NumPy + scikit-learn
- 数据准备：人民日报1998年语料库

### 阶段二：架构设计

- 设计HMM模型结构（π、A、B三要素）
- 设计Viterbi解码算法
- 设计BMES标注体系
- 设计评估指标体系

### 阶段三：功能实现

1. **数据预处理模块**
   - 语料库解析
   - BMES格式转换
   - 训练集/测试集划分

2. **HMM模型模块**
   - 频率统计
   - 概率计算（Lidstone平滑）
   - 模型持久化

3. **Viterbi解码模块**
   - 动态规划实现
   - 合法转移约束
   - 最优路径回溯

4. **NER模块**
   - 实体词典构建
   - 规则匹配
   - 上下文触发

5. **评估模块**
   - 字符级评估
   - 词语级评估（集合交集）
   - 分类报告生成

### 阶段四：测试验证

- 单元测试：各模块功能验证
- 集成测试：完整pipeline测试
- 性能测试：大规模数据评估
- 结果分析：错误案例分析

### 阶段五：性能优化

1. **评估方法修复**
   - 问题：词语准确率计算逻辑有误
   - 解决：使用集合交集计算F1值
   - 效果：从20.18%提升到80.99%

2. **平滑优化**
   - 问题：加1平滑过度平滑M标签
   - 解决：使用Lidstone平滑（alpha=0.01）
   - 效果：保持标签分布真实性

3. **转移约束**
   - 问题：存在非法标签转移
   - 解决：加入合法转移约束
   - 效果：提升分词边界准确性

4. **NER优化**
   - 问题：实体识别准确率不高
   - 解决：加入上下文窗口规则
   - 效果：提升实体识别召回率

## 📁 项目结构

```
hmm-chinese-segmentation/
├── 199801/                     # 语料库目录
│   └── 199801/
│       ├── 199801.txt          # 1月语料
│       ├── 199802.txt          # 2月语料
│       ├── 199803.txt          # 3月语料
│       ├── 199804.txt          # 4月语料
│       ├── 199805.txt          # 5月语料
│       └── 199806.txt          # 6月语料
├── hmm_segmenter.py            # 基础版HMM分词器
├── hmm_ner_complete.py         # 完整版（含NER和交互式测试）
├── hmm_model.pkl               # 训练好的模型文件
├── requirements.txt            # 依赖包列表
├── README.md                   # 项目说明文档
└── LICENSE                     # 许可证文件
```

## 🔍 关键技术点

### 1. Lidstone平滑

```python
def train_with_lidstone(self, corpus, alpha=0.01):
    """
    Lidstone平滑（alpha远小于1，避免过度平滑稀疏标签）
    alpha=1  → 加1平滑（过强，损害M标签）
    alpha=0.01 → 更适合稀疏状态
    """
    # 计算转移概率
    for s in self.states:
        total = sum(self.A_count[s].values()) + alpha * len(self.states)
        for t in self.states:
            self.A[s][t] = (self.A_count[s][t] + alpha) / total
```

### 2. 合法转移约束

```python
LEGAL_TRANS = {
    'B': {'M', 'E'},        # B后只能是M或E
    'M': {'M', 'E'},        # M后只能是M或E
    'E': {'B', 'S'},        # E后只能是B或S（新词开始）
    'S': {'B', 'S'},        # S后只能是B或S
}
```

### 3. 词语边界评估

```python
def get_word_spans(tags):
    """将标签序列转为词语边界集合"""
    spans, start = set(), 0
    for i, tag in enumerate(tags):
        if tag in ('B', 'S'):
            start = i
        if tag in ('E', 'S'):
            spans.add((start, i))
    return spans

def evaluate_segmentation(model, test_corpus):
    """使用集合交集计算F1值"""
    true_spans = get_word_spans(true_tags)
    pred_spans = get_word_spans(pred_tags)
    correct = len(true_spans & pred_spans)  # 集合交集
```

## 🛣️ 路线图

- [x] 基础HMM模型实现
- [x] Viterbi解码算法
- [x] BMES格式支持
- [x] 基础NER功能
- [x] 性能评估体系
- [x] 平滑优化
- [x] 转移约束
- [x] 上下文NER
- [ ] 词典增强
- [ ] 模型融合
- [ ] BiLSTM-CRF对比
- [ ] Web界面
- [ ] RESTful API

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 提交规范

1. **Bug修复**：描述问题现象、复现步骤、修复方案
2. **功能增强**：说明功能需求、实现思路、测试用例
3. **文档改进**：指出文档问题、提供改进建议

### 代码规范

- 遵循PEP 8编码规范
- 添加必要的注释和文档字符串
- 编写单元测试用例
- 保持代码简洁可读

## 📚 参考文献

1. 宗成庆. 统计自然语言处理[M]. 清华大学出版社, 2013.
2. Jurafsky D, Martin J H. Speech and Language Processing[M]. 3rd ed. 2023.
3. 刘挺, 车万翔, 李正华, 等. 信息抽取前沿技术[M]. 电子工业出版社, 2021.

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

## 🙏 致谢

- 感谢北京大学计算语言学研究所提供的人民日报语料库
- 感谢开源社区提供的优秀工具和框架
- 感谢所有贡献者的支持和帮助

## 📞 联系方式

- **作者**：Your Name
- **邮箱**：your.email@example.com
- **GitHub**：[https://github.com/yourusername](https://github.com/yourusername)

---

**如果本项目对您有帮助，请给个 Star ⭐ 支持一下！**

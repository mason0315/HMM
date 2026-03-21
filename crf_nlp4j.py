"""
基于 CRF 的中文序列标注模型（NLP4J 对标方案）
实验一选做项：应用 NLP4J 模型进行序列标注

NLP4J 是 Java 生态工具，本文件提供功能等价的 Python 实现：
  - 使用 sklearn-crfsuite（工业级 CRF 库）
  - 设计与 NLP4J 相近的丰富上下文特征集
  - 同样基于人民日报语料训练，方便与 HMM / BiLSTM-CRF 横向对比

依赖: pip install sklearn-crfsuite scikit-learn
运行: python crf_nlp4j.py
"""

import os
import re
import time
import pickle
from typing import List, Tuple, Dict

import numpy as np
from sklearn.metrics import classification_report, accuracy_score

try:
    import sklearn_crfsuite
    from sklearn_crfsuite import metrics as crf_metrics
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    print("[警告] 未检测到 sklearn-crfsuite，请运行: pip install sklearn-crfsuite")


# ══════════════════════════════════════════════════════════════
# 1. 数据加载（与 HMM 实验共用同一语料解析逻辑）
# ══════════════════════════════════════════════════════════════

def parse_line(line: str) -> List[str]:
    """解析人民日报格式一行，返回词列表"""
    line = re.sub(r"^\S+/m\s+", "", line.strip())
    return [w.lstrip("[") for w in re.findall(r"\[?([^\s\]/]+)/[^\s\]]+\]?", line) if w]


def word_to_bmes(word: str) -> List[str]:
    if len(word) == 1:
        return ["S"]
    return ["B"] + ["M"] * (len(word) - 2) + ["E"]


def load_corpus(file_paths: List[str]) -> List[Tuple[List[str], List[str]]]:
    """加载语料库，返回 (字符列表, 标签列表) 的列表"""
    corpus = []
    for fp in file_paths:
        if not os.path.exists(fp):
            print(f"  [跳过] 文件不存在: {fp}")
            continue
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                words = parse_line(line.strip())
                if not words:
                    continue
                chars, labels = [], []
                for w in words:
                    for ch, tag in zip(w, word_to_bmes(w)):
                        chars.append(ch)
                        labels.append(tag)
                if chars:
                    corpus.append((chars, labels))
    return corpus


# ══════════════════════════════════════════════════════════════
# 2. 特征工程（NLP4J 风格的上下文窗口特征）
# ══════════════════════════════════════════════════════════════

def is_chinese(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"

def is_punctuation(ch: str) -> bool:
    return ch in "，。！？；：""''（）【】《》、…—～"

def is_digit(ch: str) -> bool:
    return ch.isdigit() or ch in "０１２３４５６７８９"

def is_alpha(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()

def char_type(ch: str) -> str:
    if is_chinese(ch):   return "CN"
    if is_digit(ch):     return "DIG"
    if is_alpha(ch):     return "ENG"
    if is_punctuation(ch): return "PUNC"
    return "OTHER"


def extract_features(chars: List[str], i: int) -> Dict[str, object]:
    """
    为位置 i 提取 NLP4J 风格特征：
      - 当前字及其类型
      - ±3 窗口内字符及类型
      - 二元、三元字符 n-gram
      - 词典触发特征（常见前缀/后缀）
    """
    ch = chars[i]
    feats = {
        "bias": 1.0,
        # ── 当前字 ──
        "c[0]": ch,
        "type[0]": char_type(ch),
        "is_cn[0]": is_chinese(ch),
        "is_dig[0]": is_digit(ch),
        "is_punc[0]": is_punctuation(ch),
    }

    # ── 上下文窗口 ±3 ──
    for offset in [-3, -2, -1, 1, 2, 3]:
        j = i + offset
        if 0 <= j < len(chars):
            c = chars[j]
            feats[f"c[{offset}]"]    = c
            feats[f"type[{offset}]"] = char_type(c)
        else:
            feats[f"c[{offset}]"]    = "<BOS>" if offset < 0 else "<EOS>"
            feats[f"type[{offset}]"] = "BOUND"

    # ── 二元 n-gram ──
    for offset in [-2, -1, 0, 1, 2]:
        j = i + offset
        if 0 <= j < len(chars) - 1:
            feats[f"bigram[{offset}]"] = chars[j] + chars[j + 1]

    # ── 三元 n-gram（覆盖当前字） ──
    if i >= 1 and i + 1 < len(chars):
        feats["trigram[-1,0,+1]"] = chars[i-1] + ch + chars[i+1]
    if i >= 2:
        feats["trigram[-2,-1,0]"] = chars[i-2] + chars[i-1] + ch
    if i + 2 < len(chars):
        feats["trigram[0,+1,+2]"] = ch + chars[i+1] + chars[i+2]

    # ── 词首/词尾位置特征 ──
    feats["is_first"] = (i == 0)
    feats["is_last"]  = (i == len(chars) - 1)

    # ── 常见地名/机构名后缀触发 ──
    LOC_SUFFIX  = {"省", "市", "县", "区", "镇", "乡", "村", "路", "街", "国", "州", "港"}
    ORG_SUFFIX  = {"部", "局", "厅", "委", "院", "团", "组", "社", "所", "处"}
    feats["is_loc_suffix"] = ch in LOC_SUFFIX
    feats["is_org_suffix"] = ch in ORG_SUFFIX

    return feats


def sent_to_features(chars: List[str]) -> List[Dict]:
    return [extract_features(chars, i) for i in range(len(chars))]


# ══════════════════════════════════════════════════════════════
# 3. 词语级 F1 计算
# ══════════════════════════════════════════════════════════════

def get_word_spans(tags: List[str]) -> set:
    spans, start = set(), 0
    for i, tag in enumerate(tags):
        if tag in ("B", "S"):
            start = i
        if tag in ("E", "S"):
            spans.add((start, i))
    return spans


def word_level_f1(true_seqs: List[List[str]], pred_seqs: List[List[str]]) -> Dict:
    tp = fp = fn = 0
    for t_tags, p_tags in zip(true_seqs, pred_seqs):
        ts = get_word_spans(t_tags)
        ps = get_word_spans(p_tags)
        tp += len(ts & ps)
        fp += len(ps - ts)
        fn += len(ts - ps)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1,
            "true_words": tp + fn, "pred_words": tp + fp, "correct": tp}


# ══════════════════════════════════════════════════════════════
# 4. 训练与评估
# ══════════════════════════════════════════════════════════════

def train_crf(
    corpus_files: List[str],
    model_save_path: str = "crf_nlp4j_model.pkl",
    max_train_samples: int = 80000,
    c1: float = 0.05,
    c2: float = 0.05,
    max_iterations: int = 150,
):
    """
    训练 CRF 模型

    参数:
      c1             : L1 正则化系数（稀疏特征选择）
      c2             : L2 正则化系数（防止过拟合）
      max_iterations : 最大迭代次数
    """
    if not CRF_AVAILABLE:
        print("[错误] 请先安装 sklearn-crfsuite: pip install sklearn-crfsuite")
        return None

    print("=" * 60)
    print("CRF 序列标注模型（NLP4J 对标）")
    print("=" * 60)

    # ── 1. 数据加载 ──
    print("\n[1/4] 加载语料库...")
    corpus = load_corpus(corpus_files)
    if not corpus:
        print("[错误] 未能加载任何语料。")
        return None
    print(f"  加载语句: {len(corpus):,} 条")

    split = int(len(corpus) * 0.8)
    train_data = corpus[:min(split, max_train_samples)]
    test_data  = corpus[split:split + 5000]   # 取5000条测试加速评估
    print(f"  训练集: {len(train_data):,} 条 | 测试集: {len(test_data):,} 条")

    # ── 2. 特征提取 ──
    print("\n[2/4] 提取特征（NLP4J 风格上下文窗口）...")
    t0 = time.time()
    X_train = [sent_to_features(chars) for chars, _ in train_data]
    y_train = [labels for _, labels in train_data]
    X_test  = [sent_to_features(chars) for chars, _ in test_data]
    y_test  = [labels for _, labels in test_data]
    print(f"  特征提取完成，耗时 {time.time()-t0:.1f}s")
    print(f"  特征维度示例: {len(X_train[0][0])} 个特征/字符")

    # ── 3. 训练 CRF ──
    print(f"\n[3/4] 训练 CRF（c1={c1}, c2={c2}, max_iter={max_iterations}）...")
    crf = sklearn_crfsuite.CRF(
        algorithm          = "lbfgs",       # 拟牛顿法，收敛快
        c1                 = c1,
        c2                 = c2,
        max_iterations     = max_iterations,
        all_possible_transitions = True,    # 对未见转移进行惩罚
    )
    t0 = time.time()
    crf.fit(X_train, y_train)
    print(f"  训练完成，耗时 {time.time()-t0:.1f}s")

    # ── 4. 保存模型 ──
    with open(model_save_path, "wb") as f:
        pickle.dump(crf, f)
    print(f"  模型已保存到: {model_save_path}")

    # ── 5. 评估 ──
    print("\n[4/4] 模型评估...")
    y_pred = crf.predict(X_test)

    # 字符级准确率
    all_true = [t for seq in y_test  for t in seq]
    all_pred = [p for seq in y_pred  for p in seq]
    char_acc = accuracy_score(all_true, all_pred)

    # 词语级 F1
    word_metrics = word_level_f1(y_test, y_pred)

    # 各标签报告
    report = classification_report(
        all_true, all_pred,
        labels=["B", "M", "E", "S"],
        digits=4,
        output_dict=True,
    )

    print(f"\n{'─'*55}")
    print(f"  字符级准确率       : {char_acc:.4f} ({char_acc*100:.2f}%)")
    print(f"  词语级精确率       : {word_metrics['precision']:.4f}")
    print(f"  词语级召回率       : {word_metrics['recall']:.4f}")
    print(f"  词语级 F1          : {word_metrics['f1']:.4f}")
    print(f"{'─'*55}")
    print(f"  {'标签':<6} {'精确率':>8} {'召回率':>8} {'F1值':>8} {'支持度':>8}")
    print(f"  {'─'*44}")
    for label in ["B", "M", "E", "S"]:
        r = report[label]
        print(f"  {label:<6} {r['precision']:>8.4f} {r['recall']:>8.4f} "
              f"{r['f1-score']:>8.4f} {int(r['support']):>8,}")

    return crf


# ══════════════════════════════════════════════════════════════
# 5. 推理接口
# ══════════════════════════════════════════════════════════════

class CRFPredictor:
    """训练好的 CRF 模型推理接口"""

    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            self.crf = pickle.load(f)

    def predict_tags(self, sentence: str) -> List[str]:
        chars  = list(sentence)
        feats  = sent_to_features(chars)
        return self.crf.predict([feats])[0]

    def segment(self, sentence: str) -> List[str]:
        tags = self.predict_tags(sentence)
        words, buf = [], ""
        for ch, tag in zip(sentence, tags):
            buf += ch
            if tag in ("E", "S"):
                words.append(buf)
                buf = ""
        if buf:
            words.append(buf)
        return words


# ══════════════════════════════════════════════════════════════
# 6. 主程序
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    DATA_DIR     = r"199801/199801"
    corpus_files = [os.path.join(DATA_DIR, f"19980{m}.txt") for m in range(1, 7)]

    crf = train_crf(
        corpus_files    = corpus_files,
        model_save_path = "crf_nlp4j_model.pkl",
        max_train_samples = 80000,
        c1              = 0.05,
        c2              = 0.05,
        max_iterations  = 150,
    )

    if crf is not None:
        print("\n" + "=" * 60)
        print("分词演示")
        print("=" * 60)
        predictor = CRFPredictor("crf_nlp4j_model.pkl")
        for sent in [
            "中国人民解放军在北京举行阅兵式",
            "江泽民同志发表新年讲话",
            "北京市举行新年音乐会",
            "李岚清访问欧洲各国",
            "中共中央总书记主持会议",
        ]:
            words = predictor.segment(sent)
            tags  = predictor.predict_tags(sent)
            print(f"\n原文: {sent}")
            print(f"分词: {' / '.join(words)}")
            print(f"标签: {tags}")

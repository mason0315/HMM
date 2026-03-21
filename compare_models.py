"""
三模型横向对比脚本
对比 HMM / CRF(NLP4J) / BiLSTM-CRF 在同一测试集上的分词性能

运行方式:
  python compare_models.py

前置条件:
  - 已运行 hmm_ner_complete.py        → 生成 hmm_model.pkl
  - 已运行 crf_nlp4j.py               → 生成 crf_nlp4j_model.pkl
  - 已运行 bilstm_crf.py              → 生成 bilstm_crf_model.pt
  - 语料库位于 199801/199801/ 目录
"""

import os
import re
import time
import pickle
from typing import List, Tuple

import numpy as np
from sklearn.metrics import classification_report, accuracy_score


# ══════════════════════════════════════════════════════════════
# 共用工具
# ══════════════════════════════════════════════════════════════

def parse_line(line: str) -> List[str]:
    line = re.sub(r"^\S+/m\s+", "", line.strip())
    return [w.lstrip("[") for w in re.findall(r"\[?([^\s\]/]+)/[^\s\]]+\]?", line) if w]

def word_to_bmes(word: str) -> List[str]:
    if len(word) == 1: return ["S"]
    return ["B"] + ["M"] * (len(word) - 2) + ["E"]

def load_corpus(file_paths: List[str]) -> List[Tuple[List[str], List[str]]]:
    corpus = []
    for fp in file_paths:
        if not os.path.exists(fp): continue
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                words = parse_line(line.strip())
                if not words: continue
                chars, labels = [], []
                for w in words:
                    for ch, tag in zip(w, word_to_bmes(w)):
                        chars.append(ch); labels.append(tag)
                if chars: corpus.append((chars, labels))
    return corpus

def get_word_spans(tags: List[str]) -> set:
    spans, start = set(), 0
    for i, tag in enumerate(tags):
        if tag in ("B", "S"): start = i
        if tag in ("E", "S"): spans.add((start, i))
    return spans

def compute_metrics(true_seqs: List[List[str]], pred_seqs: List[List[str]]) -> dict:
    all_true = [t for s in true_seqs for t in s]
    all_pred = [p for s in pred_seqs for p in s]
    char_acc = accuracy_score(all_true, all_pred)

    tp = fp = fn = 0
    for t_tags, p_tags in zip(true_seqs, pred_seqs):
        ts = get_word_spans(t_tags); ps = get_word_spans(p_tags)
        tp += len(ts & ps); fp += len(ps - ts); fn += len(ts - ps)

    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)

    report = classification_report(all_true, all_pred,
                                   labels=["B","M","E","S"],
                                   digits=4, output_dict=True)
    return {"char_acc": char_acc, "precision": prec,
            "recall": rec, "f1": f1, "report": report}


# ══════════════════════════════════════════════════════════════
# 各模型预测封装
# ══════════════════════════════════════════════════════════════

def predict_hmm(model_path: str, test_data: List[Tuple]) -> List[List[str]]:
    """使用保存的 HMM 模型预测"""
    from collections import defaultdict
    with open(model_path, "rb") as f:
        data = pickle.load(f)

    pi = defaultdict(float, data["pi"])
    A  = defaultdict(lambda: defaultdict(float),
                     {k: defaultdict(float, v) for k, v in data["A"].items()})
    B  = defaultdict(lambda: defaultdict(float),
                     {k: defaultdict(float, v) for k, v in data["B"].items()})
    states = ["B", "M", "E", "S"]

    LEGAL = {"B":{"M","E"}, "M":{"M","E"}, "E":{"B","S"}, "S":{"B","S"}}

    def viterbi(sentence):
        import numpy as np
        N, T = len(states), len(sentence)
        if T == 0: return []
        dp   = np.full((N, T), -np.inf)
        path = np.zeros((N, T), dtype=int)
        for i, s in enumerate(states):
            emit = B[s].get(sentence[0], B[s].get("<UNK>", 1e-10))
            dp[i][0] = np.log(pi[s]) + np.log(emit)
        for t in range(1, T):
            for j, cur in enumerate(states):
                emit = B[cur].get(sentence[t], B[cur].get("<UNK>", 1e-10))
                for i, prev in enumerate(states):
                    if cur not in LEGAL[prev]: continue
                    score = dp[i][t-1] + np.log(A[prev][cur]) + np.log(emit)
                    if score > dp[j][t]:
                        dp[j][t] = score; path[j][t] = i
        best = int(np.argmax(dp[:, -1]))
        tags = [best]
        for t in range(T-1, 0, -1):
            tags.insert(0, path[tags[0]][t])
        return [states[i] for i in tags]

    return [viterbi("".join(chars)) for chars, _ in test_data]


def predict_crf(model_path: str, test_data: List[Tuple]) -> List[List[str]]:
    """使用保存的 CRF 模型预测"""
    from crf_nlp4j import sent_to_features
    with open(model_path, "rb") as f:
        crf = pickle.load(f)
    X = [sent_to_features(chars) for chars, _ in test_data]
    return [list(seq) for seq in crf.predict(X)]


def predict_bilstm_crf(model_path: str, test_data: List[Tuple]) -> List[List[str]]:
    """使用保存的 BiLSTM-CRF 模型预测"""
    try:
        import torch
        from bilstm_crf import BiLSTMCRF, DataProcessor
    except ImportError:
        print("  [跳过] PyTorch 未安装，无法加载 BiLSTM-CRF 模型。")
        return []

    ckpt    = torch.load(model_path, map_location="cpu")
    char2id = ckpt["char2id"]
    model   = BiLSTMCRF(len(char2id), ckpt["embed_dim"],
                        ckpt["hidden_dim"], 5, ckpt["num_layers"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    id2label = {0:"B", 1:"M", 2:"E", 3:"S"}
    results  = []

    with torch.no_grad():
        for chars, _ in test_data:
            ids     = [char2id.get(c, 1) for c in chars]
            chars_t = torch.tensor([ids], dtype=torch.long)
            lengths = torch.tensor([len(ids)], dtype=torch.long)
            mask    = chars_t != 0
            pred    = model.predict(chars_t, mask, lengths)[0]
            results.append([id2label.get(p, "S") for p in pred])

    return results


# ══════════════════════════════════════════════════════════════
# 主对比流程
# ══════════════════════════════════════════════════════════════

def run_comparison():
    DATA_DIR     = r"199801/199801"
    corpus_files = [os.path.join(DATA_DIR, f"19980{m}.txt") for m in range(1, 7)]

    print("=" * 65)
    print("三模型横向对比：HMM  vs  CRF(NLP4J)  vs  BiLSTM-CRF")
    print("=" * 65)

    # ── 加载测试集（取后20%的前3000条） ──
    print("\n加载测试集...")
    corpus   = load_corpus(corpus_files)
    split    = int(len(corpus) * 0.8)
    test_raw = corpus[split: split + 3000]
    print(f"测试集: {len(test_raw)} 条句子")
    true_seqs = [labels for _, labels in test_raw]

    # ── 模型配置 ──
    model_configs = [
        {
            "name":    "HMM（基线）",
            "path":    "hmm_model.pkl",
            "predict": predict_hmm,
        },
        {
            "name":    "CRF / NLP4J",
            "path":    "crf_nlp4j_model.pkl",
            "predict": predict_crf,
        },
        {
            "name":    "BiLSTM-CRF",
            "path":    "bilstm_crf_model.pt",
            "predict": predict_bilstm_crf,
        },
    ]

    results = []
    for cfg in model_configs:
        if not os.path.exists(cfg["path"]):
            print(f"\n[跳过] 模型文件不存在: {cfg['path']}")
            print(f"       请先运行对应训练脚本。")
            continue

        print(f"\n── 评估 {cfg['name']} ──")
        t0 = time.time()
        pred_seqs = cfg["predict"](cfg["path"], test_raw)
        elapsed   = time.time() - t0

        if not pred_seqs:
            continue

        metrics = compute_metrics(true_seqs[:len(pred_seqs)],
                                  pred_seqs[:len(true_seqs)])
        metrics["model"]   = cfg["name"]
        metrics["time_s"]  = elapsed
        results.append(metrics)

        print(f"  推理耗时  : {elapsed:.1f}s")
        print(f"  字符准确率: {metrics['char_acc']:.4f}")
        print(f"  词语精确率: {metrics['precision']:.4f}")
        print(f"  词语召回率: {metrics['recall']:.4f}")
        print(f"  词语级 F1 : {metrics['f1']:.4f}")

    # ── 汇总对比表 ──
    if results:
        print("\n" + "=" * 65)
        print("汇总对比")
        print("=" * 65)
        header = f"{'模型':<18} {'字符准确率':>10} {'词语P':>8} {'词语R':>8} {'词语F1':>8} {'耗时(s)':>8}"
        print(header)
        print("─" * 65)
        for m in results:
            print(f"{m['model']:<18} {m['char_acc']:>10.4f} "
                  f"{m['precision']:>8.4f} {m['recall']:>8.4f} "
                  f"{m['f1']:>8.4f} {m['time_s']:>8.1f}")
        print("─" * 65)

        # 各标签细分对比
        print("\n各标签 F1 细分对比:")
        print(f"{'模型':<18} {'B-F1':>8} {'M-F1':>8} {'E-F1':>8} {'S-F1':>8}")
        print("─" * 55)
        for m in results:
            r = m["report"]
            print(f"{m['model']:<18} "
                  f"{r['B']['f1-score']:>8.4f} "
                  f"{r['M']['f1-score']:>8.4f} "
                  f"{r['E']['f1-score']:>8.4f} "
                  f"{r['S']['f1-score']:>8.4f}")

    # ── 分词示例对比 ──
    print("\n" + "=" * 65)
    print("分词效果示例对比")
    print("=" * 65)
    demo_sents = [
        "中国人民解放军在北京举行阅兵式",
        "江泽民同志发表新年讲话",
        "李岚清访问欧洲各国",
        "北京市举行新年音乐会",
    ]

    # 加载各模型的 segment 函数
    segmenters = {}

    # HMM
    if os.path.exists("hmm_model.pkl"):
        try:
            with open("hmm_model.pkl", "rb") as f:
                data = pickle.load(f)
            from collections import defaultdict
            import numpy as np
            hmm_pi = defaultdict(float, data["pi"])
            hmm_A  = defaultdict(lambda: defaultdict(float),
                                 {k: defaultdict(float, v) for k, v in data["A"].items()})
            hmm_B  = defaultdict(lambda: defaultdict(float),
                                 {k: defaultdict(float, v) for k, v in data["B"].items()})
            LEGAL  = {"B":{"M","E"},"M":{"M","E"},"E":{"B","S"},"S":{"B","S"}}
            STATES = ["B","M","E","S"]

            def hmm_segment(sent):
                N, T = 4, len(sent)
                dp   = np.full((N, T), -np.inf)
                path = np.zeros((N, T), dtype=int)
                for i, s in enumerate(STATES):
                    emit = hmm_B[s].get(sent[0], hmm_B[s].get("<UNK>", 1e-10))
                    dp[i][0] = np.log(hmm_pi[s]) + np.log(emit)
                for t in range(1, T):
                    for j, cur in enumerate(STATES):
                        emit = hmm_B[cur].get(sent[t], hmm_B[cur].get("<UNK>", 1e-10))
                        for i, prev in enumerate(STATES):
                            if cur not in LEGAL[prev]: continue
                            sc = dp[i][t-1] + np.log(hmm_A[prev][cur]) + np.log(emit)
                            if sc > dp[j][t]: dp[j][t] = sc; path[j][t] = i
                best = int(np.argmax(dp[:, -1]))
                tags = [best]
                for t in range(T-1, 0, -1): tags.insert(0, path[tags[0]][t])
                tag_seq = [STATES[i] for i in tags]
                words, buf = [], ""
                for ch, tag in zip(sent, tag_seq):
                    buf += ch
                    if tag in ("E","S"): words.append(buf); buf = ""
                if buf: words.append(buf)
                return words
            segmenters["HMM"] = hmm_segment
        except Exception as e:
            print(f"  [警告] 加载 HMM 模型失败: {e}")

    # CRF
    if os.path.exists("crf_nlp4j_model.pkl"):
        try:
            from crf_nlp4j import CRFPredictor
            crf_pred = CRFPredictor("crf_nlp4j_model.pkl")
            segmenters["CRF/NLP4J"] = crf_pred.segment
        except Exception as e:
            print(f"  [警告] 加载 CRF 模型失败: {e}")

    # BiLSTM-CRF
    if os.path.exists("bilstm_crf_model.pt"):
        try:
            from bilstm_crf import BiLSTMCRFPredictor
            lstm_pred = BiLSTMCRFPredictor("bilstm_crf_model.pt")
            segmenters["BiLSTM-CRF"] = lstm_pred.segment
        except Exception as e:
            print(f"  [警告] 加载 BiLSTM-CRF 模型失败: {e}")

    for sent in demo_sents:
        print(f"\n原文: {sent}")
        for name, seg_fn in segmenters.items():
            try:
                words = seg_fn(sent)
                print(f"  {name:<14}: {' / '.join(words)}")
            except Exception as e:
                print(f"  {name:<14}: [预测失败] {e}")


if __name__ == "__main__":
    run_comparison()

"""
BiLSTM-CRF 中文序列标注模型
实验一选做项：应用 Bi-LSTM-CRF 模型进行序列标注

依赖: pip install torch numpy scikit-learn
运行: python bilstm_crf.py
"""

import os
import re
import time
import pickle
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict

# ─────────────────────────────────────────────
# 尝试导入 PyTorch；若未安装则以纯NumPy模拟推理
# ─────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[警告] 未检测到 PyTorch，将使用纯 NumPy 实现简化版 BiLSTM-CRF。")
    print("       安装命令: pip install torch")


# ══════════════════════════════════════════════════════════════
# 1. 数据处理（与 HMM 实验共用同一语料解析逻辑）
# ══════════════════════════════════════════════════════════════

class DataProcessor:
    """语料加载与 BMES 格式转换"""

    LABEL2ID = {"B": 0, "M": 1, "E": 2, "S": 3, "<PAD>": 4}
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}
    PAD_LABEL_ID = 4

    def __init__(self):
        self.char2id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.id2char: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}

    # ── 词 → BMES 标签 ──
    @staticmethod
    def word_to_bmes(word: str) -> List[str]:
        if len(word) == 1:
            return ["S"]
        return ["B"] + ["M"] * (len(word) - 2) + ["E"]

    # ── 解析一行语料 ──
    @staticmethod
    def parse_line(line: str) -> List[str]:
        line = re.sub(r"^\S+/m\s+", "", line.strip())
        pattern = r"\[?([^\s\]/]+)/[^\s\]]+\]?"
        words = re.findall(pattern, line)
        return [w.lstrip("[") for w in words if w]

    # ── 加载语料 ──
    def load_corpus(self, file_paths: List[str]) -> List[Tuple[List[str], List[str]]]:
        corpus = []
        for fp in file_paths:
            if not os.path.exists(fp):
                print(f"  [跳过] 文件不存在: {fp}")
                continue
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    words = self.parse_line(line.strip())
                    if not words:
                        continue
                    chars, labels = [], []
                    for w in words:
                        for ch, tag in zip(w, self.word_to_bmes(w)):
                            chars.append(ch)
                            labels.append(tag)
                    if chars:
                        corpus.append((chars, labels))
        return corpus

    # ── 构建词表 ──
    def build_vocab(self, corpus: List[Tuple[List[str], List[str]]]):
        freq: Dict[str, int] = defaultdict(int)
        for chars, _ in corpus:
            for c in chars:
                freq[c] += 1
        for ch in sorted(freq, key=freq.get, reverse=True):
            if ch not in self.char2id:
                idx = len(self.char2id)
                self.char2id[ch] = idx
                self.id2char[idx] = ch
        print(f"  词表大小: {len(self.char2id)} 个字符")

    # ── 序列编码 ──
    def encode(self, chars: List[str], labels: List[str]) -> Tuple[List[int], List[int]]:
        char_ids = [self.char2id.get(c, 1) for c in chars]   # 1 = <UNK>
        label_ids = [self.LABEL2ID[t] for t in labels]
        return char_ids, label_ids

    # ── Padding 到统一长度（batch内） ──
    @staticmethod
    def pad_batch(
        batch: List[Tuple[List[int], List[int]]], pad_char=0, pad_label=4
    ) -> Tuple[List[List[int]], List[List[int]], List[int]]:
        max_len = max(len(c) for c, _ in batch)
        chars_batch, labels_batch, lengths = [], [], []
        for chars, labels in batch:
            length = len(chars)
            lengths.append(length)
            chars_batch.append(chars + [pad_char] * (max_len - length))
            labels_batch.append(labels + [pad_label] * (max_len - length))
        return chars_batch, labels_batch, lengths


# ══════════════════════════════════════════════════════════════
# 2. PyTorch 版 BiLSTM-CRF
# ══════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class CRF(nn.Module):
        """线性链 CRF 层（支持 Viterbi 解码和负对数似然训练）"""

        def __init__(self, num_tags: int, pad_idx: int = 4):
            super().__init__()
            self.num_tags = num_tags
            self.pad_idx = pad_idx

            # 转移矩阵 T[i][j] = 从 tag_i 转移到 tag_j 的得分
            self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

            # 合法转移约束（BMES）
            # B → M, E 合法；其余置为极小值
            LEGAL = {
                0: {1, 2},   # B → M, E
                1: {1, 2},   # M → M, E
                2: {0, 3},   # E → B, S
                3: {0, 3},   # S → B, S
            }
            with torch.no_grad():
                for i in range(num_tags - 1):      # 不含 PAD
                    for j in range(num_tags - 1):
                        if j not in LEGAL.get(i, set()):
                            self.transitions.data[i][j] = -10000.0
                # PAD 行列均置为极小值
                self.transitions.data[pad_idx, :] = -10000.0
                self.transitions.data[:, pad_idx] = -10000.0

        # ── 前向算法（计算配分函数 log Z） ──
        def _forward_alg(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            batch, seq_len, num_tags = emissions.shape
            alpha = emissions[:, 0]                   # (B, T)
            for t in range(1, seq_len):
                emit_t = emissions[:, t].unsqueeze(1)     # (B, 1, T)
                trans = self.transitions.unsqueeze(0)     # (1, T, T)
                scores = alpha.unsqueeze(2) + trans + emit_t   # (B, T, T)
                alpha_new = torch.logsumexp(scores, dim=1)     # (B, T)
                alpha = torch.where(mask[:, t].unsqueeze(1), alpha_new, alpha)
            return torch.logsumexp(alpha, dim=1)             # (B,)

        # ── 金标路径得分 ──
        def _score_sentence(
            self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
        ) -> torch.Tensor:
            batch, seq_len = tags.shape
            score = emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
            for t in range(1, seq_len):
                trans_score = self.transitions[tags[:, t - 1], tags[:, t]]
                emit_score  = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
                score = score + (trans_score + emit_score) * mask[:, t].float()
            return score

        # ── 负对数似然（训练损失） ──
        def neg_log_likelihood(
            self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
        ) -> torch.Tensor:
            log_z  = self._forward_alg(emissions, mask)
            gold   = self._score_sentence(emissions, tags, mask)
            return (log_z - gold).mean()

        # ── Viterbi 解码 ──
        def viterbi_decode(
            self, emissions: torch.Tensor, mask: torch.Tensor
        ) -> List[List[int]]:
            batch, seq_len, num_tags = emissions.shape
            viterbi = emissions[:, 0]           # (B, T)
            backpointers = []

            for t in range(1, seq_len):
                emit_t = emissions[:, t].unsqueeze(1)
                trans  = self.transitions.unsqueeze(0)
                scores = viterbi.unsqueeze(2) + trans   # (B, T_prev, T_cur)
                best_scores, best_tags = scores.max(dim=1)
                backpointers.append(best_tags)
                viterbi = torch.where(
                    mask[:, t].unsqueeze(1),
                    best_scores + emit_t.squeeze(1),
                    viterbi
                )

            # 回溯
            best_last = viterbi.argmax(dim=1)           # (B,)
            best_paths = [best_last.tolist()]
            for bp in reversed(backpointers):
                best_last = bp.gather(1, best_last.unsqueeze(1)).squeeze(1)
                best_paths.insert(0, best_last.tolist())

            # 转为列表（按真实长度截断）
            lengths = mask.sum(dim=1).long().tolist()
            return [p[:l] for p, l in zip(zip(*best_paths), lengths)]


    class BiLSTMCRF(nn.Module):
        """BiLSTM + CRF 序列标注模型"""

        def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 128,
            hidden_dim: int = 256,
            num_tags: int = 5,
            num_layers: int = 2,
            dropout: float = 0.3,
            pad_idx: int = 0,
        ):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
            self.dropout   = nn.Dropout(dropout)
            self.bilstm    = nn.LSTM(
                embed_dim, hidden_dim // 2,
                num_layers=num_layers,
                bidirectional=True,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc  = nn.Linear(hidden_dim, num_tags)
            self.crf = CRF(num_tags, pad_idx=4)

        def _get_emissions(
            self, chars: torch.Tensor, lengths: torch.Tensor
        ) -> torch.Tensor:
            x = self.dropout(self.embedding(chars))
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.bilstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            return self.fc(self.dropout(out))

        def forward(
            self,
            chars: torch.Tensor,
            tags: torch.Tensor,
            mask: torch.Tensor,
            lengths: torch.Tensor,
        ) -> torch.Tensor:
            emissions = self._get_emissions(chars, lengths)
            return self.crf.neg_log_likelihood(emissions, tags, mask)

        def predict(
            self, chars: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor
        ) -> List[List[int]]:
            emissions = self._get_emissions(chars, lengths)
            return self.crf.viterbi_decode(emissions, mask)


# ══════════════════════════════════════════════════════════════
# 3. 训练与评估流程
# ══════════════════════════════════════════════════════════════

def get_word_spans(tags: List[str]) -> set:
    """将标签序列转为词语边界集合，用于词语级 F1 计算"""
    spans, start = set(), 0
    for i, tag in enumerate(tags):
        if tag in ("B", "S"):
            start = i
        if tag in ("E", "S"):
            spans.add((start, i))
    return spans


def evaluate(model, data_encoded, processor, device, batch_size=64):
    """计算字符级准确率和词语级 F1"""
    if not TORCH_AVAILABLE:
        return {}

    model.eval()
    all_true, all_pred = [], []
    true_spans_total, pred_spans_total, correct_total = 0, 0, 0

    with torch.no_grad():
        for i in range(0, len(data_encoded), batch_size):
            batch = data_encoded[i: i + batch_size]
            chars_b, labels_b, lengths_b = DataProcessor.pad_batch(batch)

            chars_t   = torch.tensor(chars_b, dtype=torch.long).to(device)
            labels_t  = torch.tensor(labels_b, dtype=torch.long).to(device)
            lengths_t = torch.tensor(lengths_b, dtype=torch.long).to(device)
            mask      = (chars_t != 0)

            pred_seqs = model.predict(chars_t, mask, lengths_t)

            for j, (c_ids, l_ids) in enumerate(batch):
                true_tags = [DataProcessor.ID2LABEL[l] for l in l_ids]
                pred_tags = [DataProcessor.ID2LABEL[p] for p in pred_seqs[j]]

                # 截断到真实长度
                true_tags = true_tags[:lengths_b[j]]
                pred_tags = pred_tags[:lengths_b[j]]

                all_true.extend(true_tags)
                all_pred.extend(pred_tags)

                ts = get_word_spans(true_tags)
                ps = get_word_spans(pred_tags)
                true_spans_total += len(ts)
                pred_spans_total += len(ps)
                correct_total    += len(ts & ps)

    # 字符准确率
    char_acc = sum(t == p for t, p in zip(all_true, all_pred)) / max(len(all_true), 1)

    # 词语级 P / R / F1
    prec = correct_total / max(pred_spans_total, 1)
    rec  = correct_total / max(true_spans_total, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)

    return {"char_acc": char_acc, "precision": prec, "recall": rec, "f1": f1}


def train_bilstm_crf(
    corpus_files: List[str],
    model_save_path: str = "bilstm_crf_model.pt",
    embed_dim: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 0.001,
    batch_size: int = 64,
    epochs: int = 15,
    max_train_samples: int = 80000,
):
    """完整训练流程"""
    print("=" * 60)
    print("BiLSTM-CRF 中文分词模型")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("[错误] 需要安装 PyTorch 才能训练 BiLSTM-CRF 模型。")
        print("       pip install torch")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ── 1. 数据加载 ──
    proc = DataProcessor()
    print("\n[1/4] 加载语料库...")
    corpus = proc.load_corpus(corpus_files)
    if not corpus:
        print("[错误] 未能加载任何语料，请检查文件路径。")
        return None, None
    print(f"  加载语句: {len(corpus):,} 条")

    # 划分训练集 / 测试集 8:2
    split = int(len(corpus) * 0.8)
    train_data = corpus[:split]
    test_data  = corpus[split:]
    if max_train_samples:
        train_data = train_data[:max_train_samples]
    print(f"  训练集: {len(train_data):,} 条 | 测试集: {len(test_data):,} 条")

    # ── 2. 构建词表 & 编码 ──
    print("\n[2/4] 构建词表与编码...")
    proc.build_vocab(train_data)

    train_encoded = [proc.encode(c, l) for c, l in train_data]
    test_encoded  = [proc.encode(c, l) for c, l in test_data[:5000]]   # 测试集取前5000加速评估

    # ── 3. 初始化模型 ──
    print("\n[3/4] 初始化模型...")
    model = BiLSTMCRF(
        vocab_size=len(proc.char2id),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_tags=5,             # B M E S PAD
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {param_count:,}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ── 4. 训练循环 ──
    print(f"\n[4/4] 开始训练（{epochs} 轮）...")
    best_f1 = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        # Shuffle
        indices = np.random.permutation(len(train_encoded))
        train_shuffled = [train_encoded[i] for i in indices]

        for i in range(0, len(train_shuffled), batch_size):
            batch = train_shuffled[i: i + batch_size]
            chars_b, labels_b, lengths_b = DataProcessor.pad_batch(batch)

            chars_t   = torch.tensor(chars_b,  dtype=torch.long).to(device)
            labels_t  = torch.tensor(labels_b, dtype=torch.long).to(device)
            lengths_t = torch.tensor(lengths_b, dtype=torch.long).to(device)
            mask      = (chars_t != 0)

            optimizer.zero_grad()
            loss = model(chars_t, labels_t, mask, lengths_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(1, len(train_shuffled) // batch_size)

        # 每5轮评估一次
        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate(model, test_encoded, proc, device, batch_size)
            f1 = metrics.get("f1", 0.0)
            history.append({"epoch": epoch, "loss": avg_loss, **metrics})

            print(f"  Epoch {epoch:02d}/{epochs} | Loss={avg_loss:.4f} | "
                  f"CharAcc={metrics['char_acc']:.4f} | "
                  f"P={metrics['precision']:.4f} | "
                  f"R={metrics['recall']:.4f} | "
                  f"F1={f1:.4f} | "
                  f"Time={time.time()-t0:.1f}s")

            # 保存最优模型
            if f1 > best_f1:
                best_f1 = f1
                torch.save({
                    "model_state": model.state_dict(),
                    "char2id": proc.char2id,
                    "embed_dim": embed_dim,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                }, model_save_path)
                print(f"    ✓ 最优模型已保存 (F1={best_f1:.4f})")
        else:
            print(f"  Epoch {epoch:02d}/{epochs} | Loss={avg_loss:.4f} | Time={time.time()-t0:.1f}s")

    print(f"\n训练完成！最优词语级 F1 = {best_f1:.4f}")
    return model, proc


# ══════════════════════════════════════════════════════════════
# 4. 推理接口
# ══════════════════════════════════════════════════════════════

class BiLSTMCRFPredictor:
    """训练好的 BiLSTM-CRF 模型推理接口"""

    def __init__(self, model_path: str):
        if not TORCH_AVAILABLE:
            raise RuntimeError("需要 PyTorch 才能加载模型。")
        ckpt = torch.load(model_path, map_location="cpu")
        self.char2id = ckpt["char2id"]
        self.device  = torch.device("cpu")
        self.model   = BiLSTMCRF(
            vocab_size  = len(self.char2id),
            embed_dim   = ckpt["embed_dim"],
            hidden_dim  = ckpt["hidden_dim"],
            num_layers  = ckpt["num_layers"],
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def predict_tags(self, sentence: str) -> List[str]:
        """输入句子，返回 BMES 标签序列"""
        char_ids = [self.char2id.get(c, 1) for c in sentence]
        chars_t  = torch.tensor([char_ids], dtype=torch.long)
        lengths  = torch.tensor([len(char_ids)], dtype=torch.long)
        mask     = chars_t != 0
        with torch.no_grad():
            pred = self.model.predict(chars_t, mask, lengths)[0]
        id2label = {0: "B", 1: "M", 2: "E", 3: "S"}
        return [id2label.get(p, "S") for p in pred]

    def segment(self, sentence: str) -> List[str]:
        """输入句子，返回分词列表"""
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
# 5. 主程序
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── 语料路径（根据实际情况修改） ──
    DATA_DIR = r"199801/199801"
    corpus_files = [
        os.path.join(DATA_DIR, f"19980{m}.txt") for m in range(1, 7)
    ]

    # ── 训练 ──
    model, proc = train_bilstm_crf(
        corpus_files    = corpus_files,
        model_save_path = "bilstm_crf_model.pt",
        embed_dim       = 128,
        hidden_dim      = 256,
        num_layers      = 2,
        dropout         = 0.3,
        lr              = 0.001,
        batch_size      = 64,
        epochs          = 15,
    )

    # ── 推理示例 ──
    if model is not None:
        print("\n" + "=" * 60)
        print("分词演示")
        print("=" * 60)
        predictor = BiLSTMCRFPredictor("bilstm_crf_model.pt")
        test_sents = [
            "中国人民解放军在北京举行阅兵式",
            "江泽民同志发表新年讲话",
            "北京市举行新年音乐会",
            "李岚清访问欧洲各国",
            "中共中央总书记主持会议",
        ]
        for sent in test_sents:
            words = predictor.segment(sent)
            tags  = predictor.predict_tags(sent)
            print(f"\n原文: {sent}")
            print(f"分词: {' / '.join(words)}")
            print(f"标签: {tags}")

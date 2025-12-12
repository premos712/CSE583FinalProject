import os
import math
import random
import itertools
from typing import List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score


CSV_PATH = "polybench_dataset2.csv"
OUT_PATH = "within_group_combo_results2.csv"

BATCH_SIZE = 64
LR = 1e-4
NUM_EPOCHS = 20

D_MODEL = 128
NUM_LAYERS = 3
NUM_HEADS = 4
FFN_HIDDEN = 1024
DROPOUT = 0.1
MAX_SEQ_LEN = 1024

SEED = 42

# "all"        : ONE RUN using ALL groups combined (structure+control_flow+operand+opcode union)
# "each_group" : baseline per group using ALL cols in that group
# "combos"     : all non-empty combinations within each group
#               BUT for opcode: run ONLY ONCE using ALL opcode cols (no enumeration)
# "both"       : each_group baseline + combos (same rule as above for opcode)
RUN_MODE = "combos"


def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_seq_cell(cell) -> List[float]:
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        s = s.strip()
        if not s:
            return []
        return [float(x) for x in s.split(";") if x != ""]
    elif isinstance(cell, (list, np.ndarray)):
        return [float(x) for x in cell]
    else:
        try:
            return [float(cell)]
        except Exception:
            return []


def build_sequence(row, seq_cols: List[str]) -> torch.Tensor:
    seqs = [parse_seq_cell(row[c]) for c in seq_cols]
    max_len = max((len(s) for s in seqs), default=1)

    mat = np.zeros((max_len, len(seq_cols)), dtype=np.float32)
    for j, s in enumerate(seqs):
        for i, v in enumerate(s):
            mat[i, j] = v
    return torch.tensor(mat, dtype=torch.float32)


class HotPathDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_cols: List[str]):
        self.seqs = []
        self.labels = []
        for _, row in df.iterrows():
            self.seqs.append(build_sequence(row, seq_cols))
            self.labels.append(int(row["is_hot"]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]


def collate_batch(batch):
    seqs, labels = zip(*batch)
    lengths = [s.size(0) for s in seqs]
    max_len = min(max(lengths), MAX_SEQ_LEN)

    B = len(seqs)
    D = seqs[0].size(1)

    padded = torch.zeros(B, max_len, D)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, s in enumerate(seqs):
        L = min(s.size(0), max_len)
        padded[i, :L] = s[:L]
        mask[i, :L] = True

    return padded, mask, torch.tensor(labels, dtype=torch.long)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class HotPathTransformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, D_MODEL)
        self.cls = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.pos = PositionalEncoding(D_MODEL, max_len=MAX_SEQ_LEN + 1)

        enc = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            dim_feedforward=FFN_HIDDEN,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, NUM_LAYERS)
        self.fc = nn.Linear(D_MODEL, 2)

    def forward(self, x, mask):
        B = x.size(0)
        x = self.proj(x)
        cls = self.cls.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        mask = torch.cat([cls_mask, mask], dim=1)

        x = self.pos(x)
        out = self.encoder(x, src_key_padding_mask=~mask)
        return self.fc(out[:, 0])


def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    tot = cor = 0
    loss_sum = 0.0
    for x, m, y in loader:
        x, m, y = x.to(device), m.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x, m)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        loss_sum += loss.item() * x.size(0)
        cor += (logits.argmax(-1) == y).sum().item()
        tot += x.size(0)
    return loss_sum / tot, cor / tot


@torch.no_grad()
def eval_model(model, loader, loss_fn, device):
    model.eval()
    ys, ps, preds = [], [], []
    loss_sum = 0.0
    tot = cor = 0

    for x, m, y in loader:
        x, m, y = x.to(device), m.to(device), y.to(device)
        logits = model(x, m)
        loss = loss_fn(logits, y)

        loss_sum += loss.item() * x.size(0)
        cor += (logits.argmax(-1) == y).sum().item()
        tot += x.size(0)

        ys.append(y.cpu().numpy())
        ps.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
        preds.append(logits.argmax(-1).cpu().numpy())

    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pr = np.concatenate(preds)

    auroc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    f1 = f1_score(y, pr) if len(np.unique(pr)) > 1 else np.nan

    return (loss_sum / tot, cor / tot, auroc, f1)


def main():
    set_all_seeds(SEED)
    df = pd.read_csv(CSV_PATH)

    seq_cols = [c for c in df.columns if c not in
        ["kernel", "program", "function", "path_index", "is_hot", "path_ir"]
    ]

    groups = {
        "structure": [c for c in seq_cols if any(k in c for k in
            ["path_len", "branch_count", "call_count", "loop_depth", "dom_depth"])],
        "control_flow": [c for c in seq_cols if any(k in c for k in
            ["num_preds", "num_succ", "dist_from_entry", "in_loop"])],
        "operand": [c for c in seq_cols if any(k in c for k in
            ["int_operands", "fp_operands", "ptr_operands", "vector_operands", "phi_incoming"])],
        "opcode": [c for c in seq_cols if c.startswith("op_")],
    }

    all_group_cols = []
    seen = set()
    for gname in ["structure", "control_flow", "operand", "opcode"]:
        for c in groups.get(gname, []):
            if c not in seen:
                seen.add(c)
                all_group_cols.append(c)

    train_df, test_df = train_test_split(
        df, test_size=0.5, random_state=SEED, stratify=df["is_hot"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    def run_one(run_group_name: str, cols: List[str], tag: str, idx: int):
        print("\n------------------------------")
        print(f"[{run_group_name}] {tag} | num_features={len(cols)}")

        combo_str = ", ".join(cols)
        if len(combo_str) > 200:
            combo_str = combo_str[:200] + " ... (truncated)"
        print(f"  combo_cols = [{combo_str}]")

        train_ds = HotPathDataset(train_df, cols)
        test_ds = HotPathDataset(test_df, cols)

        train_ld = DataLoader(train_ds, BATCH_SIZE, True, collate_fn=collate_batch)
        test_ld = DataLoader(test_ds, BATCH_SIZE, False, collate_fn=collate_batch)

        model = HotPathTransformer(len(cols)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, train_acc = train_epoch(model, train_ld, opt, loss_fn, device)
            print(f"  Epoch {epoch:02d}: train_loss={train_loss:.4f} acc={train_acc:.4f}")

        loss, acc, auroc, f1 = eval_model(model, test_ld, loss_fn, device)
        print(f"  FINAL: test_loss={loss:.4f} acc={acc:.4f} auroc={auroc:.4f} f1={f1:.4f}")

        results.append({
            "group": run_group_name,
            "tag": tag,
            "run_idx": idx,
            "num_features": len(cols),
            "cols": ";".join(cols),
            "test_loss": float(loss),
            "test_acc": float(acc),
            "test_auroc": float(auroc),
            "test_f1": float(f1),
        })

    if RUN_MODE == "all":
        if not all_group_cols:
            print("ERROR: all_group_cols is empty (check grouping rules).")
        else:
            run_one("all_groups", all_group_cols, "ALL", 0)

    elif RUN_MODE == "each_group":
        for gname, gcols in groups.items():
            if gcols:
                run_one(gname, gcols, "ALL", 0)

    elif RUN_MODE == "combos":
        for gname, gcols in groups.items():
            if not gcols:
                continue

            if gname == "opcode":
                run_one(gname, gcols, "ALL", 0)
                continue

            idx = 0
            for r in range(1, len(gcols) + 1):
                for comb in itertools.combinations(gcols, r):
                    idx += 1
                    run_one(gname, list(comb), f"C{idx}", idx)

    elif RUN_MODE == "both":
        for gname, gcols in groups.items():
            if not gcols:
                continue

            run_one(gname, gcols, "ALL", 0)

            if gname == "opcode":
                continue

            idx = 0
            for r in range(1, len(gcols) + 1):
                for comb in itertools.combinations(gcols, r):
                    idx += 1
                    run_one(gname, list(comb), f"C{idx}", idx)

    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")

    pd.DataFrame(results).to_csv(OUT_PATH, index=False)
    print("\nSaved:", OUT_PATH)


if __name__ == "__main__":
    main()

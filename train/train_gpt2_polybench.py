#!/usr/bin/env python3
import os
import random
import numpy as np
import torch
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    GPT2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed as hf_set_seed,
    DataCollatorWithPadding,
)
from sklearn.metrics import roc_auc_score, f1_score
import transformers

SEED = 666

def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds()
hf_set_seed(SEED)
print("Transformers version:", transformers.__version__)

DATA_PATH = "polybench_dataset2.csv"
df = pd.read_csv(DATA_PATH)

exclude_cols = ["kernel", "program", "function", "path_index", "is_hot", "path_ir"]
feature_cols = [c for c in df.columns if c not in exclude_cols]

structure_cols = [
    c for c in feature_cols
    if any(key in c for key in ["path_len","inst_count","branch_count","call_count","loop_depth","dom_depth"])
]
control_flow_cols = [
    c for c in feature_cols
    if any(key in c for key in ["num_preds","num_succ","dist_from_entry","in_loop"])
]
operand_cols = [
    c for c in feature_cols
    if any(key in c for key in ["int_operands","fp_operands","ptr_operands","vector_operands","phi_incoming"])
]
opcode_cols = [c for c in feature_cols if c.startswith("op_")]

groups = {
    "structure": structure_cols,
    "control_flow": control_flow_cols,
    "operand": operand_cols,
    "opcode": opcode_cols,
}

print("Structure cols:", structure_cols)
print("Control-flow cols:", control_flow_cols)
print("Operand cols:", operand_cols)
print("Opcode cols:", opcode_cols)

df["labels"] = df["is_hot"].astype(int)

selected_groups = ["structure", "control_flow", "operand", "opcode"]
selected_cols = []
for g in selected_groups:
    selected_cols += groups[g]

def row_to_text(row):
    return " ".join(f"{col}={row[col]}" for col in selected_cols)

df["text"] = df.apply(row_to_text, axis=1)

# kernel-level split
rng = np.random.default_rng(SEED)
kernels = df["kernel"].unique()
rng.shuffle(kernels)

n = len(kernels)
n_train = int(0.50 * n)
n_eval  = int(0.25 * n)

train_k = kernels[:n_train]
eval_k  = kernels[n_train:n_train + n_eval]
test_k  = kernels[n_train + n_eval:]

train_df = df[df["kernel"].isin(train_k)].reset_index(drop=True)
eval_df  = df[df["kernel"].isin(eval_k)].reset_index(drop=True)
test_df  = df[df["kernel"].isin(test_k)].reset_index(drop=True)

print("Num kernels:", n)
print("Train kernels:", len(train_k), "rows:", len(train_df))
print("Eval  kernels:", len(eval_k),  "rows:", len(eval_df))
print("Test  kernels:", len(test_k),  "rows:", len(test_df))

train_ds = Dataset.from_pandas(train_df[["text", "labels"]])
eval_ds  = Dataset.from_pandas(eval_df[["text", "labels"]])
test_ds  = Dataset.from_pandas(test_df[["text", "labels"]])

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.config.pad_token_id = tokenizer.pad_token_id

MAX_LENGTH = 256

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

train_ds = train_ds.map(tokenize_function, batched=True, remove_columns=["text"])
eval_ds  = eval_ds.map(tokenize_function,  batched=True, remove_columns=["text"])
test_ds  = test_ds.map(tokenize_function,  batched=True, remove_columns=["text"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = (preds == labels).mean()
    f1 = f1_score(labels, preds)
    try:
        auroc = roc_auc_score(labels, logits[:, 1])
    except Exception:
        auroc = float("nan")
    return {"accuracy": acc, "f1": f1, "auroc": auroc}

output_dir = "./gpt2-polybench-is_hot"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    do_train=True,
    do_eval=True,
    num_train_epochs=3,
    learning_rate=1e-4,

    # logs
    logging_strategy="steps",
    logging_steps=100,

    eval_strategy="steps",
    eval_steps=100,

    # optional: avoid wandb
    report_to="none",

    seed=SEED,
    data_seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()

    print("\n=== Eval set ===")
    print(trainer.evaluate(eval_ds))

    print("\n=== Test set ===")
    print(trainer.evaluate(test_ds))

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

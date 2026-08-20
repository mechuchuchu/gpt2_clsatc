import os
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    AutoConfig
)
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm


# ============================================================
# 0. Configuration
# ============================================================

SEED = 42

MODEL_NAME = "distilbert-base-uncased"

BATCH_SIZE = 16
EPOCHS = 3

# 동일 조건 비교를 위해 동일한 learning rate 사용
LR = 2e-5
WEIGHT_DECAY = 0.01

MAX_LENGTH = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 2. Load IMDb
# ============================================================

dataset = load_dataset("imdb")

print(dataset)
print(dataset["train"][0])


# ============================================================
# 3. Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)



def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


tokenized = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

tokenized = tokenized.rename_column("label", "labels")

tokenized.set_format("torch")

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


train_loader = DataLoader(
    tokenized["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
)

test_loader = DataLoader(
    tokenized["test"],
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
)


# ============================================================
# 4. Model 1: Embedding-only
# ============================================================

class EmbeddingOnly(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_config(config)

        # Encoder freeze
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):

        with torch.no_grad():

            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # DistilBERT의 첫 token representation
        cls_embedding = outputs.last_hidden_state[:, 0]

        logits = self.classifier(cls_embedding)

        return logits


# ============================================================
# 5. Model 2: Embedding + Attention
# ============================================================

class EmbeddingAttention(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_config(config)

        # Encoder freeze
        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size

        # Trainable attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):

        with torch.no_grad():

            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # [batch, seq_len, hidden]
        hidden = outputs.last_hidden_state

        # Attention score
        scores = self.attention(hidden).squeeze(-1)

        # Padding token masking
        scores = scores.masked_fill(
            attention_mask == 0,
            -1e9
        )

        weights = torch.softmax(scores, dim=1)

        # Weighted sum
        pooled = torch.sum(
            hidden * weights.unsqueeze(-1),
            dim=1
        )

        logits = self.classifier(pooled)

        return logits


# ============================================================
# 6. Model 3: Full training
# ============================================================

class FullTraining(nn.Module):

    def __init__(self, model_name):
        super().__init__()

        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_config(config)

        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_embedding = outputs.last_hidden_state[:, 0]

        logits = self.classifier(cls_embedding)

        return logits


# ============================================================
# 7. Parameter count
# ============================================================

def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ============================================================
# 8. Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loader):

    model.eval()

    all_preds = []
    all_labels = []

    for batch in tqdm(
        loader,
        desc="Evaluating",
        leave=False
    ):

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        preds = torch.argmax(
            logits,
            dim=-1
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        average="binary"
    )

    return accuracy, f1


# ============================================================
# 9. Training
# ============================================================

def train_model(model, name):

    model = model.to(DEVICE)

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss()

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    total_params, trainable_params_count = count_parameters(model)

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params_count:,}")

    best_f1 = 0
    best_accuracy = 0

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}"
        )

        for batch in progress:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            progress.set_postfix(
                loss=loss.item()
            )

        avg_loss = total_loss / len(train_loader)

        accuracy, f1 = evaluate(
            model,
            test_loader
        )

        print(
            f"Epoch {epoch + 1}: "
            f"loss={avg_loss:.4f}, "
            f"accuracy={accuracy:.4f}, "
            f"F1={f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_accuracy = accuracy

    return best_accuracy, best_f1


# ============================================================
# 10. Run all three experiments
# ============================================================

results = []


# ------------------------------------------------------------
# 1. Embedding-only
# ------------------------------------------------------------

model = EmbeddingOnly(MODEL_NAME)

accuracy, f1 = train_model(
    model,
    "1. Embedding-only"
)

results.append({
    "Model": "Embedding-only",
    "Accuracy": accuracy,
    "F1": f1,
})


del model
torch.cuda.empty_cache()


# ------------------------------------------------------------
# 2. Embedding + Attention
# ------------------------------------------------------------

model = EmbeddingAttention(MODEL_NAME)

accuracy, f1 = train_model(
    model,
    "2. Embedding + Attention"
)

results.append({
    "Model": "Embedding + Attention",
    "Accuracy": accuracy,
    "F1": f1,
})


del model
torch.cuda.empty_cache()


# ------------------------------------------------------------
# 3. Full training
# ------------------------------------------------------------

model = FullTraining(MODEL_NAME)

accuracy, f1 = train_model(
    model,
    "3. Full training"
)

results.append({
    "Model": "Full training",
    "Accuracy": accuracy,
    "F1": f1,
})


# ============================================================
# 11. Final result table
# ============================================================

import pandas as pd

results_df = pd.DataFrame(results)

results_df["Accuracy"] = (
    results_df["Accuracy"] * 100
).round(2)

results_df["F1"] = (
    results_df["F1"] * 100
).round(2)

print("\n")
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))

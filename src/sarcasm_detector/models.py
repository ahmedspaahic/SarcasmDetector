"""The three models compared in the project.

1. TF-IDF + Logistic Regression (baseline)
2. Static BERT [CLS] embeddings + Logistic Regression
3. Fine-tuned BERT (BertForSequenceClassification)

The BERT models import torch/transformers lazily, so the baseline runs without them.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import RANDOM_STATE

BERT_MODEL_NAME = "bert-base-uncased"


# =========================
# 1. Baseline: TF-IDF + Logistic Regression
# =========================
def train_tfidf_lr(x_training, y_training):
    """Fit a TF-IDF vectorizer and a logistic regression classifier.

    Punctuation such as `!` and `?` is kept, since it may be a signal for sarcasm.
    Returns `(vectorizer, model)`.
    """
    vectorizer = TfidfVectorizer()
    x_training_vec = vectorizer.fit_transform(x_training)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(x_training_vec, y_training)
    return vectorizer, model


def plot_top_coefficients(vectorizer, model, path=None, top_n=10):
    """Words with the largest positive (sarcastic) and negative (not sarcastic) coefficients."""
    feature_names = vectorizer.get_feature_names_out()
    coefficients  = model.coef_[0]

    top_sarcastic     = np.argsort(coefficients)[-top_n:][::-1]   # highest coef → sarcastic
    top_not_sarcastic = np.argsort(coefficients)[:top_n]          # lowest coef  → not sarcastic

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor="#F8F9FB")
    fig.suptitle("Top Words Driving Each Prediction",
                 fontsize=15, fontweight="bold", color="#1a1a2e", y=1.01)

    for ax, indices, color, title in [
        (ax1, top_not_sarcastic, "#4C72B0", "→ Not Sarcastic"),
        (ax2, top_sarcastic,     "#DD8452", "→ Sarcastic"),
    ]:
        words  = [feature_names[i] for i in indices][::-1]
        scores = [coefficients[i]  for i in indices][::-1]
        bars = ax.barh(words, scores, color=color, alpha=0.85,
                       edgecolor="white", linewidth=0.8)
        for bar, score in zip(bars, scores):
            ax.text(score + (0.005 if score > 0 else -0.005),
                    bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}", va="center", fontsize=8, color="#444444")
        ax.set_facecolor("#F8F9FB")
        ax.set_title(title, fontsize=13, fontweight="bold", color=color, pad=10)
        ax.set_xlabel("Coefficient (log-odds)", color="#444444")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")
        ax.tick_params(colors="#444444", left=False)

    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#F8F9FB")
    plt.close(fig)


# =========================
# 2. BERT embeddings + Logistic Regression
# =========================
def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_embeddings(texts, tokenizer, bert_model, device, batch_size=32, max_length=64):
    """[CLS] token embeddings (768-dim for bert-base) of a list/Series of texts."""
    import torch
    from tqdm import tqdm

    texts = list(texts)
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        encoded = tokenizer(
            texts[i:i + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = bert_model(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device)
            )
            # CLS token embedding
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())

    return np.vstack(embeddings)


def bert_embeddings_lr(x_training, y_training, x_test):
    """Train logistic regression on frozen BERT embeddings; return predictions for `x_test`."""
    from transformers import BertTokenizer, BertModel

    device = get_device()
    print(f"Using device: {device}")
    tokenizer  = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = BertModel.from_pretrained(BERT_MODEL_NAME).to(device)
    bert_model.eval()

    print("\nGenerating BERT embeddings for training data...")
    x_train_bert = generate_embeddings(x_training, tokenizer, bert_model, device)
    print("\nGenerating BERT embeddings for test data...")
    x_test_bert = generate_embeddings(x_test, tokenizer, bert_model, device)

    model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    model.fit(x_train_bert, y_training)
    return model.predict(x_test_bert)


# =========================
# 3. Fine-tuned BERT
# =========================
def finetune_bert(train_texts, train_labels, test_texts, test_labels,
                  epochs=1, batch_size=32, max_length=32, learning_rate=2e-5,
                  output_dir="results/bert-finetuned"):
    """Fine-tune `bert-base-uncased` for binary classification; return predictions for `test_texts`.

    `max_length=32` covers 99.9 % of headlines (mean BERT token length is 14.8).
    """
    import pandas as pd
    import torch
    from datasets import Dataset
    from transformers import (BertTokenizer, BertForSequenceClassification,
                              Trainer, TrainingArguments, DataCollatorWithPadding)

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            padding="max_length",  # forces all sequences to same length
            truncation=True,       # cuts sentences longer than max_length
            max_length=max_length
        )

    def to_dataset(texts, labels):
        ds = Dataset.from_pandas(pd.DataFrame({"text": list(texts), "label": list(labels)}))
        ds = ds.map(tokenize_function, batched=True).remove_columns(["text"])
        ds.set_format("torch")
        return ds

    train_dataset = to_dataset(train_texts, train_labels)
    test_dataset  = to_dataset(test_texts, test_labels)

    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="no",               # no evaluation during training (speed)
        save_strategy="no",               # no checkpoints (speed)
        logging_steps=1000,
        learning_rate=learning_rate,
        weight_decay=0.01,                # regularization against overfitting
        fp16=torch.cuda.is_available(),   # mixed precision only on GPU
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )
    trainer.train()

    pred_output = trainer.predict(test_dataset)
    return np.argmax(pred_output.predictions, axis=1)

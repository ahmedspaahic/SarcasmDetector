"""Train and evaluate the sarcasm detection models.

Usage (from the repository root):
    python scripts/run_experiments.py                         # all three models
    python scripts/run_experiments.py --models tfidf          # baseline only (no GPU needed)
    python scripts/run_experiments.py --data path/to/Sarcasm_Headlines_Dataset_v2.json

Metrics are printed and saved, together with confusion matrices, to `results/`.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sarcasm_detector.data import load_data, split_data
from sarcasm_detector.evaluation import evaluate, print_metrics, plot_confusion_matrix
from sarcasm_detector import models

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL_CHOICES = ["tfidf", "bert-lr", "bert-ft"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=MODEL_CHOICES)
    parser.add_argument("--data", default=None,
                        help="local dataset folder or JSON file (default: download from Kaggle)")
    parser.add_argument("--ft-train-size", type=int, default=8000,
                        help="training samples used for fine-tuning (speed optimization)")
    parser.add_argument("--ft-test-size", type=int, default=2000,
                        help="test samples used for fine-tuning evaluation")
    parser.add_argument("--ft-epochs", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_data(args.data)
    x_training, x_test, y_training, y_test = split_data(df)
    print(f"Training samples: {len(x_training)}, test samples: {len(x_test)}")

    results = []

    if "tfidf" in args.models:
        vectorizer, lr_model = models.train_tfidf_lr(x_training, y_training)
        print(f"Number of TF-IDF features: {len(vectorizer.get_feature_names_out())}")
        y_pred = lr_model.predict(vectorizer.transform(x_test))
        metrics = evaluate(y_test, y_pred)
        print_metrics("BASELINE — TF-IDF + LOGISTIC REGRESSION", metrics)
        plot_confusion_matrix(y_test, y_pred, "Confusion Matrix — TF-IDF + LR",
                              os.path.join(RESULTS_DIR, "confusion_matrix_tfidf_lr.png"))
        models.plot_top_coefficients(vectorizer, lr_model,
                                     os.path.join(RESULTS_DIR, "top_coefficients.png"))
        results.append({"model": "TF-IDF + Logistic Regression", **metrics})

    if "bert-lr" in args.models:
        y_pred = models.bert_embeddings_lr(x_training, y_training, x_test)
        metrics = evaluate(y_test, y_pred)
        print_metrics("BERT EMBEDDINGS + LOGISTIC REGRESSION", metrics)
        plot_confusion_matrix(y_test, y_pred, "Confusion Matrix — BERT + LR",
                              os.path.join(RESULTS_DIR, "confusion_matrix_bert_lr.png"))
        results.append({"model": "BERT Embeddings + Logistic Regression", **metrics})

    if "bert-ft" in args.models:
        train_texts  = x_training.tolist()[:args.ft_train_size]
        train_labels = y_training.tolist()[:args.ft_train_size]
        test_texts   = x_test.tolist()[:args.ft_test_size]
        test_labels  = y_test.tolist()[:args.ft_test_size]

        y_pred = models.finetune_bert(train_texts, train_labels, test_texts, test_labels,
                                      epochs=args.ft_epochs,
                                      output_dir=os.path.join(RESULTS_DIR, "bert-finetuned"))
        metrics = evaluate(test_labels, y_pred)
        print_metrics("FINE-TUNED BERT", metrics)
        plot_confusion_matrix(test_labels, y_pred, "Confusion Matrix — Fine-Tuned BERT",
                              os.path.join(RESULTS_DIR, "confusion_matrix_finetuned_bert.png"),
                              cmap="Greens")
        results.append({"model": "Fine-Tuned BERT", **metrics})

    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)
    print("\n" + summary.to_string(index=False, float_format="%.4f"))
    print(f"\nResults saved to {os.path.normpath(RESULTS_DIR)}")


if __name__ == "__main__":
    main()

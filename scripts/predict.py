"""Classify your own headlines with the TF-IDF + Logistic Regression baseline.

Usage (from the repository root):
    python scripts/predict.py "area man passionate defender of what he imagines constitution to be"
    python scripts/predict.py                # interactive mode
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sarcasm_detector import LABELS
from sarcasm_detector.data import load_data, split_data
from sarcasm_detector.models import train_tfidf_lr


def main():
    df = load_data(os.environ.get("SARCASM_DATA"))
    x_training, _, y_training, _ = split_data(df)
    vectorizer, model = train_tfidf_lr(x_training, y_training)

    def classify(headline):
        proba = model.predict_proba(vectorizer.transform([headline.lower()]))[0]
        label = int(proba[1] >= 0.5)
        print(f"{LABELS[label]:<14} (p_sarcastic = {proba[1]:.2f})  {headline}")

    if len(sys.argv) > 1:
        for headline in sys.argv[1:]:
            classify(headline)
    else:
        print("Enter a headline (empty line to quit):")
        while headline := input("> ").strip():
            classify(headline)


if __name__ == "__main__":
    main()

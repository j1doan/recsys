"""Predict shelves for a description, using the saved local fitted model."""
import argparse
import json
from pathlib import Path

import joblib


def predict(text, model_path='artifacts/full/selected_model.joblib'):
    # Load only artifacts created by this project; joblib is executable pickle.
    bundle = joblib.load(model_path)
    scores = bundle['classifier'].decision_function(bundle['vectorizer'].transform([text]))[0]
    return [{'shelf':label, 'score':float(score), 'threshold':float(threshold),
             'predicted':bool(score >= threshold)}
            for label, score, threshold in zip(bundle['labels'], scores, bundle['thresholds'])]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('description')
    parser.add_argument('--model', type=Path, default=Path('artifacts/full/selected_model.joblib'))
    args = parser.parse_args()
    print(json.dumps(predict(args.description, args.model), indent=2))

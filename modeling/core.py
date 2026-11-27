"""Reusable data, labeling, split, feature, and threshold functions.

Only description text enters the vectorizer. Shelves construct targets;
work_id is used only to keep editions together during splitting.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion

# Identical exclusions to the group's original code.py, for comparability.
EXCLUDE = {
    'to-read', 'currently-reading', 'read', 'favorites', 'favorite', 'favourites',
    'owned', 'own', 'my-books', 'books-i-own', 'default', 'library', 'wish-list',
    'wishlist', 'to-buy', 'ebook', 'ebooks', 'e-book', 'kindle', 'audiobook',
    'audiobooks', 'dnf', 'abandoned', 'did-not-finish', 're-read', 'romance', 'fiction',
}
ALIASES = {
    'historical-romance': 'historical',
    'contemporary-romance': 'contemporary',
    'paranormal-romance': 'paranormal',
}
DATA_URL = ('https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/'
            'byGenre/goodreads_books_romance.json.gz')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def content_shelves(shelves):
    # Dict semantics deliberately match the original repository implementation.
    return {s['name']: int(s['count']) for s in shelves
            if s['name'] not in EXCLUDE}


def read_books(path, limit=None):
    """Stream raw JSON, retain only fields needed for modeling and audits."""
    rows, raw_count, with_text, low_votes, missing_work = [], 0, 0, 0, 0
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            raw_count += 1
            b = json.loads(line)
            text = b.get('description') or ''
            if not text.strip():
                continue
            with_text += 1
            shelves = content_shelves(b.get('popular_shelves', []))
            votes = sum(shelves.values())
            if votes < 10:
                low_votes += 1
                continue
            work = str(b.get('work_id') or '').strip()
            if work in ('', '0', 'None'):
                missing_work += 1
                group = 'book:' + str(b['book_id'])
            else:
                group = 'work:' + work
            normalized = ' '.join(text.lower().split())
            rows.append({
                'book_id': str(b['book_id']), 'work_id': work, 'group': group,
                'title': b.get('title', ''), 'description': text,
                'shelves': shelves, 'votes': votes,
                'description_hash': hashlib.sha256(normalized.encode()).hexdigest(),
            })
    return pd.DataFrame(rows), {
        'raw_books': raw_count, 'nonempty_descriptions': with_text,
        'dropped_insufficient_votes': low_votes, 'eligible_books': len(rows),
        'missing_work_id_fallbacks': missing_work, 'partial_data': limit is not None,
    }


def split_books(df):
    """Preserve original outer split (seed 0); reserve 20% of development for val."""
    dev, test = next(GroupShuffleSplit(test_size=.2, random_state=0)
                     .split(df, groups=df['group']))
    ti, vi = next(GroupShuffleSplit(test_size=.2, random_state=1)
                  .split(df.iloc[dev], groups=df['group'].iloc[dev]))
    splits = {'train': dev[ti], 'validation': dev[vi], 'test': test}
    groups = {k: set(df['group'].iloc[v]) for k, v in splits.items()}
    assert not (groups['train'] & groups['validation'])
    assert not (groups['train'] & groups['test'])
    assert not (groups['validation'] & groups['test'])
    assert len(np.unique(np.concatenate(list(splits.values())))) == len(df)
    return splits


def select_labels(df, train, k=15):
    """Select vocabulary only from training targets; no validation/test label peeking."""
    counts = Counter(name for d, v in zip(df['shelves'].iloc[train],
                                         df['votes'].iloc[train])
                     for name, count in d.items() if count / v >= .1)
    labels = [n for n, _ in counts.most_common(k)]
    if len(labels) != k:
        raise ValueError(f'Need {k} labels, found {len(labels)}')
    return labels


def make_targets(df, labels):
    return np.array([[d.get(s, 0) / votes >= .1 for s in labels]
                     for d, votes in zip(df['shelves'], df['votes'])], dtype=np.uint8)


def merge_targets(y, labels):
    """OR original positive indicators, without summing overlapping reader votes.

    This is a *different* coarser target; never compare its F1 as a direct gain
    on the unmerged 15-shelf task. No top-K reselection is performed.
    """
    names = list(dict.fromkeys(ALIASES.get(s, s) for s in labels))
    merged = np.zeros((len(y), len(names)), dtype=np.uint8)
    for j, s in enumerate(labels):
        merged[:, names.index(ALIASES.get(s, s))] |= y[:, j]
    return merged, names


def make_features(kind):
    word = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=5,
                          stop_words='english', sublinear_tf=True,
                          dtype=np.float64)
    if kind == 'word':
        return word
    if kind != 'word_char':
        raise ValueError(kind)
    # Character fragments capture morphology, names, spelling, and non-English
    # descriptions. Each block is L2 normalized; weights are fixed in advance.
    char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=5,
                          max_features=60_000, sublinear_tf=True, dtype=np.float64)
    return FeatureUnion([('word', word), ('char', char)],
                        transformer_weights={'word': 1., 'char': .5}, n_jobs=1)


def best_f1_threshold(y, scores):
    """Exact F1 optimum among distinct score boundaries, using >= consistently.

    Whole equal-score blocks are considered together. On ties retain the
    highest threshold (fewest predicted positives). No-positive labels predict
    none; no validation labels or classifier probabilities are assumed here.
    """
    y, scores = np.asarray(y).ravel(), np.asarray(scores).ravel()
    if not len(y) or not np.isfinite(scores).all():
        raise ValueError('Need nonempty finite scores')
    if not y.sum():
        return float(np.nextafter(scores.max(), np.inf))
    order = np.argsort(-scores, kind='stable')
    s, truth = scores[order], y[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    tp = np.cumsum(truth, dtype=np.float64)[ends]
    f1 = 2 * tp / (ends + 1 + truth.sum())
    return float(s[ends[np.argmax(f1)]])


def tune_thresholds(y, scores):
    return np.array([best_f1_threshold(y[:, j], scores[:, j])
                     for j in range(y.shape[1])])


def metrics(y, scores, thresholds):
    pred = scores >= thresholds
    valid = (y.sum(axis=0) > 0) & (y.sum(axis=0) < len(y))
    auc = [roc_auc_score(y[:, j], scores[:, j]) for j in np.flatnonzero(valid)]
    return {
        'micro_f1': float(f1_score(y, pred, average='micro', zero_division=0)),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'macro_auc': float(np.mean(auc)) if auc else None,
        'macro_ap': float(average_precision_score(y, scores, average='macro')),
        'micro_precision': float(precision_score(y, pred, average='micro', zero_division=0)),
        'micro_recall': float(recall_score(y, pred, average='micro', zero_division=0)),
        'predicted_labels_per_book': float(pred.sum(axis=1).mean()),
        'auc_valid_labels': int(valid.sum()),
    }


def per_label_metrics(y, scores, thresholds, labels):
    pred = scores >= thresholds
    return pd.DataFrame({
        'shelf': labels, 'test_positives': y.sum(axis=0),
        'threshold': np.broadcast_to(thresholds, y.shape[1]),
        'f1': f1_score(y, pred, average=None, zero_division=0),
        'precision': precision_score(y, pred, average=None, zero_division=0),
        'recall': recall_score(y, pred, average=None, zero_division=0),
        'auc': [roc_auc_score(y[:, j], scores[:, j])
                if 0 < y[:, j].sum() < len(y) else np.nan
                for j in range(y.shape[1])],
        'average_precision': [average_precision_score(y[:, j], scores[:, j])
                              for j in range(y.shape[1])],
    })

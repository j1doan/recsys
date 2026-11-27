"""Exploratory data analysis of the Goodreads romance metadata.

Reads the raw file once, rebuilds the modeling sample with the same rules as
``modeling.core`` (same exclusion list, 10-vote floor, 10% vote share, label
order from ``reports/full/selection.json``) and joins the committed split IDs.
Writes small summary tables and figures to ``reports/eda/``. Trains nothing.

    python -m scripts.download_romance
    python -m scripts.eda_romance
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from modeling.core import content_shelves

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'goodreads_books_romance.json.gz'
FULL = ROOT / 'reports' / 'full'
OUT = ROOT / 'reports' / 'eda'
LABELS = json.loads((FULL / 'selection.json').read_text())['labels']
SPLIT_ORDER = ['train', 'validation', 'test']

# Reference palette (dataviz skill): one hue for magnitude, fixed slot order.
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SPLIT_COLORS = dict(zip(SPLIT_ORDER, [BLUE, ORANGE, AQUA]))
METADATA_FIELDS = ['publication_year', 'ratings_count', 'average_rating',
                   'num_pages', 'language_code', 'publisher', 'format',
                   'is_ebook', 'isbn', 'series', 'similar_books']


def number(value):
    try:
        return float(value) if value not in (None, '') else np.nan
    except (TypeError, ValueError):
        return np.nan


def keyword_hits(text):
    """Same literal shelf-name match as the keyword baseline in modeling/run.py."""
    t = ' ' + re.sub(r'[^\w]+', ' ', text.lower()) + ' '
    return [(' ' + s.replace('-', ' ') + ' ') in t for s in LABELS]


def read_raw():
    rows, ids, similar = [], set(), []
    missing = Counter()
    with gzip.open(DATA, 'rt', encoding='utf-8') as f:
        for line in f:
            b = json.loads(line)
            ids.add(str(b['book_id']))
            similar.append(b.get('similar_books') or [])
            for field in METADATA_FIELDS:
                if b.get(field) in (None, '', []):
                    missing[field] += 1
            text = (b.get('description') or '').strip()
            shelves = content_shelves(b.get('popular_shelves') or [])
            votes = sum(shelves.values())
            share = [shelves.get(s, 0) / votes if votes else 0.0 for s in LABELS]
            top = max(shelves.items(), key=lambda kv: kv[1])[0] if shelves else ''
            rows.append({
                'book_id': str(b['book_id']),
                'has_description': bool(text),
                'words': len(text.split()),
                'description_hash': (hashlib.sha256(' '.join(text.lower().split())
                                     .encode()).hexdigest() if text else ''),
                'has_html': bool(re.search(r'<[a-z/][^>]*>', text, re.I)),
                'content_votes': votes,
                'distinct_content_shelves': len(shelves),
                'top_shelf': top,
                'language': b.get('language_code') or '(empty)',
                'year': number(b.get('publication_year')),
                'ratings_count': number(b.get('ratings_count')),
                'average_rating': number(b.get('average_rating')),
                'num_pages': number(b.get('num_pages')),
                'is_ebook': b.get('is_ebook') == 'true',
                **{f'share::{s}': v for s, v in zip(LABELS, share)},
                **{f'kw::{s}': v for s, v in zip(LABELS, keyword_hits(text))},
            })
    links = sum(len(s) for s in similar)
    inside = sum(str(x) in ids for s in similar for x in s)
    return pd.DataFrame(rows), missing, {'links': links, 'inside_file': inside}


def style():
    plt.rcParams.update({
        'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
        'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': .6, 'axes.axisbelow': True, 'figure.dpi': 140,
        'axes.titleweight': 'bold', 'axes.titlesize': 11,
        'axes.titlelocation': 'left'})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, facecolor='white')
    plt.close(fig)


def figures(df, eligible, y, flow, support, co, near, kw):
    style()
    for old in OUT.glob('*.png'):
        old.unlink()

    # 1. Sample flow.
    fig, ax = plt.subplots(figsize=(8, 3.2))
    names, values = list(flow)[::-1], list(flow.values())[::-1]
    ax.barh(names, values, color=BLUE, height=.55)
    for i, v in enumerate(values):
        ax.text(v + 3000, i, f'{v:,}', va='center', color=INK, fontsize=9)
    ax.set_xlim(0, max(values) * 1.15)
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Books')
    ax.set_title('From raw file to modeling splits')
    save(fig, 'sample_flow.png')

    # 2. Description length, eligible books, log-x.
    words = eligible['words'].clip(lower=1)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.hist(words, bins=np.logspace(0, np.log10(words.max()), 60), color=BLUE)
    ax.set_xscale('log')
    for q, lab, ha, nudge in [(.5, 'median', 'right', .95), (.9, '90th pct', 'left', 1.05)]:
        v = words.quantile(q)
        ax.axvline(v, color=INK2, lw=1, ls='--')
        ax.text(v * nudge, ax.get_ylim()[1] * .92, f'{lab} {v:.0f}', color=INK2,
                fontsize=9, ha=ha)
    ax.set_xlabel('Words in description (log scale)')
    ax.set_ylabel('Eligible books')
    ax.set_title('Descriptions are short: most are one or two paragraphs')
    save(fig, 'description_lengths.png')

    # 3. Content votes per eligible book.
    fig, ax = plt.subplots(figsize=(8, 3.4))
    v = eligible['content_votes']
    ax.hist(v, bins=np.logspace(1, np.log10(v.max()), 60), color=BLUE)
    ax.set_xscale('log')
    ax.axvline(v.median(), color=INK2, lw=1, ls='--')
    ax.text(v.median() * 1.08, ax.get_ylim()[1] * .92, f'median {v.median():.0f}',
            color=INK2, fontsize=9)
    ax.set_xlabel('Content-shelf votes per book (log scale; floor is 10)')
    ax.set_ylabel('Eligible books')
    ax.set_title('Targets rest on a median of 137 votes; 1 in 10 books has under 30')
    save(fig, 'content_votes.png')

    # 4. Label prevalence by split.
    fig, ax = plt.subplots(figsize=(8, 5))
    order = support.sort_values('train_prevalence')
    pos = np.arange(len(order))
    for k, split in enumerate(SPLIT_ORDER):
        ax.barh(pos + (1 - k) * .27, order[f'{split}_prevalence'] * 100, height=.25,
                color=SPLIT_COLORS[split], label=split)
    ax.set_yticks(pos, order['shelf'])
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Books with the shelf as a positive label (%)')
    ax.set_title('Every shelf is rare, and prevalence is stable across splits')
    ax.legend(frameon=False, loc='lower right')
    save(fig, 'label_support.png')

    # 5. Labels per book.
    counts = pd.Series(y.sum(1)).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.bar(counts.index.astype(str), counts.values, color=BLUE, width=.6)
    for b, c in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, c, f'{c / counts.sum():.1%}',
                ha='center', va='bottom', fontsize=9, color=INK)
    ax.grid(axis='x', visible=False)
    ax.set_xlabel('Positive labels among the 15')
    ax.set_ylabel('Eligible books')
    ax.set_title('Two thirds of books have none of the 15 labels')
    save(fig, 'labels_per_book.png')

    # 6. Co-occurrence: P(column | row) on training books.
    fig, ax = plt.subplots(figsize=(8.2, 7))
    im = ax.imshow(co.values, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=60, ha='right')
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.grid(False)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if i != j and co.values[i, j] >= .2:
                ax.text(j, i, f'{co.values[i, j]:.2f}', ha='center', va='center',
                        fontsize=7, color='white' if co.values[i, j] > .55 else INK)
    fig.colorbar(im, ax=ax, fraction=.04, label='P(column shelf | row shelf)')
    ax.set_title('Shelf co-occurrence, training books (values >= 0.20 shown)')
    save(fig, 'label_cooccurrence.png')

    # 7. Vote-share distribution around the 10% cutoff.
    fig, ax = plt.subplots(figsize=(8, 5))
    order = near.sort_values('near_miss_to_positive')
    pos = np.arange(len(order))
    ax.barh(pos, order['near_miss_to_positive'], color=ORANGE, height=.6)
    for i, r in enumerate(order.itertuples()):
        ax.text(r.near_miss_to_positive + .01, i, f'{r.near_miss_to_positive:.2f}',
                va='center', fontsize=8, color=INK)
    ax.set_yticks(pos, order['shelf'])
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Books with share in [5%, 10%)  /  books with share >= 10%')
    ax.set_title('Many negatives sit just under the 10% cutoff')
    save(fig, 'near_threshold.png')

    # 8. Literal shelf name in description, positives vs negatives.
    fig, ax = plt.subplots(figsize=(8, 5))
    order = kw.sort_values('positive_rate')
    pos = np.arange(len(order))
    ax.barh(pos + .2, order['positive_rate'] * 100, height=.38, color=BLUE,
            label='positive books')
    ax.barh(pos - .2, order['negative_rate'] * 100, height=.38, color=ORANGE,
            label='negative books')
    ax.set_yticks(pos, order['shelf'])
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Descriptions containing the shelf name (%)')
    ax.set_title('Shelf names rarely appear in the description')
    ax.legend(frameon=False, loc='lower right')
    save(fig, 'keyword_in_description.png')

    # 9. Ratings tail (all raw books).
    r = df['ratings_count'].dropna()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.hist(np.log10(r[r > 0]), bins=60, color=BLUE)
    ax.set_xlabel('log10(ratings_count), books with at least one rating')
    ax.set_ylabel('Books')
    ax.set_title('Popularity is heavy-tailed')
    save(fig, 'ratings_tail.png')

    # 10. Publication years (valid range only).
    yr = df['year'].dropna()
    yr = yr[(yr >= 1950) & (yr <= 2017)]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.hist(yr, bins=np.arange(1950, 2019), color=BLUE)
    ax.set_xlabel('Publication year (1950-2017 shown)')
    ax.set_ylabel('Books')
    ax.set_title('The file is dominated by 2010s releases')
    save(fig, 'publication_years.png')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, missing, similar = read_raw()
    raw = len(df)

    eligible = df[df.has_description & (df.content_votes >= 10)].copy()
    splits = pd.read_csv(FULL / 'splits.csv.gz', dtype={'book_id': str})
    eligible = eligible.merge(splits[['book_id', 'work_id', 'split']], on='book_id', how='left')
    assert eligible['split'].notna().all(), 'eligible rows must all have a split'
    shares = eligible[[f'share::{s}' for s in LABELS]].to_numpy()
    y = (shares >= .1).astype(np.uint8)
    kwm = eligible[[f'kw::{s}' for s in LABELS]].to_numpy()

    flow = {
        'Raw books': raw,
        'With a description': int(df.has_description.sum()),
        '>= 10 content votes (eligible)': len(eligible),
        **{f'{s.capitalize()} split': int((eligible.split == s).sum())
           for s in SPLIT_ORDER},
    }

    support = pd.DataFrame({'shelf': LABELS,
                            'eligible_positives': y.sum(0)})
    for s in SPLIT_ORDER:
        m = (eligible.split == s).to_numpy()
        support[f'{s}_positives'] = y[m].sum(0)
        support[f'{s}_prevalence'] = y[m].mean(0)

    tr = (eligible.split == 'train').to_numpy()
    yt = y[tr].astype(float)
    joint = yt.T @ yt
    co = pd.DataFrame(joint / np.maximum(joint.diagonal()[:, None], 1),
                      index=LABELS, columns=LABELS)

    near = pd.DataFrame({
        'shelf': LABELS,
        'share_5_to_10pct': ((shares >= .05) & (shares < .1)).sum(0),
        'share_at_least_10pct': y.sum(0),
        'median_share_of_positives': [np.median(shares[y[:, j] == 1, j])
                                      for j in range(len(LABELS))],
    })
    near['near_miss_to_positive'] = near.share_5_to_10pct / near.share_at_least_10pct

    kw = pd.DataFrame({
        'shelf': LABELS,
        'positive_rate': [kwm[y[:, j] == 1, j].mean() for j in range(len(LABELS))],
        'negative_rate': [kwm[y[:, j] == 0, j].mean() for j in range(len(LABELS))],
    })
    kw['precision_if_used_as_rule'] = [
        y[kwm[:, j], j].mean() if kwm[:, j].any() else np.nan
        for j in range(len(LABELS))]

    length_by_label = pd.DataFrame({
        'shelf': ['(no label)'] + LABELS,
        'median_words': [eligible.words[y.sum(1) == 0].median()] +
                        [eligible.words[y[:, j] == 1].median() for j in range(len(LABELS))],
    })

    dup = eligible.description_hash.duplicated(keep=False)
    hash_splits = eligible.groupby('description_hash').split.nunique()
    cross = eligible.description_hash.map(hash_splits) > 1
    hash_works = eligible.groupby('description_hash').work_id.nunique()
    multi_work = eligible.description_hash.map(hash_works) > 1

    yr = df.year
    rc = df.ratings_count
    lang = eligible.language.value_counts()
    eng = eligible.language.isin(['eng', 'en-US', 'en-GB', 'en-CA'])
    stats = {
        'dataset_file': DATA.relative_to(ROOT).as_posix(),
        'flow': flow,
        'dropped_no_description': raw - int(df.has_description.sum()),
        'dropped_fewer_than_10_votes': int(df.has_description.sum()) - len(eligible),
        'labels': LABELS,
        'zero_label_fraction': float((y.sum(1) == 0).mean()),
        'mean_labels_per_book': float(y.sum(1).mean()),
        'labels_per_book': {str(k): int(v) for k, v in
                            pd.Series(y.sum(1)).value_counts().sort_index().items()},
        'label_level_negative_rate': float(1 - y.mean()),
        'content_votes': {'median': float(eligible.content_votes.median()),
                          'p10': float(eligible.content_votes.quantile(.1)),
                          'p90': float(eligible.content_votes.quantile(.9)),
                          'below_30': float((eligible.content_votes < 30).mean())},
        'description_words': {q: float(eligible.words.quantile(p)) for q, p in
                              [('p10', .1), ('median', .5), ('p90', .9), ('p99', .99)]}
                             | {'max': int(eligible.words.max()),
                                'under_20_words': int((eligible.words < 20).sum())},
        'descriptions_with_html_tags': int(eligible.has_html.sum()),
        'duplicate_descriptions': {
            'books_sharing_exact_text': int(dup.sum()),
            'distinct_shared_texts': int(eligible.description_hash[dup].nunique()),
            'books_whose_text_spans_several_works': int(multi_work.sum()),
            'books_whose_text_crosses_splits': int(cross.sum()),
        },
        'language_eligible': {
            'top': {k: int(v) for k, v in lang.head(8).items()},
            'english_codes_share': float(eng.mean()),
            'empty_share': float((eligible.language == '(empty)').mean()),
        },
        'top_shelf_is_one_of_15': float(eligible.top_shelf.isin(LABELS).mean()),
        'top_shelves_overall': {k: int(v) for k, v in
                                eligible.top_shelf.value_counts().head(10).items()},
        'raw_missing_or_empty': {k: int(missing[k]) for k in METADATA_FIELDS},
        'publication_year': {
            'missing': int(yr.isna().sum()),
            'equal_to_1': int((yr == 1).sum()),
            'equal_to_65535': int((yr == 65535).sum()),
            'before_1800': int((yr < 1800).sum()),
            'after_2017': int((yr > 2017).sum()),
            'median_valid': float(yr[(yr >= 1800) & (yr <= 2017)].median()),
        },
        'ratings_count': {
            'zero': int((rc == 0).sum()),
            'median': float(rc.median()), 'mean': float(rc.mean()),
            'p99': float(rc.quantile(.99)), 'max': float(rc.max()),
        },
        'average_rating_zero': int((df.average_rating == 0).sum()),
        'similar_books': similar | {'inside_share': similar['inside_file'] / similar['links']},
        'not_model_inputs': ['publication_year', 'ratings_count', 'average_rating',
                             'num_pages', 'language_code', 'similar_books',
                             'publisher', 'format'],
    }
    (OUT / 'eda_stats.json').write_text(json.dumps(stats, indent=2))
    support.to_csv(OUT / 'label_support_by_split.csv', index=False)
    co.round(4).to_csv(OUT / 'label_cooccurrence_train.csv')
    near.to_csv(OUT / 'near_threshold.csv', index=False)
    kw.to_csv(OUT / 'keyword_in_description.csv', index=False)
    length_by_label.to_csv(OUT / 'description_words_by_label.csv', index=False)
    figures(df, eligible, y, flow, support, co, near, kw)
    print(json.dumps({k: stats[k] for k in
                      ['flow', 'zero_label_fraction', 'duplicate_descriptions']}, indent=2))


if __name__ == '__main__':
    main()

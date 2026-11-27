"""Exploratory counts and plots for the Goodreads romance metadata.

Reads data/goodreads_books_romance.json.gz. Does not train a model.
Label names and the 10% / 10-vote rule match the modeling pipeline.
"""
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'goodreads_books_romance.json.gz'
SPLITS = ROOT / 'reports' / 'full' / 'splits.csv.gz'
OUT = ROOT / 'reports' / 'eda'
LABELS = [
    'm-m', 'historical-romance', 'paranormal', 'historical',
    'contemporary-romance', 'paranormal-romance', 'contemporary', 'chick-lit',
    'new-adult', 'young-adult', 'erotica', 'bdsm', 'manga',
    'romantic-suspense', 'fantasy',
]
PAIRS = [
    ('historical', 'historical-romance'),
    ('contemporary', 'contemporary-romance'),
    ('paranormal', 'paranormal-romance'),
]
EXCLUDE = {
    'to-read', 'currently-reading', 'read', 'favorites', 'favorite', 'favourites',
    'owned', 'own', 'my-books', 'books-i-own', 'default', 'library', 'wish-list',
    'wishlist', 'to-buy', 'ebook', 'ebooks', 'e-book', 'kindle', 'audiobook',
    'audiobooks', 'dnf', 'abandoned', 'did-not-finish', 're-read', 'romance', 'fiction',
}


def to_number(value):
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def quantiles(values, ps):
    arr = np.asarray(values, dtype=float)
    return {str(p): float(np.quantile(arr, p)) for p in ps}


def content_shelves(shelves):
    return {s['name']: int(s['count']) for s in shelves if s['name'] not in EXCLUDE}


def load_splits():
    train = set()
    with gzip.open(SPLITS, 'rt', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['split'] == 'train':
                train.add(row['book_id'])
    return train


def main():
    train_ids = load_splits()
    ids = set()
    raw = nonempty = eligible = 0
    ratings, years_ok = [], []
    zero_ratings = missing_ratings = 0
    missing_year = year_lt_1800 = year_gt_2017 = year_one = year_65535 = 0
    rating_zero_average = missing_average = 0
    languages = Counter()
    desc_words_text, desc_words_eligible = [], []
    label_books = Counter()
    n_pos = Counter()
    pair_counts = {pair: Counter() for pair in PAIRS}
    zero_label = 0

    with gzip.open(DATA, 'rt', encoding='utf-8') as f:
        for line in f:
            book = json.loads(line)
            raw += 1
            book_id = str(book['book_id'])
            ids.add(book_id)
            ratings_count = to_number(book.get('ratings_count'))
            if ratings_count is None:
                missing_ratings += 1
            else:
                ratings.append(ratings_count)
                if ratings_count == 0:
                    zero_ratings += 1
            average = to_number(book.get('average_rating'))
            if average is None:
                missing_average += 1
            elif average == 0:
                rating_zero_average += 1
            year = to_number(book.get('publication_year'))
            if year is None:
                missing_year += 1
            else:
                year = int(year)
                if year == 1:
                    year_one += 1
                if year == 65535:
                    year_65535 += 1
                if year < 1800:
                    year_lt_1800 += 1
                elif year > 2017:
                    year_gt_2017 += 1
                else:
                    years_ok.append(year)
            lang = book.get('language_code') or ''
            languages[lang or '(empty)'] += 1
            text = (book.get('description') or '').strip()
            if not text:
                continue
            nonempty += 1
            words = len(text.split())
            desc_words_text.append(words)
            shelves = content_shelves(book.get('popular_shelves') or [])
            votes = sum(shelves.values())
            if votes < 10:
                continue
            eligible += 1
            desc_words_eligible.append(words)
            positive = [name for name in LABELS if shelves.get(name, 0) / votes >= 0.10]
            n_pos[len(positive)] += 1
            if not positive:
                zero_label += 1
            for name in positive:
                label_books[name] += 1
            if book_id in train_ids:
                owned = set(positive)
                for pair in PAIRS:
                    a, b = pair
                    key = (a in owned, b in owned)
                    pair_counts[pair][str(key)] += 1

    link_total = link_inside = 0
    with gzip.open(DATA, 'rt', encoding='utf-8') as f:
        for line in f:
            book = json.loads(line)
            for other in book.get('similar_books') or []:
                link_total += 1
                if str(other) in ids:
                    link_inside += 1

    ratings_arr = np.asarray(ratings, dtype=float)
    stats = {
        'source_file': str(DATA.relative_to(ROOT)),
        'raw_books': raw,
        'nonempty_descriptions': nonempty,
        'eligible_books': eligible,
        'dropped_no_description': raw - nonempty,
        'dropped_fewer_than_10_content_votes': nonempty - eligible,
        'zero_label_books': zero_label,
        'zero_label_fraction': zero_label / eligible,
        'mean_labels_per_eligible_book': float(np.average(
            np.repeat(list(n_pos.keys()), list(n_pos.values()))
        )),
        'labels_per_book': {str(k): n_pos[k] for k in sorted(n_pos)},
        'label_support_eligible': {name: label_books[name] for name in LABELS},
        'description_words_with_text': {
            'mean': float(np.mean(desc_words_text)),
            'median': float(np.median(desc_words_text)),
            'p90': float(np.quantile(desc_words_text, 0.9)),
            'max': int(np.max(desc_words_text)),
        },
        'description_words_eligible': {
            'mean': float(np.mean(desc_words_eligible)),
            'median': float(np.median(desc_words_eligible)),
            'p90': float(np.quantile(desc_words_eligible, 0.9)),
            'max': int(np.max(desc_words_eligible)),
        },
        'ratings_count': {
            'missing': missing_ratings,
            'zero': zero_ratings,
            'zero_share': zero_ratings / raw,
            'mean': float(ratings_arr.mean()),
            'quantiles': quantiles(ratings_arr, [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]),
            'max': float(ratings_arr.max()),
        },
        'average_rating_zero': rating_zero_average,
        'average_rating_missing': missing_average,
        'publication_year': {
            'missing': missing_year,
            'equal_to_1': year_one,
            'equal_to_65535': year_65535,
            'below_1800': year_lt_1800,
            'after_2017': year_gt_2017,
            'in_1800_to_2017': len(years_ok),
        },
        'language_top': languages.most_common(8),
        'similar_books': {
            'links': link_total,
            'inside_file': link_inside,
            'inside_share': link_inside / link_total,
        },
        'train_alias_pairs': {
            f'{a}__{b}': dict(pair_counts[(a, b)]) for a, b in PAIRS
        },
        'metadata_not_used_by_text_model': [
            'publication_year', 'ratings_count', 'average_rating', 'language_code', 'similar_books'
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'eda_stats.json').write_text(json.dumps(stats, indent=2))
    save_plots(stats, desc_words_eligible, n_pos, label_books, ratings_arr, years_ok)
    print(json.dumps({
        'raw_books': raw,
        'nonempty_descriptions': nonempty,
        'eligible_books': eligible,
        'zero_label_fraction': round(stats['zero_label_fraction'], 4),
        'ratings_zero_share': round(stats['ratings_count']['zero_share'], 4),
        'ratings_median': stats['ratings_count']['quantiles']['0.5'],
        'similar_inside_share': round(stats['similar_books']['inside_share'], 4),
        'year_65535': year_65535,
        'year_one': year_one,
    }, indent=2))


def save_plots(stats, desc_words, n_pos, label_books, ratings, years):
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(desc_words, bins=40, color='#3d5a80', range=(0, 400))
    ax.set_xlabel('Words in the description')
    ax.set_ylabel('Eligible books')
    ax.set_title('Description length, books kept for modeling')
    fig.tight_layout()
    fig.savefig(OUT / 'description_lengths.png', dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    names = list(reversed(LABELS))
    counts = [label_books[name] for name in names]
    ax.barh(names, counts, color='#3d5a80')
    ax.set_xlabel('Eligible books with the shelf as a positive label')
    ax.set_title('Support of the 15 modeled shelves')
    fig.tight_layout()
    fig.savefig(OUT / 'label_support.png', dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    keys = sorted(n_pos)
    ax.bar([str(k) for k in keys], [n_pos[k] for k in keys], color='#ee6c4d')
    ax.set_xlabel('Number of positive shelves among the 15')
    ax.set_ylabel('Eligible books')
    ax.set_title('Most kept books have none of the 15 shelves')
    fig.tight_layout()
    fig.savefig(OUT / 'labels_per_book.png', dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    positive = ratings[ratings > 0]
    ax.hist(np.log10(positive), bins=40, color='#3d5a80')
    ax.set_xlabel('log10 of ratings_count, books with at least one rating')
    ax.set_ylabel('Books')
    ax.set_title('Rating counts have a long right tail')
    fig.tight_layout()
    fig.savefig(OUT / 'ratings_tail.png', dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(years, bins=range(1950, 2019), color='#98c1d9')
    ax.set_xlabel('Publication year')
    ax.set_ylabel('Books')
    ax.set_title('Publication years from 1950 through 2017')
    fig.tight_layout()
    fig.savefig(OUT / 'publication_years.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()

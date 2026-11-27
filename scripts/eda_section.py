"""Section 2 (data, collection, cleaning and EDA) of workbook.ipynb.

Every code cell here is executed while the workbook is built, and its real
printed output is stored in the notebook. The cells read only committed files
in reports/ (written by ``python -m scripts.eda_romance``), so the section can be
rebuilt without the raw data.
"""
import contextlib
import io
import os
from pathlib import Path

from nbformat.v4 import new_markdown_cell

ROOT = Path(__file__).resolve().parents[1]

SETUP = """import json
import pandas as pd
from IPython.display import Image, display
from scipy.stats import spearmanr

pd.set_option('display.width', 120)
EDA = 'reports/eda/'
stats = json.load(open(EDA + 'eda_stats.json'))
for step, n in stats['flow'].items():
    print(f'{step:32s}{n:>9,}')
print(f"{'dropped: empty description':32s}{stats['dropped_no_description']:>9,}")
print(f"{'dropped: < 10 content votes':32s}{stats['dropped_fewer_than_10_votes']:>9,}")"""

TEXT = """d = stats['description_words']
print('description words  p10 {p10:.0f}  median {median:.0f}  p90 {p90:.0f}  p99 {p99:.0f}  max {max:,}'.format(**d))
print('descriptions under 20 words:', f"{d['under_20_words']:,}")
print('descriptions with HTML tags:', stats['descriptions_with_html_tags'])
lang = stats['language_eligible']
print(f"English language codes: {lang['english_codes_share']:.1%}   empty code: {lang['empty_share']:.1%}")
print('top codes:', ', '.join(f'{k} {v:,}' for k, v in lang['top'].items()))
dup = stats['duplicate_descriptions']
print(f"books sharing an exact description: {dup['books_sharing_exact_text']:,}"
      f" ({dup['distinct_shared_texts']:,} distinct texts)")
print(f"  ...shared across different works: {dup['books_whose_text_spans_several_works']:,}")
print(f"  ...shared across splits:          {dup['books_whose_text_crosses_splits']:,}")
words = pd.read_csv(EDA + 'description_words_by_label.csv')
print(words.to_string(index=False))"""

TARGETS = """v = stats['content_votes']
print(f"content votes per book: p10 {v['p10']:.0f}, median {v['median']:.0f}, p90 {v['p90']:.0f}")
print('labels per book:', stats['labels_per_book'])
print(f"books with none of the 15: {stats['zero_label_fraction']:.1%}")
print(f"book-shelf cells that are negative: {stats['label_level_negative_rate']:.1%}")
print(f"most-voted content shelf is one of the 15: {stats['top_shelf_is_one_of_15']:.1%}")
support = pd.read_csv(EDA + 'label_support_by_split.csv')
cols = ['shelf', 'eligible_positives', 'train_prevalence', 'validation_prevalence', 'test_prevalence']
print(support[cols].to_string(index=False, formatters={
    c: '{:.2%}'.format for c in cols if c.endswith('prevalence')}))"""

COOC = """co = pd.read_csv(EDA + 'label_cooccurrence_train.csv', index_col=0)
pairs = [(a, b, co.loc[a, b]) for a in co.index for b in co.columns if a != b]
top = sorted(pairs, key=lambda p: -p[2])[:8]
print('P(column | row), training books, largest off-diagonal values')
for a, b, p in top:
    print(f'  P({b} | {a}) = {p:.2f}')"""

NOISE = """near = pd.read_csv(EDA + 'near_threshold.csv')
f1 = pd.read_csv('reports/full/per_label_wordchar_lr_C1.csv')[['shelf', 'f1', 'auc', 'test_positives']]
kw = pd.read_csv(EDA + 'keyword_in_description.csv')
table = near.merge(kw, on='shelf').merge(f1, on='shelf').sort_values('near_miss_to_positive')
print(table[['shelf', 'share_5_to_10pct', 'share_at_least_10pct', 'near_miss_to_positive',
             'positive_rate', 'f1']].to_string(index=False, float_format='{:.3f}'.format))
rho, p = spearmanr(table.near_miss_to_positive, table.f1)
print(f'\\nSpearman, near-miss ratio vs test F1:      rho = {rho:+.2f} (p = {p:.1e}, 15 shelves)')
rho, p = spearmanr(table.test_positives, table.f1)
print(f'Spearman, test positives vs test F1:        rho = {rho:+.2f} (p = {p:.2f})')
rho, p = spearmanr(table.positive_rate, table.f1)
print(f'Spearman, shelf name in positive text vs F1: rho = {rho:+.2f} (p = {p:.2f})')"""

META = """m = stats['raw_missing_or_empty']
print('missing or empty, of', f"{stats['flow']['Raw books']:,}", 'raw books')
for k, n in m.items():
    print(f'  {k:18s}{n:>9,}  {n / stats["flow"]["Raw books"]:6.1%}')
y, r = stats['publication_year'], stats['ratings_count']
print('publication_year: median', int(y['median_valid']), '| year 1:', y['equal_to_1'],
      '| year 65535:', y['equal_to_65535'], '| before 1800:', y['before_1800'],
      '| after 2017:', y['after_2017'])
print(f"ratings_count: zero {r['zero']:,}, median {r['median']:.0f}, mean {r['mean']:.0f},"
      f" p99 {r['p99']:,.0f}, max {r['max']:,.0f}")
print(f"similar_books links pointing inside this file: {stats['similar_books']['inside_share']:.1%}")"""


def run(source, images=()):
    """Execute a cell in-process, capture its stdout, then attach saved figures."""
    from scripts.build_workbook import code  # local import avoids a cycle
    buf = io.StringIO()
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        with contextlib.redirect_stdout(buf):
            exec(source, run.namespace)
    finally:
        os.chdir(cwd)
    if images:
        source += '\n' + '\n'.join(f"display(Image(EDA + '{Path(p).name}'))"
                                   for p in images)
    return code(source, buf.getvalue(), images)


run.namespace = {}


def fig(*names):
    return [f'reports/eda/{n}.png' for n in names]


def cells():
    run.namespace = {}
    return [
        new_markdown_cell("""## 2. Data, collection, and cleaning

**Source.** The file is the romance subset of the UCSD Goodreads crawl (Wan and McAuley, 2018; Wan et al., 2019), collected in late 2017 from public Goodreads shelves. The authors ask that it be used for academic work only and not redistributed, so `data/` stays out of Git. The romance file is book metadata only: no users, no reviews.

**What the task uses.** Three fields: `description` is the only model input, `popular_shelves` builds the targets, and `work_id` groups editions for the split. Everything else (year, ratings, language, pages, publisher, `similar_books`) is described below but never reaches the classifier.

**Cleaning, in order.** `scripts/eda_romance.py` re-reads the raw file with the same rules as `modeling/core.py`, and its counts match `reports/full/data_audit.json` exactly.

1. Drop books whose description is empty after stripping whitespace.
2. Remove 27 shelves that describe reading status, format, or the whole file (`to-read`, `owned`, `kindle`, `romance`, `fiction`, ...). What remains are *content votes*.
3. Drop books with fewer than 10 content votes, because a 10% share of a handful of votes is not a label.
4. A shelf is positive when it holds at least 10% of a book's content votes. The 15 shelves are the most frequent positives in the training split only.
5. Split by `work_id` into train / validation / test so editions of one book never straddle splits.

The text itself is not rewritten: no stemming, HTML stripping, or language filtering. The TF-IDF vectorizer lowercases and tokenizes it."""),
        run(SETUP, fig('sample_flow')),
        new_markdown_cell("""### 2.1 The descriptions

A description is short publisher copy: median 147 words, 90% under 241. `manga` volumes are the exception, with a median of 69 words, because their blurbs are often a sentence or two. 2,378 eligible descriptions have under 20 words, which gives the model very little to work with.

The text is not all English. Only 61% of eligible books carry an English language code; 28% have no code at all, and Italian, Spanish and Indonesian editions each run to thousands of books. The model sees those descriptions as they are, so a shelf that is common in one language can partly be learned as "this language".

Editions repeat text. 99,319 eligible books share an exact description with another book, but only 1,419 of them share it across *different* works. Almost all duplication is editions of the same work, which is exactly what the `work_id` split keeps on one side. 1,013 books still share text across splits; the modeling diagnostic removes the 279 of those in test and macro-F1 moves from 0.4843 to 0.4842."""),
        run(TEXT, fig('description_lengths')),
        new_markdown_cell("""### 2.2 The targets

Each label is built from reader votes, and the vote counts vary a lot: the median book has 137 content votes, but 10% have fewer than 30, where three readers can decide a 10% share.

The labels are sparse. 65.6% of eligible books have none of the 15 shelves, and 97.1% of all book-shelf cells are negative, so predicting "no" everywhere is 97.1% accurate and has F1 zero. That is why the evaluation uses F1, AUC and AP rather than accuracy. Prevalence runs from 6.8% (`m-m`) down to 1.1% (`fantasy`) and is close across the three splits, so the split itself does not shift the label mix. For about half the books (51.9%), the single most-voted content shelf is one of the 15; the rest are dominated by shelves like `series` or less common genres."""),
        run(TARGETS, fig('content_votes', 'label_support', 'labels_per_book')),
        new_markdown_cell("""### 2.3 Overlapping shelves

The 15 labels are not 15 separate genres. Readers use a generic and a "-romance" spelling side by side: 86% of training books tagged `historical` are also tagged `historical-romance`, and 79% of `paranormal-romance` books are also `paranormal`. The contemporary pair overlaps less (about 0.43 each way), and `contemporary` also absorbs 37% of `young-adult`. `erotica`, `bdsm` and `m-m` form a looser second cluster: a quarter of `erotica` books are also `bdsm`, and a quarter of `bdsm` books are also `m-m`. This is why the modeling section also reports a separate 12-label run that merges the three alias pairs."""),
        run(COOC, fig('label_cooccurrence')),
        new_markdown_cell("""### 2.4 How noisy is each label?

The 10% rule draws a hard line through a continuous vote share. For each shelf, the chart below divides the number of books just *under* the line (5-10% of votes) by the number at or above it. For `manga` the ratio is 0.02: readers either call a book manga or they don't. For `contemporary` it is 5.3, so for every positive there are five near-identical books labeled negative. `contemporary-romance` (3.7), `fantasy` (3.4) and `erotica` (2.8) follow.

Those are the shelves the model does worst on. Across the 15 shelves, this ratio has a Spearman correlation of -0.84 with the selected model's test F1, while the number of test positives barely relates to F1 (rho +0.26). The weak shelves are not weak because they are small. Their labels are the least decisive, and the evaluation's error review found the same thing book by book.

The second chart explains the keyword baseline. A shelf name rarely appears in the description of its own positives: 36% for `bdsm`, around 10% for `m-m`, `paranormal` and `new-adult`, and 1-3% for `historical-romance`, `contemporary`, `chick-lit` and `young-adult`. A literal match cannot do much, and its test macro-F1 of 0.106 shows it. A learned model has to rely on indirect cues: pronouns, settings, names."""),
        run(NOISE, fig('near_threshold', 'keyword_in_description')),
        new_markdown_cell("""### 2.5 Metadata the model does not use

These fields describe the collection, not the task. Publication year is missing for 30% of raw books, and the remaining values include placeholder years: one book in year 1, two in year 65535, 83 before 1800 and 255 after the 2017 crawl. The plot shows 1950-2017 only, and its median is 2013. Rating counts are heavy-tailed: median 52, mean 448, maximum about 2.08 million, and 1,499 books with no ratings at all. 84% of `similar_books` links point to another book in this file, so a link-prediction task would have been possible; this project does not use it."""),
        run(META, fig('publication_years', 'ratings_tail')),
        new_markdown_cell("""### 2.6 What the data implies for modeling

- **Sparse, multi-label targets.** Use one binary classifier per shelf, class weights, per-shelf thresholds, and F1/AP rather than accuracy.
- **Group the split by work.** Most repeated text is editions of the same work; splitting by `book_id` would leak descriptions into test.
- **Expect a ceiling on noisy shelves.** `contemporary`, `contemporary-romance`, `fantasy` and `erotica` have many books just under the 10% line, so some of their "errors" are label noise rather than model failures.
- **Overlapping labels.** Report the 15-label task as defined, and treat the alias merge as a separate sensitivity check.
- **Text only.** Year, ratings and language are incomplete or noisy and are not used, so the comparison stays a description-only task."""),
    ]

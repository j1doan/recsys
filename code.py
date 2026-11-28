# %% [markdown]
# # Predicting Goodreads Shelves from Book Descriptions
#
# *CSE 258, Assignment 2*
#
# **Abstract.** Goodreads readers assign books to user-defined shelves that act as genre tags, but a newly published book has no shelves, which leaves shelf-based recommendation without input for new items. We study whether the description of a romance novel is sufficient to predict 15 common reader shelves. Using 299,789 books from the UCSD Goodreads dataset, we train one logistic regression per shelf on TF-IDF word and character features and evaluate it on 60,410 held-out books. The model reaches a macro-F1 of 0.454 (95% bootstrap interval 0.443 to 0.464), compared with 0.106 for a keyword baseline. Performance varies widely across shelves, from an F1 of 0.812 for `manga` to 0.244 for `contemporary`. Across shelves, F1 is strongly correlated with a measure of reader disagreement about the label (Spearman $\rho = -0.82$) and only weakly with the number of training examples ($\rho = +0.23$). Removing the most contested negatives from the test set tends to raise precision more for the contested shelves than for other shelves of similar recall, but leaves their low recall unexplained. We conclude that descriptions predict shelves with a distinctive vocabulary well, and that disagreement among readers explains part, but not all, of the failure on the remaining shelves.

# %% [markdown]
# ## I. Introduction
#
# Goodreads users organize books into shelves that they name themselves. Shelves such as `historical-romance`, `paranormal`, and `m-m` function as community-assigned genre tags, and recommendation and search systems can use them to retrieve books of a given kind. Shelves accumulate only after readers have engaged with a book, so a newly published book has none. This is an instance of the item cold-start problem.
#
# A new book does have a description, usually supplied by the publisher. This report addresses the following question: given only its description, can a model predict which shelves readers will assign to a book? We study the question with a TF-IDF representation and a linear classifier, a standard approach to text classification [3].
#
# Sections II to VI follow the five parts of the assignment. Section II defines the predictive task and the evaluation protocol. Section III describes the dataset, its exploratory analysis, and the construction of labels. Section IV presents the model. Section V reports the evaluation. Section VI discusses related work, and Section VII concludes.

# %% [markdown]
# ## II. Predictive Task

# %% [markdown]
# ### A. Problem Formulation
#
# Each example is one book. The input is the text of its description, and the output is a vector of 15 binary labels, one for each of the 15 most frequent genre shelves. A book may carry several of these shelves or none, so the task is multi-label classification.
#
# The labels are derived from reader behavior. For every book, the dataset lists its most popular shelves together with the number of times readers placed the book on each. A shelf is considered a true label for a book when it receives at least 10% of the book's genre-related shelf votes (Section III-C). This definition assumes that a shelf chosen by a tenth of a book's readers reflects a shared judgment about the book, whereas a shelf chosen by one or two readers does not.

# %% [markdown]
# ### B. Cold-Start Approximation
#
# The dataset is a single snapshot and does not record books at the time of release. We therefore approximate the cold-start setting by hiding the shelves of books whose shelves are known and predicting them from the description alone. The approximation assumes that the descriptions of established books are representative of those of new books. It also restricts the study to books with at least 10 genre votes, which are more established than a typical new release.

# %% [markdown]
# ### C. Evaluation Protocol
#
# The books were split into training (64%), validation (16%), and test (20%) sets. Editions of the same work usually share a description, so the split was performed by work, and all editions of a book belong to the same set. Without this grouping, the model could be evaluated on descriptions it had already seen during training. The validation set was used only to select a decision threshold for each shelf, and the test set was used once, after all choices had been fixed.
#
# Performance is measured primarily by F1, the harmonic mean of precision (the fraction of predicted shelves that are correct) and recall (the fraction of true shelves that are predicted). Micro-F1 pools all book-shelf decisions and is dominated by frequent shelves; macro-F1 averages the per-shelf F1 and weights every shelf equally. Macro-F1 is the primary metric, since the research question concerns individual shelves, including rare ones. ROC AUC and average precision (AP) are also reported; both measure the quality of the ranking before a threshold is applied. Accuracy is not reported. Because 97.1% of book-shelf pairs are negative, a model that never predicts a shelf would attain 97.1% accuracy while being of no use.

# %% [markdown]
# ### D. Baselines
#
# Three baselines involve no learning. The label-prior baseline predicts a shelf whenever more than half of the training books carry it. No shelf is that frequent, so this baseline predicts nothing and attains an F1 of zero, which is the floor any useful model must exceed. The every-shelf baseline predicts all 15 shelves for every book; it attains a recall of 1, and its F1 on a shelf with positive rate $p$ is $2p/(1+p)$, so it shows the score obtainable without reading the text. The keyword baseline predicts a shelf whenever the shelf name, for example "historical romance", occurs in the description, and tests whether the task can be solved by matching the obvious term. The proposed model is additionally evaluated with the default threshold of probability 0.5, which separates the contribution of the learned weights from that of threshold selection.

# %% [markdown]
# ### E. Validity
#
# The validity of the predictions is assessed in three ways. First, the split guarantees that test books, and all other editions of them, play no part in training or threshold selection. Second, the learned weights are inspected to confirm that the model relies on genre-related terms. Third, because shelf votes are a noisy proxy for genre, the label noise of each shelf is quantified (Section III-D) and related to the model's performance (Section V-C).

# %% [markdown]
# ## III. Data

# %% [markdown]
# ### A. Dataset
#
# We use the romance subset of the UCSD Goodreads dataset [1], [2], which was collected in late 2017 by crawling public Goodreads shelves. Each record describes one book: its title, authors, description, publication details, rating summary, similar books, and most popular shelves with their vote counts. The file contains neither user identifiers nor reviews, so interaction-based models such as latent factor models and Bayesian personalized ranking cannot be applied, and a content-based approach is required. The dataset is distributed for academic use only and is not included in the repository; the notebook reads a local copy whose path is given in `secret.env`.
#
# The code below loads the full file and summarizes each field.

# %% imports
import gzip
import json
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import find_dotenv, load_dotenv
from scipy.sparse import csr_matrix, hstack
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

# %% dotenv
load_dotenv(find_dotenv('secret.env', usecwd=True))
PATH = os.environ['GOODREADS_PATH']

# %% plot style: one neutral color for all shelves, three focus shelves keep the same color in every figure
os.makedirs('figs', exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#666666', 'axes.grid': True, 'grid.color': '#e6e6e6',
                     'axes.axisbelow': True, 'savefig.bbox': 'tight'})
GRAY = '#b3b3b3'
FOCUS = {'manga': '#257de0', 'historical-romance': '#f47230', 'contemporary': '#00b894'}


def color(shelf):
    return FOCUS.get(shelf, GRAY)


def save(n):
    plt.tight_layout()
    plt.savefig(f'figs/fig{n}.png', dpi=200)
    plt.show()

# %% load
# set LIMIT (e.g. 50_000) for a quick pass
LIMIT = None
rows = []
with gzip.open(PATH, 'rt', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if LIMIT and i >= LIMIT:
            break
        rows.append(json.loads(line))
df = pd.DataFrame(rows)
del rows
print(df.shape)

# %% first record (long fields cut to 80 characters)
df.iloc[0].astype(str).str.slice(0, 80)

# %% coverage: share of non-empty values per column
empty = df.map(lambda v: v in ('', None) or (isinstance(v, list) and not v))
(1 - empty.mean()).sort_values().round(3)

# %% numeric columns are strings: convert
for c in ['average_rating', 'ratings_count', 'text_reviews_count', 'num_pages',
          'publication_year', 'publication_month', 'publication_day']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df[['average_rating', 'ratings_count', 'text_reviews_count', 'num_pages', 'publication_year']].describe()

# %% rating distribution
df['average_rating'].round(1).value_counts().sort_index()

# %% popularity: heavy tail
zero_share = (df['ratings_count'] == 0).mean()
print(f'{zero_share} share with 0 ratings')
df['ratings_count'].quantile([.1, .25, .5, .75, .9, .99])

# %% rating vs popularity
df[['average_rating', 'ratings_count', 'text_reviews_count', 'num_pages', 'publication_year']].corr(method='spearman')

# %% publication years
df['publication_year'].value_counts().sort_index().tail(40)

# %% categorical: language, format, ebook
for c in ['language_code', 'format', 'is_ebook', 'country_code']:
    print(df[c].replace('', '(empty)').value_counts().head(10), '\n')

# %% list columns: sizes
for c in ['authors', 'series', 'similar_books', 'popular_shelves']:
    print(c, df[c].map(len).describe()[['mean', '50%', 'max']].round(1).to_dict())

# %% top shelves: total count and number of books per shelf name
tot, nb = Counter(), Counter()
for shelves in df['popular_shelves']:
    for s in shelves:
        tot[s['name']] += int(s['count'])
        nb[s['name']] += 1
pd.DataFrame({'total': tot, 'books': nb}).sort_values('total', ascending=False).head(40)

# %% authors and series: books per author, books per series
a = Counter(x['author_id'] for au in df['authors'] for x in au)
a_stats = pd.Series(a).describe()[['mean', '50%', 'max']].round(1).to_dict()
print(f'{len(a)} authors; books per author: {a_stats}')
s = Counter(x for se in df['series'] for x in se)
s_stats = pd.Series(s).describe()[['mean', '50%', 'max']].round(1).to_dict()
print(f'{len(s)} series; books per series: {s_stats}')

# %% descriptions
d = df['description'].replace('', pd.NA).dropna()
n_words = d.str.split().str.len()
d_words = n_words.describe()[['mean', '50%', 'max']].round(0).to_dict()
print(f'{len(d)} with description; words: {d_words}')
d.iloc[:3].tolist()

# %% similar_books: how many point to books inside this file?
ids = set(df['book_id'])
sim = [x for lst in df['similar_books'] for x in lst]
inside = sum(x in ids for x in sim) / max(len(sim), 1)
print(f'{len(sim)} links; {inside:.3f} point inside the file')

# %% Fig. 1: words per description
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(n_words, bins=np.logspace(0, np.log10(n_words.max()), 60), color=GRAY, edgecolor='white', linewidth=0.5)
ax.axvline(n_words.median(), color='#191919', linewidth=1)
ax.annotate(f'median {n_words.median():.0f}', (n_words.median(), ax.get_ylim()[1] * 0.9), xytext=(4, 0),
            textcoords='offset points')
ax.set(xscale='log', xlabel='words per description (log scale)', ylabel='books')
ax.grid(axis='x', visible=False)
save(1)

# %% [markdown]
# *Fig. 1. Distribution of description length over the books with a description, in words, on a logarithmic axis. The vertical line marks the median.*

# %% [markdown]
# ### B. Exploratory Analysis
#
# The file contains 335,449 books, of which 90.9% have a non-empty description. The median description length is 147 words, and most descriptions are a paragraph or longer (Fig. 1).
#
# The remaining metadata is noisy. Publication year is missing for 30% of books and includes invalid values such as 1, 20015, and 65535, which are most likely entry errors. Rating counts are highly skewed, with a median of 52, a mean of 448, and a maximum of about 2.08 million. Language codes are missing for 31% of books, and Italian, Spanish, and Indonesian editions each number several thousand, so not all descriptions are in English. Since the model reads only the description, these fields require no cleaning; the mixture of languages, however, introduces noise into the text itself.
#
# The shelf summary shows why the raw shelves cannot serve as labels directly. The most frequently used shelves, `to-read`, `currently-reading`, and `favorites`, record a reader's interaction with a book and carry no genre information. Genre shelves such as `historical-romance` and `paranormal` rank lower. Finally, 83.8% of `similar_books` links point to books within the file, which would support a link-prediction task; that task is outside the scope of this report.

# %% [markdown]
# ### C. Label Construction
#
# Labels were constructed from the raw shelf votes in four steps. First, shelves that do not describe content were removed: reading status (`to-read`), format (`kindle`, `ebook`), ownership (`owned`), and the tags `romance` and `fiction`, which nearly every book in the file carries and which therefore cannot distinguish books from one another. These shelves form the set `EXCLUDE`, and the remaining votes are referred to as content votes. Second, only books with at least 10 content votes were retained, since with fewer votes a label could be determined by two or three readers. Third, a shelf was taken as a positive label for a book when it received at least 10% of the book's content votes. Fourth, the 15 shelves most often positive among the training books were selected as the label set; selecting them from the training data alone keeps the test set unseen.
#
# The 10% threshold trades label quality against label quantity. The sweep below shows that, with 10 required votes, a threshold of 5% gives the 15th most frequent shelf 10,511 positive books, 10% gives 3,105, and 20% gives only 376. A low threshold admits shelves chosen by only a few readers, while a high threshold leaves too few positive examples for training. The 10% threshold retains several thousand positive examples for every shelf.

# %% label rule parameters
EXCLUDE = {
    'to-read', 'currently-reading', 'read', 'favorites', 'favorite', 'favourites', 'owned', 'own',
    'my-books', 'books-i-own', 'default', 'library', 'wish-list', 'wishlist', 'to-buy',
    'ebook', 'ebooks', 'e-book', 'kindle', 'audiobook', 'audiobooks', 'dnf', 'abandoned',
    'did-not-finish', 're-read', 'romance', 'fiction',
}
MIN_VOTES = 10
MIN_FRAC = 0.10
K = 15

# %% content shelves per book
def content_shelves(shelves):
    return {s['name']: int(s['count']) for s in shelves if s['name'] not in EXCLUDE}

df['sh'] = df['popular_shelves'].map(content_shelves)
df['sh_votes'] = df['sh'].map(lambda d: sum(d.values()))
has_text = df['description'].str.strip() != ''

# %% cutoff sweep: books kept, and positives of the K-th most common shelf
rows = []
for votes in (5, 10, 20):
    keep = df[has_text & (df['sh_votes'] >= votes)]
    for frac in (0.05, 0.10, 0.20, 0.30):
        c = Counter(n for d, v in zip(keep['sh'], keep['sh_votes']) for n, k in d.items() if k / v >= frac)
        pos = [p for _, p in c.most_common(K)]
        rows.append({'min_votes': votes, 'min_frac': frac, 'books': len(keep),
                     'positives_top1': pos[0], f'positives_top{K}': pos[-1]})
pd.DataFrame(rows)

# %% eligible books
elig = df[has_text & (df['sh_votes'] >= MIN_VOTES)].reset_index(drop=True)
# editions of one work share a description, so they must land in the same split
valid_work = ~elig['work_id'].isin(['', '0'])
elig['group'] = np.where(valid_work, 'work:' + elig['work_id'], 'book:' + elig['book_id'])
print(f'{len(df)} raw, {has_text.sum()} with description, {len(elig)} eligible')

# %% split by work: 64% train, 16% validation (model and cutoff choices), 20% test (scored once)
dev, te = next(GroupShuffleSplit(test_size=0.2, random_state=0).split(elig, groups=elig['group']))
i_tr, i_va = next(GroupShuffleSplit(test_size=0.2, random_state=1).split(dev, groups=elig['group'].iloc[dev]))
tr, va = dev[i_tr], dev[i_va]
assert not set(elig['group'].iloc[tr]) & set(elig['group'].iloc[te])
print(f'{len(tr)} train, {len(va)} validation, {len(te)} test')

# %% labels: the K shelves that pass the rule most often among training books
c = Counter(n for d, v in zip(elig['sh'].iloc[tr], elig['sh_votes'].iloc[tr]) for n, k in d.items() if k / v >= MIN_FRAC)
SHELVES = [n for n, _ in c.most_common(K)]
share = np.array([[d.get(s, 0) / v for s in SHELVES] for d, v in zip(elig['sh'], elig['sh_votes'])])
Y = (share >= MIN_FRAC).astype(np.float32)
print(f'no label: {(Y.sum(1) == 0).mean():.1%} of books; negative book-shelf pairs: {1 - Y.mean():.1%}')
rate = pd.DataFrame({'train': Y[tr].mean(0), 'validation': Y[va].mean(0), 'test': Y[te].mean(0)}, index=SHELVES)
rate.round(3)

# %% Fig. 2: share of books that carry each shelf, by split
fig, ax = plt.subplots(figsize=(6, 4))
y = np.arange(K)
ax.barh(y, 100 * rate['train'], color=[color(s) for s in SHELVES], height=0.7, label='train')
ax.scatter(100 * rate['validation'], y, marker='o', s=30, facecolors='none', edgecolors='#191919', label='validation')
ax.scatter(100 * rate['test'], y, marker='x', s=30, color='#191919', label='test')
ax.set(yticks=y, yticklabels=SHELVES, xlabel='books carrying the shelf (%)')
ax.invert_yaxis()
ax.grid(axis='y', visible=False)
ax.legend(loc='lower right', frameon=False)
save(2)

# %% [markdown]
# *Fig. 2. Percentage of books that carry each shelf, in the training set (bars), validation set (circles), and test set (crosses). Shelves are ordered by frequency in the training set; the three shelves discussed in Section V are colored.*

# %% Fig. 3: shelf co-occurrence, P(column shelf | row shelf), training books
co = Y[tr].T @ Y[tr]
cond = co / np.diag(co)[:, None]
np.fill_diagonal(cond, np.nan)  # diagonal is 1 by definition and would set the color scale
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cond, cmap='Blues', vmin=0)
for i, j in zip(*np.nonzero(np.nan_to_num(cond) >= 0.2)):
    ax.text(j, i, f'{cond[i, j]:.2f}', ha='center', va='center', fontsize=7,
            color='white' if cond[i, j] > 0.5 * np.nanmax(cond) else '#191919')
ax.set(xticks=range(K), yticks=range(K), xlabel='shelf B', ylabel='shelf A')
ax.set_xticklabels(SHELVES, rotation=60, ha='right')
ax.set_yticklabels(SHELVES)
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.8, label='P(shelf B | shelf A)')
save(3)

# %% [markdown]
# *Fig. 3. Co-occurrence of shelves among training books. Each cell gives the fraction of books labeled with the row shelf (A) that also carry the column shelf (B); values of 0.2 or more are printed, and the diagonal is left blank.*

# %% [markdown]
# After filtering, 299,789 books remained: 191,692 for training, 47,687 for validation, and 60,410 for testing. The labels are sparse. Of all books, 65.6% carry none of the 15 shelves; the most frequent shelf, `m-m`, applies to 6.8% of training books and the least frequent, `fantasy`, to 1.1%. Shelf frequencies are nearly identical across the three sets (Fig. 2), so the split does not shift the label distribution.
#
# Several of the 15 shelves are near-synonyms, such as `historical` and `historical-romance`, because readers use both names, and such pairs co-occur on many books (Fig. 3). They are kept as separate labels because they are separate shelves on Goodreads, but this overlap is one reason to expect some labels to be unreliable.

# %% [markdown]
# ### D. Label Noise
#
# A fixed threshold of 10% imposes a hard boundary on what is a continuous vote share. A book that receives 9% of its votes as `contemporary` is labeled negative, and one that receives 11% is labeled positive, although readers treated the two books almost identically. When many books lie just below the boundary, the labels of that shelf are unstable. We quantify this effect for each shelf as the ratio of training books just below the threshold, with 5% to 10% of their votes, to books at or above it. A ratio near zero indicates that readers agree on which books belong to the shelf, and a high ratio indicates that many books divide their readers. The lower edge of 5%, half the threshold, was fixed before any model was trained and was not adjusted afterwards. A different band would change the exact values, so the ratio is used only to rank shelves by how contested their labels are.

# %% label noise: books just under the cutoff (5-10% of votes) per book at or above it
near = ((share[tr] >= 0.05) & (share[tr] < MIN_FRAC)).sum(0) / Y[tr].sum(0)
pd.Series(near, index=SHELVES).sort_values().round(2)

# %% Fig. 4: vote share of the focus shelves near the 10% threshold (training books with a share of at least 1%)
focus = [s for s in FOCUS if s in SHELVES]
fig, axes = plt.subplots(1, len(focus), figsize=(7, 2.6), sharex=True, sharey=True, squeeze=False)
bins = np.arange(0.01, 0.305, 0.01)
for ax, s in zip(axes[0], focus):
    j = SHELVES.index(s)
    counts, _ = np.histogram(share[tr][:, j], bins=bins)
    ax.bar(100 * bins[:-1], counts / Y[tr][:, j].sum(), width=0.9, align='edge', color=FOCUS[s])
    ax.axvspan(5, 10, color='#191919', alpha=0.06, linewidth=0)
    ax.axvline(10, color='#191919', linewidth=1)
    ax.set(title=f'{s}\nnear-cutoff ratio {near[j]:.2f}', xlabel='share of content votes (%)')
    ax.grid(axis='x', visible=False)
axes[0][0].set_ylabel('books per labeled book')
save(4)

# %% [markdown]
# *Fig. 4. Distribution of a shelf's share of a book's content votes among training books, for the three focus shelves. Bars are one percentage point wide and are divided by the number of books labeled with the shelf. The vertical line marks the 10% threshold, and the shaded band marks the 5% to 10% range used in the near-cutoff ratio.*

# %% [markdown]
# The ratios span two orders of magnitude. For `manga` the ratio is 0.03, since readers either shelve a book as manga or do not. For `contemporary` it is 5.51, meaning that for every book labeled `contemporary` about five more lie just below the threshold and are labeled negative. Fig. 4 shows the difference: the `manga` panel is nearly empty because almost no book has a `manga` share between 1% and 30%, whereas for `contemporary` the number of books rises steadily as the share falls below the threshold. `contemporary-romance` (3.61), `fantasy` (3.30), and `erotica` (2.78) follow. We hypothesize that the shelves with high ratios will be the most difficult to predict regardless of the model, because their labels depend in part on where an arbitrary boundary falls.

# %% [markdown]
# ## IV. Model

# %% [markdown]
# ### A. Formulation
#
# The task is formulated as 15 independent binary classification problems over a shared representation of the text. For shelf $j$ and a book with feature vector $x$, the model computes the score
#
# $$s_j(x) = w_j^\top x + b_j \tag{1}$$
#
# and predicts the shelf when $s_j(x)$ exceeds a threshold $t_j$. Treating the shelves independently ignores correlations between them, such as the overlap between `historical` and `historical-romance`, but keeps the model simple to train and to interpret.

# %% [markdown]
# ### B. Evaluation Procedure and Baselines
#
# The code below implements the evaluation and the three baselines, which require no training. The function `best_cutoff` selects, for one shelf, the threshold that maximizes F1 on the validation set by scanning every distinct score. The function `scores` computes the metrics of Section II-C on the test set.

# %% metrics
def best_cutoff(y, s):
    """Score cutoff with the highest F1 on these labels."""
    order = np.argsort(-s, kind='stable')
    tp = np.cumsum(y[order])
    f1 = 2 * tp / (np.arange(1, len(y) + 1) + y.sum())
    return s[order][np.argmax(f1)]


def scores(y, s, cut):
    p = s >= cut
    return {'micro_f1': f1_score(y, p, average='micro', zero_division=0),
            'macro_f1': f1_score(y, p, average='macro', zero_division=0),
            'macro_auc': roc_auc_score(y, s, average='macro'),
            'macro_ap': average_precision_score(y, s, average='macro'),
            'labels_per_book': p.sum(1).mean()}


results = {}

# %% baseline 1: label prior (predict a shelf when most training books have it)
prior = Y[tr].mean(0)
results['label prior'] = scores(Y[te], np.tile(prior, (len(te), 1)), 0.5)

# %% baseline 2: predict every shelf for every book (recall 1, F1 = 2p / (1 + p) per shelf)
results['every shelf'] = scores(Y[te], np.ones((len(te), K)), 0.5)

# %% baseline 3: the shelf name appears in the description
def keyword(texts):
    norm = [' ' + re.sub(r'[^\w]+', ' ', t.lower()) + ' ' for t in texts]
    return np.array([[' ' + s.replace('-', ' ') + ' ' in t for s in SHELVES] for t in norm], dtype=float)

KW_te = keyword(elig['description'].iloc[te])
results['shelf name in text'] = scores(Y[te], KW_te, 0.5)

# %% [markdown]
# ### C. Text Representation
#
# Each description is represented by TF-IDF features. The term frequency counts the occurrences of a term in a description, and the inverse document frequency down-weights terms that occur in many descriptions. Terms that occur in nearly every romance description, such as "love" or "life", therefore contribute little, while distinctive terms such as "duke" or "vampire" contribute more. Two kinds of terms are used. Word unigrams and bigrams (50,000 features, English stop words removed) capture topical vocabulary and multi-word names such as series titles. Character 3- to 5-grams (60,000 features, weighted by 0.5) capture fragments of words, which help with names, inflected forms, and descriptions in other languages. Both vectorizers were fit on the training descriptions only. The vocabulary sizes, the minimum document frequency of 5, and the weight on character features were fixed in our group's earlier experiments and were not tuned here. A description contains many more distinct character sequences than words, and the weight of 0.5 prevents the character features from dominating the word features.
#
# This representation assumes that genre is expressed by which terms occur, independent of their order. The assumption is reasonable for genres with a characteristic vocabulary, but the representation cannot capture meaning that depends on word order or information that a description omits.

# %% features (fit on training books only)
word = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=5, stop_words='english',
                       sublinear_tf=True, dtype=np.float32)
char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=5, max_features=60_000,
                       sublinear_tf=True, dtype=np.float32)
text = elig['description']
word.fit(text.iloc[tr])
char.fit(text.iloc[tr])


def features(idx):
    return hstack([word.transform(text.iloc[idx]), 0.5 * char.transform(text.iloc[idx])]).tocsr()


X_tr, X_va, X_te = features(tr), features(va), features(te)
FEATURES = np.r_[word.get_feature_names_out(), char.get_feature_names_out()]
print(X_tr.shape, f'{X_tr.nnz / X_tr.shape[0]:.0f} nonzero features per book')

# %% [markdown]
# ### D. Classifier
#
# One logistic regression per shelf was trained, implemented in PyTorch as a single linear layer with 15 outputs so that it runs on a GPU. The loss is the binary cross-entropy over all shelves plus an $L_2$ penalty on the weights; it equals the objective of scikit-learn's `LogisticRegression(C=1, class_weight='balanced')` divided by the number of training books. Balanced class weights scale the loss of positive examples in proportion to their rarity. Without them, the model could attain a low loss by predicting no shelves, since 97% of the targets are negative. Optimization used Adam with a learning rate of 0.01 on mini-batches of 4,096 books for 8 epochs. These settings were chosen for speed and memory and were not tuned. Each book uses about 1,200 of the 110,000 features, so the feature matrix is stored in sparse form, which fits in the 8 GB memory of a laptop GPU.

# %% [markdown]
# ### E. Model Selection
#
# Logistic regression on TF-IDF features was chosen because it is an established baseline for text classification [3], it is part of the course material, it trains in minutes on 190,000 documents, and its weights can be inspected directly, which Section V-D uses to check the evidence the model relies on. In earlier experiments, our group compared eight scikit-learn configurations by validation macro-F1. The configurations varied the features (words only, or words and characters), the regularization strength ($C \in \{0.25, 1, 4\}$), class weighting, and the classifier (logistic regression or a linear SVM). Logistic regression with word and character features and $C = 1$ performed best and is the configuration used here. Models that use word order, such as fine-tuned transformers, may capture more of a description's meaning, but they are considerably more expensive to train on this corpus and were not evaluated.
#
# The alternatives differ mainly in cost and in what they demand of the implementation. A linear SVM costs the same as logistic regression; its scores are not probabilities, which matters little here because a threshold is selected for each shelf in any case. A nearest-neighbor classifier over TF-IDF vectors needs no training, but it compares each new book with all 191,692 training books, so the cost of prediction grows with the training set. Methods that model dependence between shelves, such as classifier chains, could use the overlap shown in Fig. 3, but they make each shelf's prediction depend on the others, and we kept the shelves independent so that each can be analyzed on its own. Fine-tuned transformers read word order, but they require far more computation and GPU memory than a linear model, and their predictions cannot be traced to individual terms as in Section V-D. The implementation of the chosen model involved a further trade-off. scikit-learn's solver fits the objective to convergence but runs on the CPU, one shelf at a time. The PyTorch version trains all 15 shelves at once on the GPU through sparse matrix products, at the cost of a learning rate and a number of epochs that must be chosen, and of an optimization that had not converged when training stopped (Section IV-F).

# %% logistic regression on the GPU
# Same objective as sklearn's LogisticRegression(C=1, class_weight='balanced'), divided by n:
# mean weighted log-loss + ||W||^2 / (2 C n). Mini-batches keep memory small on an 8 GB card.
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.sparse.check_sparse_tensor_invariants.disable()  # explicit opt-out; checks cost time, our indices come from scipy


def to_torch(x):
    x = x.tocoo()
    i = torch.from_numpy(np.vstack([x.row, x.col]).astype(np.int64))
    return torch.sparse_coo_tensor(i, torch.from_numpy(x.data), x.shape).to(DEV)


def fit_lr(x, y, x_va, y_va, C=1.0, epochs=8, batch=4096, lr=0.01, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(y)
    yt = torch.from_numpy(y).to(DEV)
    pos = yt.sum(0)
    w_pos, w_neg = n / (2 * pos), n / (2 * (n - pos))  # class_weight='balanced'
    W = torch.zeros(x.shape[1], y.shape[1], device=DEV, requires_grad=True)
    b = torch.zeros(y.shape[1], device=DEV, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    yv = torch.from_numpy(y_va)
    hist = []
    for epoch in range(epochs):
        total = 0.0
        for idx in np.array_split(rng.permutation(n), max(n // batch, 1)):
            yb = yt[idx]
            logits = torch.sparse.mm(to_torch(x[idx]), W) + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yb, weight=yb * w_pos + (1 - yb) * w_neg)
            loss = loss + (W ** 2).sum() / (2 * C * n)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        # validation: same class weights, no penalty
        val = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(decision(x_va, W.detach(), b.detach())), yv,
            weight=yv * w_pos.cpu() + (1 - yv) * w_neg.cpu()).item()
        hist.append((total / n, val))
        print(f'epoch {epoch + 1}: train objective {total / n:.4f}, validation log-loss {val:.4f}')
    return W.detach(), b.detach(), hist


@torch.no_grad()
def decision(x, W, b, batch=8192):
    out = [(torch.sparse.mm(to_torch(x[i:i + batch]), W) + b).cpu().numpy() for i in range(0, x.shape[0], batch)]
    return np.vstack(out)


W, b, hist = fit_lr(X_tr, Y[tr], X_va, Y[va])
S_va, S_te = decision(X_va, W, b), decision(X_te, W, b)

# %% Fig. 5: loss per epoch
h = np.array(hist)
ep = np.arange(1, len(h) + 1)
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(ep, h[:, 0], color='#191919', linewidth=2, marker='o', markersize=4,
        label='training objective (mean over epoch)')
ax.plot(ep, h[:, 1], color='#666666', linewidth=2, linestyle='--', marker='o', markersize=4,
        label='validation log-loss (end of epoch)')
ax.set(xlabel='epoch', ylabel='weighted log-loss', xticks=ep)
ax.legend(frameon=False)
save(5)

# %% [markdown]
# *Fig. 5. Training objective, averaged over the mini-batches of each epoch and including the $L_2$ penalty, and weighted log-loss on the validation set at the end of each epoch.*

# %% [markdown]
# ### F. Convergence
#
# Both the training objective and the validation loss were still decreasing at the eighth epoch (Fig. 5); the validation loss fell from 0.3608 to 0.3574 over the last epoch, so the model had not fully converged. The two curves are not directly comparable: the training value is a mean over the epoch that includes the $L_2$ penalty, while the validation value is measured at the end of the epoch without it, so their crossing near the fourth epoch does not indicate overfitting. In our group's earlier experiments, scikit-learn's solver fit the same objective to convergence and reached a test macro-F1 of 0.484, compared with 0.454 here. The results in Section V are therefore a slight underestimate of what this model can achieve, and the untuned optimizer settings are the most likely source of the difference.

# %% [markdown]
# ## V. Evaluation

# %% [markdown]
# ### A. Overall Results
#
# The threshold of each shelf was selected on the validation set, and every model was evaluated once on the test set. To measure the contribution of the character features, the same model was also trained on the word features alone, with its own thresholds. The uncertainty of each score was estimated with 1,000 bootstrap resamples of the test set. Works were resampled instead of books, since editions of one work share a description and are not independent.

# %% cutoffs: one per shelf, chosen on validation; the default is logit 0 (probability 0.5)
cut = np.array([best_cutoff(Y[va][:, j], S_va[:, j]) for j in range(K)])
results['model, cutoff 0'] = scores(Y[te], S_te, 0.0)
results['model, cutoff per shelf'] = scores(Y[te], S_te, cut)

# %% ablation: the same model on word features only, with its own validation cutoffs
nw = len(word.vocabulary_)
W_w, b_w, _ = fit_lr(X_tr[:, :nw], Y[tr], X_va[:, :nw], Y[va])
S_va_w, S_te_w = decision(X_va[:, :nw], W_w, b_w), decision(X_te[:, :nw], W_w, b_w)
cut_w = np.array([best_cutoff(Y[va][:, j], S_va_w[:, j]) for j in range(K)])
results['model, words only'] = scores(Y[te], S_te_w, cut_w)
pd.DataFrame(results).T.round(3)

# %% bootstrap over test works: 95% intervals for macro-F1 and paired differences (cutoffs fixed)
systems = {'label prior': np.zeros((len(te), K), bool), 'every shelf': np.ones((len(te), K), bool),
           'shelf name in text': KW_te >= 0.5, 'model, cutoff 0': S_te >= 0.0,
           'model, words only': S_te_w >= cut_w, 'model, cutoff per shelf': S_te >= cut}
Yt = Y[te].astype(bool)
g = pd.factorize(elig['group'].iloc[te])[0]
G = csr_matrix((np.ones(len(te), np.float32), (g, np.arange(len(te)))))
# per work and shelf: true positives, false positives, false negatives of every system
counts = np.hstack([G @ np.hstack([Yt & p, ~Yt & p, Yt & ~p]).astype(np.float32) for p in systems.values()])


def macro_f1(c):
    c = c.reshape(*c.shape[:-1], len(systems), 3, K)
    tp, fp, fn = c[..., 0, :], c[..., 1, :], c[..., 2, :]
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0).mean(-1)


point = macro_f1(counts.sum(0))
for k, name in enumerate(systems):
    assert np.isclose(point[k], results[name]['macro_f1'], atol=1e-6), name
rng = np.random.default_rng(0)
n_works = G.shape[0]
boot = np.array([macro_f1(np.bincount(rng.integers(0, n_works, n_works), minlength=n_works) @ counts)
                 for _ in range(1000)])
names = list(systems)
ci = pd.DataFrame({'macro_f1': point, 'low': np.percentile(boot, 2.5, 0), 'high': np.percentile(boot, 97.5, 0)},
                  index=names)
best = names.index('model, cutoff per shelf')
for other in ['model, cutoff 0', 'shelf name in text', 'model, words only']:
    diff = boot[:, best] - boot[:, names.index(other)]
    ci.loc[f'difference: cutoff per shelf - {other}'] = [point[best] - point[names.index(other)],
                                                         *np.percentile(diff, [2.5, 97.5])]
print(f'{n_works} test works, 1000 resamples')
ci.round(3)

# %% Fig. 6: test macro-F1 with 95% bootstrap intervals
c6 = ci.loc[names].sort_values('macro_f1')
fig, ax = plt.subplots(figsize=(6, 2.8))
ax.barh(c6.index, c6['macro_f1'], height=0.6,
        color=['#257de0' if n == 'model, cutoff per shelf' else GRAY for n in c6.index],
        xerr=[c6['macro_f1'] - c6['low'], c6['high'] - c6['macro_f1']], error_kw={'ecolor': '#191919', 'lw': 1})
for i, v in enumerate(c6['macro_f1']):
    ax.text(c6['high'].iloc[i] + 0.01, i, f'{v:.3f}', va='center')
ax.set(xlabel='test macro-F1', xlim=(0, max(c6['high']) + 0.1))
ax.grid(axis='y', visible=False)
save(6)

# %% [markdown]
# *Fig. 6. Macro-F1 on the test set for each baseline and model. Error bars are 95% percentile intervals from 1,000 bootstrap resamples of test works, with the thresholds held fixed; they reflect sampling of the test set only, not variation between training runs.*

# %% [markdown]
# The description carries substantial information about a book's shelves, well beyond what a keyword rule extracts. The label prior scores zero, as expected. The keyword baseline reaches a macro-F1 of only 0.106, because reader-defined shelf names rarely occur in publisher descriptions; a book shelved as `contemporary` seldom describes itself as contemporary. The learned model reaches a macro-F1 of 0.454 and a micro-F1 of 0.480, more than four times the keyword baseline.
#
# The intervals in Fig. 6 are narrow. Every paired difference excludes zero: thresholds selected per shelf add 0.113 (95% interval 0.105 to 0.121) over the default threshold, and the model exceeds the keyword baseline by 0.348 (0.335 to 0.360). The every-shelf baseline reaches 0.056, half the score of the keyword baseline. Trained on word features alone, the model reaches 0.425, so the character features add 0.029 (0.025 to 0.033).
#
# Threshold selection accounts for a large part of the score. At the default threshold of 0.5, the balanced class weights make the model over-predict: it assigns 1.59 shelves per book, whereas books carry about 0.44 on average, and its macro-F1 is 0.341. Thresholds selected on the validation set reduce the rate to 0.61 shelves per book and raise macro-F1 to 0.454. AUC is independent of the threshold and is therefore identical in both rows.
#
# The difference between AUC and AP characterizes the remaining errors. An AUC of 0.933 means that, averaged over shelves, a randomly chosen book that carries a shelf receives a higher score than a randomly chosen book that does not about 93% of the time; the ranking is good. AP, at 0.397, reflects the precision among the highest-ranked books and is much lower because positives are rare. A small error rate on the large number of negative books suffices to place many false positives at the top of the ranking.

# %% per shelf (test)
P = S_te >= cut
per_shelf = pd.DataFrame({
    'test_positives': Y[te].sum(0).astype(int),
    'f1': f1_score(Y[te], P, average=None, zero_division=0),
    'precision': precision_score(Y[te], P, average=None, zero_division=0),
    'recall': recall_score(Y[te], P, average=None, zero_division=0),
    'auc': [roc_auc_score(Y[te][:, j], S_te[:, j]) for j in range(K)],
    'keyword_f1': f1_score(Y[te], KW_te >= 0.5, average=None, zero_division=0),
    'near_cutoff_ratio': near,
}, index=SHELVES)
per_shelf.sort_values('f1').round(3)

# %% Fig. 7: per-shelf test F1 (this F1 order is used for every per-shelf figure below)
order = per_shelf.sort_values('f1').index
y = np.arange(K)
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(y, per_shelf.loc[order, 'f1'], height=0.7, color=[color(s) for s in order], label='model')
ax.scatter(per_shelf.loc[order, 'keyword_f1'], y, marker='D', s=20, color='#191919', label='keyword baseline')
ax.set(yticks=y, yticklabels=order, xlabel='test F1', xlim=(0, 1))
ax.grid(axis='y', visible=False)
ax.legend(loc='lower right', frameon=False)
save(7)

# %% [markdown]
# *Fig. 7. Test F1 of the model for each shelf (bars) and of the keyword baseline (diamonds), with shelves ordered by model F1.*

# %% Fig. 8: precision-recall curves of the focus shelves; dot = threshold chosen on validation
fig, ax = plt.subplots(figsize=(5, 4))
for s in focus:
    j = SHELVES.index(s)
    prec, rec, _ = precision_recall_curve(Y[te][:, j], S_te[:, j])
    ax.plot(rec, prec, color=FOCUS[s], linewidth=2, label=s)
    ax.plot(per_shelf.loc[s, 'recall'], per_shelf.loc[s, 'precision'], 'o', markersize=8, color=FOCUS[s],
            markeredgecolor='white', markeredgewidth=2)
ax.set(xlabel='recall', ylabel='precision', xlim=(0, 1), ylim=(0, 1.02))
ax.legend(frameon=False)
save(8)

# %% [markdown]
# *Fig. 8. Precision-recall curves on the test set for the three focus shelves. The dot on each curve marks the precision and recall at the threshold selected on the validation set.*

# %% [markdown]
# ### B. Per-Shelf Results
#
# The average conceals large differences between shelves (Fig. 7). `manga` reaches an F1 of 0.812, with a precision of 0.86 and a recall of 0.77. `historical-romance` (0.685) and `m-m` (0.627) also perform well. At the other extreme, `contemporary` reaches an F1 of 0.244 with a precision of 0.19, and `contemporary-romance` reaches 0.279. These two shelves also have the lowest AUC (0.834 and 0.859). A low AUC indicates that the model fails to separate their books from the rest even in its ranking, so no choice of threshold can correct them (Fig. 8). In Fig. 8, the precision of `manga` stays at about 0.9 or above for recall between about 0.2 and 0.7, whereas the precision of `contemporary` is below 0.3 at every recall above about 0.2. The keyword baseline is weak on almost every shelf (Fig. 7). It is strongest on `bdsm` (0.386), whose name publishers do use in descriptions, and on `new-adult` (0.207).
#
# The per-shelf results thus give a qualified answer to the research question. For shelves with a distinctive vocabulary, the description predicts the shelf with useful precision and recall; for shelves such as `contemporary`, it largely does not. The next analysis examines the cause.

# %% does label noise or label count explain per-shelf F1?
rho_noise, p_noise = spearmanr(per_shelf['near_cutoff_ratio'], per_shelf['f1'])
rho_count, p_count = spearmanr(per_shelf['test_positives'], per_shelf['f1'])
print(f'Spearman with F1 (n = {K} shelves): near-cutoff ratio {rho_noise:+.2f} (p = {p_noise:.3f}), '
      f'test positives {rho_count:+.2f} (p = {p_count:.3f})')
# other lower edges of the band, to check that the result does not depend on the 5% choice
for lo in (0.025, 0.05, 0.075):
    r = ((share[tr] >= lo) & (share[tr] < MIN_FRAC)).sum(0) / Y[tr].sum(0)
    rho, p = spearmanr(r, per_shelf['f1'])
    print(f'band {lo:.1%} to {MIN_FRAC:.0%}: rho {rho:+.2f} (p = {p:.3f})')

# %% Fig. 9: per-shelf F1 against label noise and against label count
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
for ax, col, xlabel, rho in [(axes[0], 'near_cutoff_ratio', 'near-cutoff ratio (log scale)', rho_noise),
                             (axes[1], 'test_positives', 'positive test books (log scale)', rho_count)]:
    ax.scatter(per_shelf[col], per_shelf['f1'], s=40, c=[color(s) for s in SHELVES],
               edgecolors='white', linewidths=1.5, zorder=3)
    for s in focus:  # labels on all 15 collide; the per-shelf table gives the rest
        ax.annotate(s, (per_shelf.loc[s, col], per_shelf.loc[s, 'f1']), xytext=(5, 3), textcoords='offset points')
    ax.set(xscale='log', xlabel=xlabel, title=f'Spearman rho = {rho:+.2f}')
    ax.minorticks_off()
    ax.xaxis.set_major_locator(plt.LogLocator(subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%g'))
axes[0].set_ylabel('test F1')
save(9)

# %% [markdown]
# *Fig. 9. Test F1 of each shelf against its near-cutoff ratio (left) and against its number of positive test books (right), both on logarithmic axes. Each point is one shelf, and the three focus shelves are labeled; the panel titles give Spearman's rank correlation over the 15 shelves.*

# %% [markdown]
# ### C. Label Noise and Performance
#
# The hypothesis of Section III-D was tested by comparing each shelf's label-noise ratio with its test F1 (Fig. 9). Spearman's rank correlation was used because only 15 shelves are available, the ratio is heavily skewed (from 0.03 to 5.51), and the question is whether noisier shelves score lower, irrespective of the functional form of the relationship. The correlation is $\rho = -0.82$ ($p < 0.001$): the shelves on which readers disagree most are those the model predicts worst. Moving the lower edge of the band to 2.5% or 7.5% gives $\rho = -0.87$ and $-0.83$, so the result does not depend on the choice of 5%. The number of positive test books is only weakly related to F1 ($\rho = +0.23$, $p = 0.41$), so the weak shelves are not weak merely because they are small. For example, `contemporary` and `historical` have a similar number of positive test books (1,833 and 2,307), yet their F1 scores are 0.244 and 0.525.
#
# The correlation is consistent with a ceiling set by the labels. If readers do not agree on whether a book is `contemporary`, a classifier trained to reproduce their votes inherits that disagreement. With 15 shelves, however, a correlation does not establish a cause, and it does not exclude other explanations, such as the absence of a distinctive vocabulary discussed in Section V-D.
#
# A direct test of the hypothesis rescored each shelf after removing the test books whose share for that shelf lies just below the threshold, between 5% and 10%. All of these books are negatives, so their removal raises precision by construction, even if the books are no harder than other negatives. As a control, the same number of randomly chosen negatives outside the band was removed instead, 20 times per shelf. The effect of the band is the difference between the two scores.

# %% clean-label check: drop test books in the 5-10% band of a shelf, versus dropping as many random negatives
band = (share[te] >= 0.05) & (share[te] < MIN_FRAC)
rng = np.random.default_rng(0)
rows = []
for j, s in enumerate(SHELVES):
    neg = np.flatnonzero(~Yt[:, j] & ~band[:, j])
    control = []
    for _ in range(20):
        keep = np.ones(len(te), bool)
        keep[rng.choice(neg, band[:, j].sum(), replace=False)] = False
        control.append(f1_score(Yt[keep, j], P[keep, j], zero_division=0))
    rows.append({'band_books': band[:, j].sum(), 'f1_all': per_shelf.loc[s, 'f1'],
                 'f1_random_negatives_dropped': np.mean(control),
                 'f1_band_dropped': f1_score(Yt[~band[:, j], j], P[~band[:, j], j], zero_division=0)})
clean = pd.DataFrame(rows, index=SHELVES)
clean['band_effect'] = clean['f1_band_dropped'] - clean['f1_random_negatives_dropped']
clean.loc[order].round(3)

# %% Fig. 10: clean-label check per shelf
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(clean.loc[order, 'f1_all'], y, s=40, facecolors='none', edgecolors='#666666', label='all test books')
ax.scatter(clean.loc[order, 'f1_random_negatives_dropped'], y, s=40, marker='x', color='#666666',
           label='random negatives dropped')
ax.scatter(clean.loc[order, 'f1_band_dropped'], y, s=40, color='#191919', label='5-10% band dropped')
ax.set(yticks=y, yticklabels=order, xlabel='test F1', xlim=(0, 1))
ax.grid(axis='y', visible=False)
ax.legend(loc='lower right', frameon=False)
save(10)

# %% [markdown]
# *Fig. 10. Test F1 per shelf on all test books (circles), after removing a random set of negatives (crosses), and after removing the negatives in the 5% to 10% band of the shelf (dots). Both removals drop the same number of books. Shelves are ordered as in Fig. 7.*

# %% [markdown]
# Removing the band books raised F1 above the control on 14 of the 15 shelves, by between 0.029 and 0.135 (Fig. 10). For `manga`, which has only 15 band books in the test set, the difference was 0.002. Because only negatives are removed, the possible gain is bounded by a shelf's recall, so shelves are best compared at similar recall. Among shelves with a recall between 0.30 and 0.35, the gain was 0.061 for `contemporary-romance`, 0.055 for `contemporary`, and 0.042 for `fantasy`, against 0.029 for `new-adult`, whose ratio is the lowest of the four. Among shelves with a recall between 0.45 and 0.48, it was 0.077 for `erotica`, against 0.040 for `chick-lit` and 0.033 for `young-adult`. Among shelves with a recall between 0.69 and 0.79, it was 0.121 for `historical` and between 0.087 and 0.093 for `m-m` and `historical-romance`, which have lower ratios, and 0.002 for `manga`. The pattern does not hold among the four shelves with a recall between 0.57 and 0.64: `paranormal-romance`, with the highest ratio (1.61), gained 0.091, less than `paranormal` (1.43) at 0.135 and about as much as `bdsm` (1.15) at 0.093, although the ratios in this group span only 1.15 to 1.61. At similar recall, the shelves with more contested labels thus tend to lose more precision to books just below the threshold, in three of the four groups, which partly supports the hypothesis on the side of precision.
#
# The check leaves the ranking of the shelves unchanged. `contemporary` remains the weakest shelf, with an F1 of 0.314, and its recall of 0.343 cannot be affected by removing negatives. The low recall of the contested shelves is therefore not explained by this test. The test also understates the effect for shelves with diffuse votes, since the band of 5% to 10% omits the many `contemporary` negatives with smaller shares (Fig. 4).

# %% strongest features per shelf
Wn = W.cpu().numpy()
top = pd.DataFrame([FEATURES[np.argsort(-Wn[:, j])[:8]] for j in range(K)], index=SHELVES)
print(top.to_string(header=False))

# %% [markdown]
# ### D. Learned Features
#
# The output above lists the eight terms with the largest positive weight for each shelf. For the well-predicted shelves, these are the terms a human reader would expect: `historical-romance` relies on *lady*, *earl*, *lord*, and *duke*; `bdsm` on *dom*, *master*, and *submissive*; `manga` on the word *manga* and fragments of Japanese names; and `m-m` on *gay* and male pronouns. The model therefore bases its predictions for these shelves on genre evidence, which supports their validity.
#
# For `contemporary` and `contemporary-romance`, the strongest terms are names of particular series and places, such as *Virgin River*, *Idaho*, and *Lucky Harbor*. We hypothesize that contemporary romance has little vocabulary of its own, because it is defined mainly by the absence of other markers, namely a present-day setting without historical or supernatural elements. The weights are consistent with this hypothesis, since the model's strongest evidence for these shelves consists of popular series, which cannot generalize to a standalone novel; the hypothesis was not tested directly. A few character-level features, such as single punctuation marks, also rank among the top terms. They may reflect the formatting conventions of particular publishers, which was not verified.

# %% most confident errors: 10 false positives and 10 false negatives, all shelves pooled
margin = S_te - cut
info = elig.iloc[te].reset_index(drop=True)


def errors(mask, sign):
    i, j = np.nonzero(mask)
    k = np.argsort(sign * margin[i, j])[:10]
    return pd.DataFrame({'shelf': np.array(SHELVES)[j[k]], 'vote_share': share[te][i[k], j[k]].round(3),
                         'content_votes': info['sh_votes'].iloc[i[k]].values,
                         'margin': margin[i[k], j[k]].round(2), 'title': info['title'].iloc[i[k]].values,
                         'description': info['description'].iloc[i[k]].str.replace(r'\s+', ' ', regex=True).str.slice(0, 150).values})


with pd.option_context('display.max_colwidth', 150, 'display.width', 250):
    print('false positives\n', errors(~Yt & P, -1).to_string(index=False))
    print('\nfalse negatives\n', errors(Yt & ~P, 1).to_string(index=False))

# %% [markdown]
# ### E. Error Analysis
#
# The output above lists the ten most confident false positives and false negatives over all shelves, with the shelf's share of the book's content votes. Seven of the ten false positives have a share between 5% and 10%: readers did place these books on the predicted shelf, but too few of them for the label to pass the threshold. One book, *The Mistress Files*, appears twice, as a false positive for both `bdsm` and `erotica`. The other three false positives are Chinese-language novels whose descriptions are written in romanized form and which the model assigns to `manga`, two of them with a recorded `manga` share of zero. We hypothesize that the character features treat romanized Chinese syllables like the fragments of Japanese names that indicate `manga` (Section V-D); this was not verified.
#
# All ten false negatives lie close above the threshold, with shares between 10.0% and 17.6%. Four rest on 10 to 16 content votes, where a single vote fewer would remove the label. For example, *Alpha Feud*, presented as a paranormal shifter romance, is labeled `contemporary` because one of its ten votes is for that shelf. One description is the summary of an audiobook that begins with its running time. Most of the confident errors therefore arise at the label boundary, in agreement with Section V-C, and not from descriptions that the model misreads. Because the examples were selected for high confidence, they show which kinds of errors occur but not how often.

# %% [markdown]
# ### F. Limitations
#
# The labels are a single, unverified snapshot of reader votes, so measured performance is bounded by their quality. The evaluated books already have at least 10 genre votes and are more established than a typical new release. The model ignores correlations between shelves, and near-duplicate shelves are kept separate. In the earlier run, merging the three pairs of near-duplicate shelves into 12 labels raised the scores, but this changes the task to an easier one and does not indicate a better model. Finally, the optimizer settings were not tuned and the model had not converged (Section IV-F).

# %% [markdown]
# ## VI. Related Work
#
# The dataset was introduced by Wan and McAuley [1], who used Goodreads interactions to study recommendation when a user's actions form a monotonic chain of shelving, reading, rating, and recommending a book. Their models learn from users' past interactions with each item, which a newly released book lacks. The present work addresses the complementary, item-side problem of characterizing a new book from its own content. Wan et al. [2] used reviews from the same crawl for fine-grained spoiler detection, a classification task on review sentences. The romance file contains descriptions and no reviews, so a direct comparison with that task is not possible.
#
# The closest task in the literature is tag suggestion in social bookmarking systems. Katakis et al. [4] treated the suggestion of user-assigned tags for BibSonomy bookmarks and publications as multi-label text classification, with one naive Bayes classifier per tag over a bag of words and a single confidence threshold shared by all tags. As in our work, they kept only frequent tags (208 for bookmarks and 159 for publications), and they describe user tags as personal and noisy: users choose tags "based on their personal opinions, their knowledge background and their preferences," which "creates a noisy tag space." Their setting differs from ours in three respects. The text classifier served only as a fallback for items and users absent from the training data. The tags of a post were chosen by one user, whereas our labels aggregate the votes of many readers. Finally, they evaluated with example-based measures averaged over posts: precision is the fraction of a post's suggested tags that are correct, and the per-post F is the Dice coefficient of the suggested and true tag sets, a monotone function of their Jaccard similarity. Their F of 0.0856 on their own held-out split therefore cannot be compared with our macro-F1, which averages over shelves instead of books.
#
# Wang and Manning [3] showed that linear classifiers over word unigrams and bigrams, such as naive Bayes and linear support vector machines, are strong baselines for sentiment and topic classification, and that bigrams improve performance. Our results are consistent with theirs: a linear model over TF-IDF features performs well above trivial baselines, and adding character n-grams to word features raised macro-F1 by 0.029 (Section V-A). Wang and Manning also note that bag-of-words models cannot capture meaning arising from the composition of words. Our error analysis indicates a different limitation: the most confident errors lie at the boundary of the label rule, on books whose shelf received slightly more or slightly less than 10% of the votes, which no model of the text can resolve.
#
# Our selection of one threshold per shelf on a validation set is the SCut strategy studied by Yang [5], who found per-category thresholds "particularly effective when macro-averaged performance is the target function to optimize," but also found that they overfit for rare categories with few training examples. In our data, the rarest shelf still has several hundred positive books in the validation set, and the thresholds were evaluated once on a separate test set, where they raised macro-F1 from 0.341 to 0.454. Lipton et al. [6] showed that the F1-optimal decision rule can be written in terms of the Jaccard index of the optimal classifier, and that for calibrated probabilities the optimal threshold is half the achievable F1 and therefore never above 0.5. Our probabilities are not calibrated, because the balanced class weights inflate them, which is consistent with the over-prediction we observed at the default threshold. Lipton et al. also proved that for an uninformative classifier the F1-optimal rule predicts every example positive, with an expected F1 of $2p/(1+p)$ for a base rate $p$. Our every-shelf baseline is therefore the best score attainable without information from the text. They further warn that macro-F1 weights rare labels heavily and can hide such over-prediction. Our model predicts 0.61 shelves per book against about 0.44 true shelves, so it does not show the extreme over-prediction they describe, although this was not checked shelf by shelf.
#
# We are not aware of published results for shelf prediction on this dataset, so our F1 scores cannot be compared with prior work directly; the baselines of Section II-D serve as the points of comparison.

# %% [markdown]
# ## VII. Conclusion
#
# Given only its description, a TF-IDF model predicts a romance book's shelves well when the shelf corresponds to a distinctive vocabulary, as for `manga`, `historical-romance`, and `m-m`, with F1 between 0.63 and 0.81. It predicts poorly for shelves whose labels readers contest, such as `contemporary`, with F1 below 0.3. Across all 15 shelves, the model scores more than four times higher than a keyword baseline, and the strongest predictor of where it fails is reader disagreement about the label rather than the amount of training data. Removing the most contested negatives from the test set tended to raise precision more for the contested shelves at similar recall, but did not change their low recall, so disagreement explains part of the gap and the lack of a distinctive vocabulary remains a plausible second cause. For a cold-start system, descriptions are therefore a reliable source of genre tags for some shelves and an unreliable one for others.

# %% [markdown]
# ## References
#
# [1] M. Wan and J. McAuley, "Item recommendation on monotonic behavior chains," in *Proc. 12th ACM Conf. Recommender Systems (RecSys)*, 2018, pp. 86-94, doi: 10.1145/3240323.3240369.
#
# [2] M. Wan, R. Misra, N. Nakashole, and J. McAuley, "Fine-grained spoiler detection from large-scale review corpora," in *Proc. 57th Annu. Meeting Assoc. Computational Linguistics (ACL)*, 2019, pp. 2605-2610. [Online]. Available: https://aclanthology.org/P19-1248/
#
# [3] S. Wang and C. D. Manning, "Baselines and bigrams: Simple, good sentiment and topic classification," in *Proc. 50th Annu. Meeting Assoc. Computational Linguistics (ACL), Vol. 2: Short Papers*, 2012, pp. 90-94. [Online]. Available: https://aclanthology.org/P12-2018/
#
# [4] I. Katakis, G. Tsoumakas, and I. Vlahavas, "Multilabel text classification for automated tag suggestion," in *Proc. ECML PKDD Discovery Challenge 2008 (RSDC'08)*, A. Hotho, D. Benz, R. Jäschke, and B. Krause, Eds., Antwerp, Belgium, 2008, pp. 75-83. [Online]. Available: https://www.kde.cs.uni-kassel.de/ws/rsdc08/pdf/all_rsdc_v2.pdf
#
# [5] Y. Yang, "A study of thresholding strategies for text categorization," in *Proc. 24th Annu. Int. ACM SIGIR Conf. Research and Development in Information Retrieval (SIGIR)*, New Orleans, LA, USA. New York, NY, USA: ACM, 2001, pp. 137-145, doi: 10.1145/383952.383975.
#
# [6] Z. C. Lipton, C. Elkan, and B. Naryanaswamy, "Optimal thresholding of classifiers to maximize F1 measure," in *Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2014), Part II*, T. Calders, F. Esposito, E. Hüllermeier, and R. Meo, Eds., ser. Lecture Notes in Computer Science, vol. 8725. Berlin, Heidelberg: Springer, 2014, pp. 225-239, doi: 10.1007/978-3-662-44851-9_15.


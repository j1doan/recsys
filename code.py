# %% [markdown]
# # Goodreads romance books: first look
# One JSON object per line. Book metadata only (no per-user reviews).

# %% imports
# standard library
import gzip
import json
import os
from collections import Counter

# data
import numpy as np
import pandas as pd

# environment
from dotenv import find_dotenv, load_dotenv

# modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multiclass import OneVsRestClassifier

# %% dotenv
load_dotenv(find_dotenv('secret.env', usecwd=True))
PATH = os.environ['GOODREADS_PATH']

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

# %% first record
df.iloc[0].to_dict()

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
d_words = d.str.split().str.len().describe()[['mean', '50%', 'max']].round(0).to_dict()
print(f'{len(d)} with description; words: {d_words}')
d.iloc[:3].tolist()

# %% similar_books: how many point to books inside this file?
ids = set(df['book_id'])
sim = [x for l in df['similar_books'] for x in l]
inside = sum(x in ids for x in sim) / max(len(sim), 1)
print(f'{len(sim)} links; {inside:.3f} point inside the file')


# %% [markdown]
# ## Task D: predict shelves from the description
# A "shelf" is a tag users put a book on (`popular_shelves`: name + how many users).
# Labels come from those counts by a rule (below). No hand labelling.
# Tune EXCLUDE, MIN_VOTES, MIN_FRAC, K after looking at the top-shelves and sweep cells.

# %% label rule parameters
# EXCLUDE holds shelves that are not about content: reading status, format, ownership, the genre itself.
# MIN_VOTES: a book needs this many content-shelf votes in total.
# MIN_FRAC: a shelf is a label when it has this share of the book's content-shelf votes.
# K: number of shelves used as labels.
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
has_text = df['description'] != ''

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

# %% build labels with the chosen cutoff
# a shelf is positive for a book when its share of the book's content votes is at least MIN_FRAC
elig = df[has_text & (df['sh_votes'] >= MIN_VOTES)].reset_index(drop=True)
c = Counter(n for d, v in zip(elig['sh'], elig['sh_votes']) for n, k in d.items() if k / v >= MIN_FRAC)
SHELVES = [n for n, _ in c.most_common(K)]
Y = np.array([[d.get(s, 0) / v >= MIN_FRAC for s in SHELVES] for d, v in zip(elig['sh'], elig['sh_votes'])])
print(f'{len(elig)} books; {Y.shape[1]} labels; mean labels per book: {Y.sum(1).mean():.2f}')
pd.Series(Y.mean(0), index=SHELVES).sort_values(ascending=False).round(3)

# %% split by work (editions of one work stay on the same side)
tr, te = next(GroupShuffleSplit(test_size=0.2, random_state=0).split(elig, groups=elig['work_id']))
print(f'{len(tr)} train; {len(te)} test')

# %% baseline 1: label prior (score = share of train books with the label)
prior = Y[tr].mean(0)
pred_prior = np.tile(prior >= 0.5, (len(te), 1))
f1_prior = f1_score(Y[te], pred_prior, average='micro', zero_division=0)
print(f'prior micro-F1 {f1_prior:.3f}, AUC 0.5 by construction')

# %% baseline 2: TF-IDF + one-vs-rest logistic regression
# vectorizer is fit on train only
tfidf = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=5, stop_words='english', sublinear_tf=True)
Xtr = tfidf.fit_transform(elig['description'].iloc[tr])
Xte = tfidf.transform(elig['description'].iloc[te])
clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', class_weight='balanced')).fit(Xtr, Y[tr])
S = clf.decision_function(Xte)
P = S > 0
micro = f1_score(Y[te], P, average='micro')
macro = f1_score(Y[te], P, average='macro')
auc = roc_auc_score(Y[te], S, average='macro')
print(f'tfidf-LR micro-F1 {micro:.3f}, macro-F1 {macro:.3f}, macro-AUC {auc:.3f}')

# %% per-shelf results
pd.DataFrame({'shelf': SHELVES, 'test_positives': Y[te].sum(0),
              'F1': f1_score(Y[te], P, average=None).round(3),
              'AUC': [round(roc_auc_score(Y[te][:, j], S[:, j]), 3) for j in range(len(SHELVES))]})

# %% [markdown]
# # Predicting Goodreads Shelves from Book Descriptions
#
# CSE 258, Assignment 2. This notebook accompanies our presentation and follows the five parts of the assignment in order: the predictive task, the data, the model, the evaluation, and related work.
#
# On Goodreads, readers organize books into "shelves" that they name themselves. Shelves such as `historical-romance`, `paranormal`, or `m-m` work as community-assigned genre tags, and a recommender or a search page can use them to find books of a given kind. These tags only accumulate after people have read a book. A newly published book has no shelves, so a system that relies on shelves has no way to place it. This is an instance of the item cold-start problem discussed in the course.
#
# A new book does come with a description, usually written by the publisher. Our research question is whether a model that sees only a book's description can predict which shelves readers will put it on. We study this question with a TF-IDF representation of the description and one logistic regression per shelf, a standard linear text classifier.
#
# The short answer is that it depends on the shelf. For shelves whose books share a recognizable vocabulary, the description is a strong signal: on books the model has never seen, it reaches an F1 score of 0.81 for `manga`, 0.69 for `historical-romance`, and 0.63 for `m-m`. For shelves that readers apply inconsistently, the signal is weak, with an F1 of 0.24 for `contemporary` and 0.28 for `contemporary-romance`. Averaged over all 15 shelves, the model's macro-F1 is 0.45, compared with 0.11 for a rule that looks for the shelf's name in the text. Section 4 shows that the main reason some shelves fail is disagreement among readers about the label, and that the amount of training data plays a much smaller role.

# %% [markdown]
# ## 1. Predictive task
#
# ### Problem statement
#
# Each example is one book. The input is the text of its description, and the output is a vector of 15 binary labels, one for each of the 15 most common genre shelves. A book may carry several shelves or none, so this is a multi-label classification problem.
#
# The labels come from reader behavior. For every book, the dataset lists its most popular shelves and how many times readers placed the book on each. We treat a shelf as a true label for a book when it receives at least 10% of the book's genre-related shelf votes; section 2.2 explains this choice. The definition rests on an assumption: a shelf chosen by a tenth of a book's readers reflects a shared opinion about the book, while a shelf chosen by one or two readers does not.
#
# ### Simulating cold start
#
# The data is a single snapshot, so we cannot observe books at the moment of release. We approximate the cold-start setting by taking books whose shelves are known, hiding those shelves, and asking the model to recover them from the description alone. This approximation assumes that the descriptions of established books resemble those of new books. It also limits us to books with at least 10 genre votes, which are more established than a typical new release.
#
# ### Evaluation protocol
#
# We split the books into training (64%), validation (16%), and test (20%) sets. Different editions of the same work usually share a description, so we split by work, which places all editions of a book in the same set. Without this grouping, the model could be tested on descriptions it had already seen during training. The validation set is used only to choose a decision threshold for each shelf, and the test set is used once, after every choice has been fixed.
#
# We measure performance mainly with F1, the harmonic mean of precision (the share of predicted shelves that are correct) and recall (the share of true shelves that the model finds). We report micro-F1, which pools all book-shelf decisions and is therefore dominated by frequent shelves, and macro-F1, which averages the F1 of each shelf and gives every shelf equal weight. We treat macro-F1 as the primary metric because our question concerns individual shelves, including rare ones. We also report ROC AUC and average precision (AP), which measure how well the model ranks books before any threshold is applied. Accuracy is not informative for this task. Since 97.1% of book-shelf pairs are negative, a model that never predicts any shelf would be 97.1% accurate and entirely useless.
#
# ### Baselines
#
# We compare against two baselines that involve no learning. The label-prior baseline predicts a shelf whenever more than half of the training books carry it. No shelf is that common, so it never predicts anything and scores an F1 of zero; it marks the floor that any useful model must clear. The keyword baseline predicts a shelf whenever the shelf's name, such as "historical romance", appears in the description, which tests whether the task can be solved by looking for the obvious word. We also report our own model with the default threshold of probability 0.5, to separate the contribution of the learned weights from the contribution of threshold tuning.
#
# ### Validity
#
# We check whether the predictions can be trusted in three ways. First, the split guarantees that test books, and all other editions of them, play no role in training or in threshold selection. Second, we inspect the learned weights to confirm that the model relies on genre-related words. Third, since shelf votes are a noisy proxy for genre, we measure the noise in each shelf's labels (section 2.3) and test whether it explains where the model fails (section 4).

# %% [markdown]
# ## 2. Data, exploration, and preprocessing
#
# ### 2.1 Source
#
# We use the romance subset of the UCSD Goodreads dataset [1, 2], which was collected in late 2017 by crawling public Goodreads shelves. Each line of the file describes one book: its title, authors, description, publication details, rating summary, similar books, and most popular shelves with their vote counts. The file contains no user identifiers and no reviews, so the interaction-based models from the course, such as latent factor models and BPR, cannot be applied, and a content-based approach is the natural choice. The dataset is distributed for academic use only, so it is not included in this repository; the notebook reads a local copy whose path is set in `secret.env`.
#
# The following cells load the full file and examine each field.

# %% imports
import gzip
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from dotenv import find_dotenv, load_dotenv
from scipy.sparse import hstack
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

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
sim = [x for lst in df['similar_books'] for x in lst]
inside = sum(x in ids for x in sim) / max(len(sim), 1)
print(f'{len(sim)} links; {inside:.3f} point inside the file')

# %% [markdown]
# ### What the exploration shows
#
# The file contains 335,449 books. Of these, 90.9% have a non-empty description, with a median length of 147 words, so for most books the model has a full paragraph of text to work with.
#
# The remaining metadata is noisy. Publication year is missing for 30% of books and includes impossible values such as 1, 20015, and 65535, which are most likely data-entry errors. Rating counts are highly skewed: the median book has 52 ratings, the mean is 448, and the most-rated book has about 2.08 million. Language codes are missing for 31% of books, and Italian, Spanish, and Indonesian editions each number several thousand, so not every description is in English. Our model reads only the description, so these fields need no cleaning, but the mix of languages is a source of noise in the text itself.
#
# The shelf table shows why the labels need preprocessing. The most-used shelves are `to-read`, `currently-reading`, and `favorites`, which record what a reader did with a book and say nothing about its genre. Genre shelves such as `historical-romance` and `paranormal` appear further down the list. Finally, 83.8% of `similar_books` links point to other books in this file, which would make a link-prediction task possible; we leave that direction for future work.

# %% [markdown]
# ### 2.2 Constructing labels
#
# We turn the raw shelf votes into labels in four steps. First, we remove shelves that do not describe content: reading status (`to-read`), format (`kindle`, `ebook`), ownership (`owned`), and the tags `romance` and `fiction`, which nearly every book in this file carries and which cannot distinguish one book from another. These shelves are listed in `EXCLUDE`, and we call the remaining votes content votes. Second, we keep only books with at least 10 content votes, because with fewer votes a label could be decided by two or three readers. Third, a shelf is a positive label for a book when it receives at least 10% of the book's content votes. Fourth, we select the 15 shelves that are most often positive among the training books. Selecting them from training data alone keeps the test set unseen.
#
# The 10% threshold trades the quality of the labels against their quantity. The sweep below shows that, with 10 required votes, a 5% threshold gives the 15th most common shelf 10,511 positive books, a 10% threshold gives it 3,105, and a 20% threshold only 376. A low threshold admits shelves that only a handful of readers used, while a high threshold leaves too few positive examples to learn from. We use 10%, which keeps several thousand positive examples for every shelf.

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
pd.DataFrame({'train': Y[tr].mean(0), 'validation': Y[va].mean(0), 'test': Y[te].mean(0)}, index=SHELVES).round(3)

# %% [markdown]
# After filtering, 299,789 books remain: 191,692 for training, 47,687 for validation, and 60,410 for testing. The labels are sparse. 65.6% of the books carry none of the 15 shelves, the most common shelf (`m-m`) applies to only 6.8% of training books, and the least common (`fantasy`) applies to 1.1%. Shelf frequencies are nearly identical across the three splits, so the split does not shift the label distribution.
#
# Some of the 15 shelves are near-synonyms, such as `historical` and `historical-romance`, because readers use both names. We keep them as separate labels because they are separate shelves on Goodreads, but the overlap is one reason to expect some labels to be unreliable.
#
# ### 2.3 Measuring label noise
#
# A fixed 10% threshold draws a hard line through what is really a continuous vote share. A book that receives 9% of its votes as `contemporary` is labeled negative, while one at 11% is labeled positive, even though readers treated the two books almost the same way. When many books sit just below the line, the labels for that shelf are unstable. We measure this for each shelf as the number of training books just below the threshold (5% to 10% of votes) divided by the number at or above it. A ratio near zero means readers agree about which books belong on the shelf, and a high ratio means that many books split their readers.

# %% label noise: books just under the cutoff (5-10% of votes) per book at or above it
near = ((share[tr] >= 0.05) & (share[tr] < MIN_FRAC)).sum(0) / Y[tr].sum(0)
pd.Series(near, index=SHELVES).sort_values().round(2)

# %% [markdown]
# The ratios differ by two orders of magnitude. For `manga` the ratio is 0.03, because readers either shelve a book as manga or they do not. For `contemporary` it is 5.51, which means that for every book labeled `contemporary` there are about five more just under the threshold that are labeled negative. `contemporary-romance` (3.61), `fantasy` (3.30), and `erotica` (2.78) follow. We hypothesize that these high-ratio shelves will be the hardest to predict regardless of the model, because their labels depend partly on where an arbitrary line falls.

# %% [markdown]
# ## 3. Modeling
#
# ### 3.1 Formulation
#
# We formulate the task as 15 independent binary classification problems that share a single representation of the text. For shelf $j$ and a book with feature vector $x$, the model computes a score $s_j(x) = w_j^\top x + b_j$ and predicts the shelf when the score exceeds a threshold $t_j$. Treating the shelves independently ignores correlations between them, such as the overlap between `historical` and `historical-romance`, but it keeps the model simple to train and to interpret.
#
# The next cells define the evaluation code and run the two baselines, which require no training. `best_cutoff` chooses, for one shelf, the threshold that maximizes F1 on the validation set by scanning every distinct score. `scores` computes the metrics from section 1 on the test set.

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

# %% baseline 2: the shelf name appears in the description
def keyword(texts):
    norm = [' ' + re.sub(r'[^\w]+', ' ', t.lower()) + ' ' for t in texts]
    return np.array([[' ' + s.replace('-', ' ') + ' ' in t for s in SHELVES] for t in norm], dtype=float)

results['shelf name in text'] = scores(Y[te], keyword(elig['description'].iloc[te]), 0.5)

# %% [markdown]
# ### 3.2 Text representation
#
# We represent each description with TF-IDF features. The term frequency counts how often a term appears in a description, and the inverse document frequency down-weights terms that appear in many descriptions. Words that appear in nearly every romance blurb, such as "love" or "life", therefore contribute little, while distinctive words such as "duke" or "vampire" contribute more. We use two kinds of terms. Word unigrams and bigrams (50,000 features, with English stop words removed) capture topical vocabulary and multi-word names such as series titles. Character 3- to 5-grams (60,000 features, given half weight) capture fragments of words, which help with names, inflected forms, and descriptions in other languages. Both vectorizers are fit on the training descriptions only.
#
# This representation assumes that genre is expressed by which words appear, independent of their order. The assumption is reasonable for genres with a characteristic vocabulary, but it cannot capture meaning that depends on word order, or information that a description leaves out.

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
# ### 3.3 Classifier
#
# We train one logistic regression per shelf, implemented as a single linear layer with 15 outputs in PyTorch so that it runs on a GPU. The loss is the binary cross-entropy over all shelves plus an L2 penalty on the weights, which is the objective of scikit-learn's `LogisticRegression(C=1, class_weight='balanced')` divided by the number of training books. Balanced class weights increase the loss on positive examples in proportion to their rarity. Without them, the model could reach a low loss by predicting no shelves at all, since 97% of the targets are negative. We optimize with Adam on mini-batches of 4,096 books for 8 epochs. Each book uses only about 1,200 of the 110,000 features, so we keep the feature matrix in sparse form, which fits in the memory of an 8 GB laptop GPU.
#
# ### Choice of model
#
# We chose logistic regression on TF-IDF features because it is a well-established baseline for text classification [3], it is part of the course material, it trains in minutes on 190,000 documents, and its weights can be read directly, which lets us check in section 4 whether the model relies on sensible evidence. In earlier experiments, our group compared eight scikit-learn configurations on validation macro-F1. They varied the features (words only, or words and characters), the regularization strength (`C` of 0.25, 1, or 4), class weighting, and the classifier (logistic regression or a linear SVM). Logistic regression with word and character features and `C=1` performed best, and this notebook uses that configuration. Models that read word order, such as fine-tuned transformers, might capture more of a description's meaning, but they are far more expensive to train on this many documents, and we did not evaluate them.

# %% logistic regression on the GPU
# Same objective as sklearn's LogisticRegression(C=1, class_weight='balanced'), divided by n:
# mean weighted log-loss + ||W||^2 / (2 C n). Mini-batches keep memory small on an 8 GB card.
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def to_torch(x):
    x = x.tocoo()
    i = torch.from_numpy(np.vstack([x.row, x.col]).astype(np.int64))
    return torch.sparse_coo_tensor(i, torch.from_numpy(x.data), x.shape).to(DEV)


def fit_lr(x, y, C=1.0, epochs=8, batch=4096, lr=0.01, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(y)
    yt = torch.from_numpy(y).to(DEV)
    pos = yt.sum(0)
    w_pos, w_neg = n / (2 * pos), n / (2 * (n - pos))  # class_weight='balanced'
    W = torch.zeros(x.shape[1], y.shape[1], device=DEV, requires_grad=True)
    b = torch.zeros(y.shape[1], device=DEV, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
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
        print(f'epoch {epoch + 1}: loss {total / n:.4f}')
    return W.detach(), b.detach()


@torch.no_grad()
def decision(x, W, b, batch=8192):
    out = [(torch.sparse.mm(to_torch(x[i:i + batch]), W) + b).cpu().numpy() for i in range(0, x.shape[0], batch)]
    return np.vstack(out)


W, b = fit_lr(X_tr, Y[tr])
S_va, S_te = decision(X_va, W, b), decision(X_te, W, b)

# %% [markdown]
# The training loss is still decreasing at the eighth epoch, so this model has not fully converged. In our earlier experiments, scikit-learn's solver fit the same objective to convergence and reached a test macro-F1 of 0.484, against 0.454 here. The results below are therefore a slight underestimate of what this model can achieve.
#
# ## 4. Evaluation
#
# ### 4.1 Overall results
#
# We now choose each shelf's threshold on the validation set and evaluate every model once on the test set.

# %% cutoffs: one per shelf, chosen on validation; the default is logit 0 (probability 0.5)
cut = np.array([best_cutoff(Y[va][:, j], S_va[:, j]) for j in range(K)])
results['model, cutoff 0'] = scores(Y[te], S_te, 0.0)
results['model, cutoff per shelf'] = scores(Y[te], S_te, cut)
pd.DataFrame(results).T.round(3)

# %% [markdown]
# The results answer the first part of our question: the description carries substantial information about a book's shelves, far more than an obvious keyword rule can extract. The label prior scores zero, as expected. Searching the description for the shelf's name reaches a macro-F1 of only 0.106, because reader-invented shelf names rarely appear in publisher text; a book that readers shelve as `contemporary` almost never calls itself contemporary. The learned model reaches a macro-F1 of 0.454 and a micro-F1 of 0.480, more than four times the keyword baseline.
#
# Threshold tuning matters as much as the learned weights. With the default threshold of 0.5, the balanced class weights make the model generous: it predicts 1.59 shelves per book, while books carry about 0.44 on average, and its macro-F1 is 0.341. Thresholds tuned on validation reduce this to 0.61 shelves per book and raise macro-F1 to 0.454. AUC does not depend on the threshold, which is why it is identical in both rows.
#
# The gap between AUC and AP shows what kind of errors remain. An AUC of 0.933 means that, for a given shelf, a randomly chosen book that has the shelf receives a higher score than a randomly chosen book that does not about 93% of the time, averaged over shelves. In other words, the model ranks books well. AP, at 0.397, reflects how many of the highest-ranked books are correct, and it is much lower because positives are rare. Even a small error rate on the large number of negative books is enough to fill the top of the ranking with false positives.

# %% per shelf (test)
P = S_te >= cut
per_shelf = pd.DataFrame({
    'test_positives': Y[te].sum(0).astype(int),
    'f1': f1_score(Y[te], P, average=None, zero_division=0),
    'precision': precision_score(Y[te], P, average=None, zero_division=0),
    'recall': recall_score(Y[te], P, average=None, zero_division=0),
    'auc': [roc_auc_score(Y[te][:, j], S_te[:, j]) for j in range(K)],
    'near_cutoff_ratio': near,
}, index=SHELVES)
per_shelf.sort_values('f1').round(3)

# %% [markdown]
# ### 4.2 Results by shelf
#
# The average hides large differences between shelves. `manga` reaches an F1 of 0.812: 86% of the books the model shelves as manga are manga, and it finds 77% of all manga in the test set. `historical-romance` (0.685) and `m-m` (0.627) also perform well. At the other end, `contemporary` reaches an F1 of 0.244, with only 19% of its predictions correct, and `contemporary-romance` reaches 0.279. These two shelves also have the lowest AUC (0.834 and 0.859). A low AUC means the model cannot separate their books from the rest even in its ranking, so no choice of threshold would fix them.
#
# To the question of whether the model can predict a book's shelf from its description, the per-shelf results give a split answer. For genres with a distinctive vocabulary it can, with useful precision and recall. For genres such as `contemporary` it largely cannot. The next cell tests why.

# %% does label noise or label count explain per-shelf F1?
rho_noise = spearmanr(per_shelf['near_cutoff_ratio'], per_shelf['f1'])[0]
rho_count = spearmanr(per_shelf['test_positives'], per_shelf['f1'])[0]
print(f'rank correlation with F1: near-cutoff ratio {rho_noise:+.2f}, test positives {rho_count:+.2f}')

# %% [markdown]
# ### 4.3 Why some shelves fail
#
# We test the hypothesis from section 2.3 by comparing each shelf's label-noise ratio with its test F1. The Spearman rank correlation across the 15 shelves is -0.82: the shelves on which readers disagree most are the shelves the model predicts worst. The number of positive test books, in contrast, is only weakly related to F1 (+0.23), so the weak shelves are not weak simply because they are small. For example, `contemporary` has 1,833 positive test books and `historical` has 2,307, a similar number, yet `historical` reaches an F1 of 0.525 and `contemporary` only 0.244.
#
# This suggests that the ceiling on these shelves comes mainly from the labels. If readers themselves do not agree on whether a book is `contemporary`, a classifier trained to reproduce their votes inherits that disagreement. With only 15 shelves, the correlation is suggestive and not conclusive, and it does not rule out other explanations, such as the lack of a distinctive vocabulary that we turn to next.

# %% strongest features per shelf
Wn = W.cpu().numpy()
top = pd.DataFrame([FEATURES[np.argsort(-Wn[:, j])[:8]] for j in range(K)], index=SHELVES)
print(top.to_string(header=False))

# %% [markdown]
# ### 4.4 What the model learned
#
# The table above lists the eight terms with the largest positive weight for each shelf. For the well-predicted shelves, these are the terms a human reader would expect. `historical-romance` relies on lady, earl, lord, and duke; `bdsm` on dom, master, and submissive; `manga` on the word manga and fragments of Japanese names; and `m-m` on gay and male pronouns. The model uses genre evidence on these shelves, which supports the validity of its predictions there.
#
# For `contemporary` and `contemporary-romance`, the strongest terms are the names of particular series and places, such as Virgin River, Idaho, and Lucky Harbor. A contemporary romance is marked mainly by the absence of other markers, a present-day setting with no historical or supernatural elements, so the genre has little vocabulary of its own, and the model falls back on memorizing popular series. That strategy cannot generalize to a standalone novel. A few character-level features, such as single punctuation marks, also appear among the top terms, which suggests the model picks up formatting habits of particular publishers.
#
# ### 4.5 Errors and limitations
#
# In an earlier run of the same kind of model, our group read ten of the most confident errors alongside their descriptions and vote counts. Most of them were not failures to understand the text. In some cases, readers had split their votes across spellings (`manga`, `mangas`, `manga-manhwa`), so the exact shelf stayed under 10% even though the description fits it. In others, a sibling shelf such as `contemporary-romance` was positive while `contemporary` fell just below the threshold, or the label rested on two to five votes for a book whose description was about something else. Three of the ten were cases where the description never mentioned the relevant fact, such as a manga volume whose stored description is a prose plot summary. Ten examples chosen for high confidence show which kinds of errors occur, but not how often each kind occurs.
#
# The setup has further limitations. The labels are a single snapshot of reader votes that nobody verified, so any measured performance is bounded by their quality. The books we evaluate already have at least 10 genre votes and are more established than a typical new release. The model ignores correlations between shelves, and we keep near-duplicate shelves separate. In the earlier run, merging the three pairs of near-duplicate shelves into 12 labels raised the scores, but that answers an easier question and does not reflect a better model.
#
# ### 4.6 Answer to the research question
#
# Based on its description alone, a TF-IDF model can predict a romance book's shelves well when the shelf corresponds to a distinctive vocabulary, as with manga, historical romance, or m/m romance, where F1 ranges from 0.63 to 0.81. It predicts poorly when the shelf is one that readers apply inconsistently, as with contemporary romance, where F1 stays below 0.3. Across all 15 shelves, the model scores more than four times higher than a keyword rule, and the strongest predictor of where it fails is disagreement among readers about the label, not the amount of training data. For a cold-start system, this means that descriptions are a reliable source of genre tags for some shelves and an unreliable one for others.

# %% [markdown]
# ## 5. Related work
#
# The dataset was introduced by Wan and McAuley [1], who used Goodreads interactions to study recommendation when a user's actions form a monotonic chain: shelving a book, reading it, rating it, and recommending it. Their models learn from users' past interactions with each item, which a newly released book does not yet have. Our work addresses the complementary, item-side problem of characterizing a new book from its own content. Wan et al. [2] used reviews from the same crawl for fine-grained spoiler detection, a classification task on review sentences. Our romance file contains descriptions and no reviews, so we cannot compare with that task directly.
#
# On the modeling side, Wang and Manning [3] showed that linear classifiers over word unigram and bigram features, such as Naive Bayes and linear SVMs, are strong baselines for sentiment and topic classification, and that bigrams help. Our results are consistent with theirs. A linear model over TF-IDF features performs far above trivial baselines, and in our group's earlier experiments, adding character n-grams to word features raised macro-F1 by 0.022. Wang and Manning also note that bag-of-words models cannot capture meaning that arises from how words combine. Our error analysis points to a different limit: many errors come from the labels themselves, or from information that is absent from the description and that no text model could recover.
#
# We are not aware of published results for shelf prediction on this dataset, so we cannot compare our F1 scores with prior work directly; our comparison points are the baselines in section 4.
#
# ### References
#
# [1] Mengting Wan and Julian McAuley. 2018. Item Recommendation on Monotonic Behavior Chains. In *Proceedings of the 12th ACM Conference on Recommender Systems (RecSys '18)*. ACM, 86-94. https://doi.org/10.1145/3240323.3240369
#
# [2] Mengting Wan, Rishabh Misra, Ndapa Nakashole, and Julian McAuley. 2019. Fine-Grained Spoiler Detection from Large-Scale Review Corpora. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, 2605-2610. https://aclanthology.org/P19-1248/
#
# [3] Sida Wang and Christopher D. Manning. 2012. Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. In *Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, 90-94. https://aclanthology.org/P12-2018/

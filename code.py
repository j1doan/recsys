# %% [markdown]
# # Predicting Goodreads Shelves from Book Descriptions
#
# *CSE 258, Assignment 2*
#
# **Abstract.** Goodreads readers assign books to user-defined shelves that act as genre tags, but a newly published book has no shelves, which leaves shelf-based recommendation without input for new items. We study whether the description of a romance novel is sufficient to predict 15 common reader shelves. Using 299,789 books from the UCSD Goodreads dataset, we train one logistic regression per shelf on TF-IDF word and character features and evaluate it on 60,410 held-out books. The model reaches a macro-F1 of 0.454, compared with 0.106 for a keyword baseline. Performance varies widely across shelves, from an F1 of 0.812 for `manga` to 0.244 for `contemporary`. Across shelves, F1 is strongly correlated with a measure of reader disagreement about the label (Spearman $\rho = -0.82$) and only weakly with the number of training examples ($\rho = +0.23$). We conclude that descriptions predict shelves with a distinctive vocabulary well, and that the main limit on the remaining shelves is the consistency of the labels.

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
# Two baselines involve no learning. The label-prior baseline predicts a shelf whenever more than half of the training books carry it. No shelf is that frequent, so this baseline predicts nothing and attains an F1 of zero, which is the floor any useful model must exceed. The keyword baseline predicts a shelf whenever the shelf name, for example "historical romance", occurs in the description, and tests whether the task can be solved by matching the obvious term. The proposed model is additionally evaluated with the default threshold of probability 0.5, which separates the contribution of the learned weights from that of threshold selection.

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
# ### B. Exploratory Analysis
#
# The file contains 335,449 books, of which 90.9% have a non-empty description. The median description length is 147 words.
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
pd.DataFrame({'train': Y[tr].mean(0), 'validation': Y[va].mean(0), 'test': Y[te].mean(0)}, index=SHELVES).round(3)

# %% [markdown]
# After filtering, 299,789 books remained: 191,692 for training, 47,687 for validation, and 60,410 for testing. The labels are sparse. Of all books, 65.6% carry none of the 15 shelves; the most frequent shelf, `m-m`, applies to 6.8% of training books and the least frequent, `fantasy`, to 1.1%. Shelf frequencies are nearly identical across the three sets, so the split does not shift the label distribution.
#
# Several of the 15 shelves are near-synonyms, such as `historical` and `historical-romance`, because readers use both names. They are kept as separate labels because they are separate shelves on Goodreads, but this overlap is one reason to expect some labels to be unreliable.

# %% [markdown]
# ### D. Label Noise
#
# A fixed threshold of 10% imposes a hard boundary on what is a continuous vote share. A book that receives 9% of its votes as `contemporary` is labeled negative, and one that receives 11% is labeled positive, although readers treated the two books almost identically. When many books lie just below the boundary, the labels of that shelf are unstable. We quantify this effect for each shelf as the ratio of training books just below the threshold, with 5% to 10% of their votes, to books at or above it. A ratio near zero indicates that readers agree on which books belong to the shelf, and a high ratio indicates that many books divide their readers. The lower edge of 5%, half the threshold, was fixed before any model was trained and was not adjusted afterwards. A different band would change the exact values, so the ratio is used only to rank shelves by how contested their labels are.

# %% label noise: books just under the cutoff (5-10% of votes) per book at or above it
near = ((share[tr] >= 0.05) & (share[tr] < MIN_FRAC)).sum(0) / Y[tr].sum(0)
pd.Series(near, index=SHELVES).sort_values().round(2)

# %% [markdown]
# The ratios span two orders of magnitude. For `manga` the ratio is 0.03, since readers either shelve a book as manga or do not. For `contemporary` it is 5.51, meaning that for every book labeled `contemporary` about five more lie just below the threshold and are labeled negative. `contemporary-romance` (3.61), `fantasy` (3.30), and `erotica` (2.78) follow. We hypothesize that the shelves with high ratios will be the most difficult to predict regardless of the model, because their labels depend in part on where an arbitrary boundary falls.

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
# The code below implements the evaluation and the two baselines, which require no training. The function `best_cutoff` selects, for one shelf, the threshold that maximizes F1 on the validation set by scanning every distinct score. The function `scores` computes the metrics of Section II-C on the test set.

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
# ### F. Convergence
#
# The training loss was still decreasing at the eighth epoch, so the model had not fully converged. In our group's earlier experiments, scikit-learn's solver fit the same objective to convergence and reached a test macro-F1 of 0.484, compared with 0.454 here. The results in Section V are therefore a slight underestimate of what this model can achieve, and the untuned optimizer settings are the most likely source of the difference.

# %% [markdown]
# ## V. Evaluation

# %% [markdown]
# ### A. Overall Results
#
# The threshold of each shelf was selected on the validation set, and every model was evaluated once on the test set. The resulting scores are listed below.

# %% cutoffs: one per shelf, chosen on validation; the default is logit 0 (probability 0.5)
cut = np.array([best_cutoff(Y[va][:, j], S_va[:, j]) for j in range(K)])
results['model, cutoff 0'] = scores(Y[te], S_te, 0.0)
results['model, cutoff per shelf'] = scores(Y[te], S_te, cut)
pd.DataFrame(results).T.round(3)

# %% [markdown]
# The description carries substantial information about a book's shelves, well beyond what a keyword rule extracts. The label prior scores zero, as expected. The keyword baseline reaches a macro-F1 of only 0.106, because reader-defined shelf names rarely occur in publisher descriptions; a book shelved as `contemporary` seldom describes itself as contemporary. The learned model reaches a macro-F1 of 0.454 and a micro-F1 of 0.480, more than four times the keyword baseline.
#
# Threshold selection contributes as much as the learned weights. At the default threshold of 0.5, the balanced class weights make the model over-predict: it assigns 1.59 shelves per book, whereas books carry about 0.44 on average, and its macro-F1 is 0.341. Thresholds selected on the validation set reduce the rate to 0.61 shelves per book and raise macro-F1 to 0.454. AUC is independent of the threshold and is therefore identical in both rows.
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
    'near_cutoff_ratio': near,
}, index=SHELVES)
per_shelf.sort_values('f1').round(3)

# %% [markdown]
# ### B. Per-Shelf Results
#
# The average conceals large differences between shelves. `manga` reaches an F1 of 0.812, with a precision of 0.86 and a recall of 0.77. `historical-romance` (0.685) and `m-m` (0.627) also perform well. At the other extreme, `contemporary` reaches an F1 of 0.244 with a precision of 0.19, and `contemporary-romance` reaches 0.279. These two shelves also have the lowest AUC (0.834 and 0.859). A low AUC indicates that the model fails to separate their books from the rest even in its ranking, so no choice of threshold can correct them.
#
# The per-shelf results thus give a qualified answer to the research question. For shelves with a distinctive vocabulary, the description predicts the shelf with useful precision and recall; for shelves such as `contemporary`, it largely does not. The next analysis examines the cause.

# %% does label noise or label count explain per-shelf F1?
rho_noise = spearmanr(per_shelf['near_cutoff_ratio'], per_shelf['f1'])[0]
rho_count = spearmanr(per_shelf['test_positives'], per_shelf['f1'])[0]
print(f'rank correlation with F1: near-cutoff ratio {rho_noise:+.2f}, test positives {rho_count:+.2f}')

# %% [markdown]
# ### C. Label Noise and Performance
#
# The hypothesis of Section III-D was tested by comparing each shelf's label-noise ratio with its test F1. Spearman's rank correlation was used because only 15 shelves are available, the ratio is heavily skewed (from 0.03 to 5.51), and the question is whether noisier shelves score lower, irrespective of the functional form of the relationship. The correlation is $\rho = -0.82$: the shelves on which readers disagree most are those the model predicts worst. The number of positive test books is only weakly related to F1 ($\rho = +0.23$), so the weak shelves are not weak merely because they are small. For example, `contemporary` and `historical` have a similar number of positive test books (1,833 and 2,307), yet their F1 scores are 0.244 and 0.525.
#
# These results suggest that the performance ceiling on these shelves is set mainly by the labels. If readers do not agree on whether a book is `contemporary`, a classifier trained to reproduce their votes inherits that disagreement. With 15 shelves, the correlation is suggestive rather than conclusive, and it does not exclude other explanations, such as the absence of a distinctive vocabulary discussed next.

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

# %% [markdown]
# ### E. Error Analysis
#
# In an earlier run of the same kind of model, our group examined ten of the most confident errors together with their descriptions and vote counts. Most were not failures to interpret the text. In some cases, readers had split their votes across spellings (`manga`, `mangas`, `manga-manhwa`), so the exact shelf remained below 10% even though the description matched it. In others, a sibling shelf such as `contemporary-romance` was positive while `contemporary` fell just below the threshold, or the label rested on two to five votes for a book whose description concerned something else. Three of the ten errors were cases in which the description did not mention the relevant fact, such as a manga volume whose stored description is a prose plot summary. Because the ten examples were selected for high confidence, they indicate which kinds of errors occur but not their frequency.

# %% [markdown]
# ### F. Limitations
#
# The labels are a single, unverified snapshot of reader votes, so measured performance is bounded by their quality. The evaluated books already have at least 10 genre votes and are more established than a typical new release. The model ignores correlations between shelves, and near-duplicate shelves are kept separate. In the earlier run, merging the three pairs of near-duplicate shelves into 12 labels raised the scores, but this changes the task to an easier one and does not indicate a better model. Finally, the optimizer settings were not tuned and the model had not converged (Section IV-F).

# %% [markdown]
# ## VI. Related Work
#
# The dataset was introduced by Wan and McAuley [1], who used Goodreads interactions to study recommendation when a user's actions form a monotonic chain of shelving, reading, rating, and recommending a book. Their models learn from users' past interactions with each item, which a newly released book lacks. The present work addresses the complementary, item-side problem of characterizing a new book from its own content. Wan et al. [2] used reviews from the same crawl for fine-grained spoiler detection, a classification task on review sentences. The romance file contains descriptions and no reviews, so a direct comparison with that task is not possible.
#
# Wang and Manning [3] showed that linear classifiers over word unigrams and bigrams, such as naive Bayes and linear support vector machines, are strong baselines for sentiment and topic classification, and that bigrams improve performance. Our results are consistent with theirs: a linear model over TF-IDF features performs well above trivial baselines, and in our group's earlier experiments, adding character n-grams to word features raised macro-F1 by 0.022. Wang and Manning also note that bag-of-words models cannot capture meaning arising from the composition of words. Our error analysis indicates a different limitation, namely that many errors originate in the labels or in information absent from the description, which no model of the text can recover.
#
# We are not aware of published results for shelf prediction on this dataset, so our F1 scores cannot be compared with prior work directly; the baselines of Section II-D serve as the points of comparison.

# %% [markdown]
# ## VII. Conclusion
#
# Given only its description, a TF-IDF model predicts a romance book's shelves well when the shelf corresponds to a distinctive vocabulary, as for `manga`, `historical-romance`, and `m-m`, with F1 between 0.63 and 0.81. It predicts poorly when readers apply the shelf inconsistently, as for `contemporary`, with F1 below 0.3. Across all 15 shelves, the model scores more than four times higher than a keyword baseline, and the strongest predictor of where it fails is reader disagreement about the label rather than the amount of training data. For a cold-start system, descriptions are therefore a reliable source of genre tags for some shelves and an unreliable one for others.

# %% [markdown]
# ## References
#
# [1] M. Wan and J. McAuley, "Item recommendation on monotonic behavior chains," in *Proc. 12th ACM Conf. Recommender Systems (RecSys)*, 2018, pp. 86-94, doi: 10.1145/3240323.3240369.
#
# [2] M. Wan, R. Misra, N. Nakashole, and J. McAuley, "Fine-grained spoiler detection from large-scale review corpora," in *Proc. 57th Annu. Meeting Assoc. Computational Linguistics (ACL)*, 2019, pp. 2605-2610. [Online]. Available: https://aclanthology.org/P19-1248/
#
# [3] S. Wang and C. D. Manning, "Baselines and bigrams: Simple, good sentiment and topic classification," in *Proc. 50th Annu. Meeting Assoc. Computational Linguistics (ACL), Vol. 2: Short Papers*, 2012, pp. 90-94. [Online]. Available: https://aclanthology.org/P12-2018/

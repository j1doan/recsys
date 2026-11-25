# %% [markdown]
# # Goodreads romance books: first look
# One JSON object per line. Book metadata only (no per-user reviews).

# %%
import gzip, json, os
from collections import Counter
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
PATH = os.environ["GOODREADS_PATH"]
LIMIT = None  # set e.g. 50_000 for a quick pass

# %% load
rows = []
with gzip.open(PATH, "rt", encoding="utf-8") as f:
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
empty = df.map(lambda v: v in ("", None) or (isinstance(v, list) and not v))
(1 - empty.mean()).sort_values().round(3)

# %% numeric columns are strings: convert
for c in ["average_rating", "ratings_count", "text_reviews_count", "num_pages",
          "publication_year", "publication_month", "publication_day"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df[["average_rating", "ratings_count", "text_reviews_count", "num_pages", "publication_year"]].describe()

# %% rating distribution
df["average_rating"].round(1).value_counts().sort_index()

# %% popularity: heavy tail
print((df["ratings_count"] == 0).mean(), "share with 0 ratings")
df["ratings_count"].quantile([.1, .25, .5, .75, .9, .99])

# %% rating vs popularity
df[["average_rating", "ratings_count", "text_reviews_count", "num_pages", "publication_year"]].corr(method="spearman")

# %% publication years
df["publication_year"].value_counts().sort_index().tail(40)

# %% categorical: language, format, ebook
for c in ["language_code", "format", "is_ebook", "country_code"]:
    print(df[c].replace("", "(empty)").value_counts().head(10), "\n")

# %% list columns: sizes
for c in ["authors", "series", "similar_books", "popular_shelves"]:
    print(c, df[c].map(len).describe()[["mean", "50%", "max"]].round(1).to_dict())

# %% top shelves (shelf name -> total count and number of books)
tot, nb = Counter(), Counter()
for shelves in df["popular_shelves"]:
    for s in shelves:
        tot[s["name"]] += int(s["count"])
        nb[s["name"]] += 1
pd.DataFrame({"total": tot, "books": nb}).sort_values("total", ascending=False).head(40)

# %% authors and series: books per author, books per series
a = Counter(x["author_id"] for au in df["authors"] for x in au)
print(len(a), "authors; books per author:", pd.Series(a).describe()[["mean", "50%", "max"]].round(1).to_dict())
s = Counter(x for se in df["series"] for x in se)
print(len(s), "series; books per series:", pd.Series(s).describe()[["mean", "50%", "max"]].round(1).to_dict())

# %% descriptions
d = df["description"].replace("", pd.NA).dropna()
print(len(d), "with description; words:", d.str.split().str.len().describe()[["mean", "50%", "max"]].round(0).to_dict())
d.iloc[:3].tolist()

# %% similar_books: how many point to books inside this file?
ids = set(df["book_id"])
sim = [x for l in df["similar_books"] for x in l]
print(len(sim), "links;", sum(x in ids for x in sim) / max(len(sim), 1), "point inside the file")


# %% [markdown]
# ## Task D: predict shelves from the description
# A "shelf" is a tag users put a book on (`popular_shelves`: name + how many users).
# Labels come from those counts by a rule (below). No hand labelling.
# Tune EXCLUDE, MIN_VOTES, MIN_FRAC, K after looking at the top-shelves and sweep cells.

# %% label rule parameters
import numpy as np

# shelves that are not about content (reading status, format, ownership, the genre itself)
EXCLUDE = {
    "to-read", "currently-reading", "read", "favorites", "favorite", "favourites", "owned", "own",
    "my-books", "books-i-own", "default", "library", "wish-list", "wishlist", "to-buy",
    "ebook", "ebooks", "e-book", "kindle", "audiobook", "audiobooks", "dnf", "abandoned",
    "did-not-finish", "re-read", "romance", "fiction",
}
MIN_VOTES = 10   # book needs at least this many content-shelf votes in total
MIN_FRAC = 0.10  # shelf counts as a label if it has this share of the book's content-shelf votes
K = 15           # number of shelves used as labels

# %% content shelves per book
def content_shelves(shelves):
    return {s["name"]: int(s["count"]) for s in shelves if s["name"] not in EXCLUDE}

df["sh"] = df["popular_shelves"].map(content_shelves)
df["sh_votes"] = df["sh"].map(lambda d: sum(d.values()))
has_text = df["description"] != ""

# %% cutoff sweep: books kept, and positives of the K-th most common shelf
rows = []
for votes in (5, 10, 20):
    keep = df[has_text & (df["sh_votes"] >= votes)]
    for frac in (0.05, 0.10, 0.20, 0.30):
        c = Counter(n for d, v in zip(keep["sh"], keep["sh_votes"]) for n, k in d.items() if k / v >= frac)
        pos = [p for _, p in c.most_common(K)]
        rows.append({"min_votes": votes, "min_frac": frac, "books": len(keep),
                     "positives_top1": pos[0], f"positives_top{K}": pos[-1]})
pd.DataFrame(rows)

# %% build labels with the chosen cutoff
elig = df[has_text & (df["sh_votes"] >= MIN_VOTES)].reset_index(drop=True)
c = Counter(n for d, v in zip(elig["sh"], elig["sh_votes"]) for n, k in d.items() if k / v >= MIN_FRAC)
SHELVES = [n for n, _ in c.most_common(K)]
Y = np.array([[d.get(s, 0) / v >= MIN_FRAC for s in SHELVES] for d, v in zip(elig["sh"], elig["sh_votes"])])
print(len(elig), "books;", Y.shape[1], "labels; mean labels per book:", Y.sum(1).mean().round(2))
pd.Series(Y.mean(0), index=SHELVES).sort_values(ascending=False).round(3)

# %% split by work (editions of one work stay on the same side)
from sklearn.model_selection import GroupShuffleSplit
tr, te = next(GroupShuffleSplit(test_size=0.2, random_state=0).split(elig, groups=elig["work_id"]))
print(len(tr), "train;", len(te), "test")

# %% baseline 1: label prior (score = share of train books with the label)
from sklearn.metrics import f1_score, roc_auc_score
prior = Y[tr].mean(0)
pred_prior = np.tile(prior >= 0.5, (len(te), 1))  # almost always predicts nothing
print("prior  micro-F1", round(f1_score(Y[te], pred_prior, average="micro", zero_division=0), 3),
      " AUC 0.5 by construction")

# %% baseline 2: TF-IDF + one-vs-rest logistic regression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

tfidf = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=5, stop_words="english", sublinear_tf=True)
Xtr = tfidf.fit_transform(elig["description"].iloc[tr])  # fit on train only
Xte = tfidf.transform(elig["description"].iloc[te])
clf = OneVsRestClassifier(LogisticRegression(solver="liblinear", class_weight="balanced")).fit(Xtr, Y[tr])
S = clf.decision_function(Xte)
P = S > 0
print("tfidf-LR  micro-F1", round(f1_score(Y[te], P, average="micro"), 3),
      " macro-F1", round(f1_score(Y[te], P, average="macro"), 3),
      " macro-AUC", round(roc_auc_score(Y[te], S, average="macro"), 3))

# %% per-shelf results
pd.DataFrame({"shelf": SHELVES, "test_positives": Y[te].sum(0),
              "F1": f1_score(Y[te], P, average=None).round(3),
              "AUC": [round(roc_auc_score(Y[te][:, j], S[:, j]), 3) for j in range(len(SHELVES))]})

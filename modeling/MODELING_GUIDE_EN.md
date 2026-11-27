# Goodreads Romance Modeling Guide

## What this assignment and our modeling section cover

Assignment 2 asks the group to build a prediction project using real data and
explain the task, data, models, evaluation, and related work. The group submits
a readable notebook exported as HTML and an approximately 20-minute presentation
video, followed by peer grading. Our group plan allocates approximately four
minutes to modeling. The original assignment PDF is one directory above this
repository in the local workspace; the group plan is in [PROJECT_PLAN.md](PROJECT_PLAN.md).

**Our task is to predict a book's reader-assigned shelf labels from its description
in the Goodreads romance subset.** A book can belong to several shelves, such as
`historical` and `historical-romance`, or to none of the selected shelves.
This is a multi-label text classification problem.

The input is description text only. The output is a 15-element binary vector,
with one prediction for each shelf. This metadata file does not contain a
per-user interaction matrix, so user-item matrix factorization and BPR do not
directly match the current task.

This guide covers the modeling contribution and its supporting experiments.
The group still needs to integrate the other sections into the shared workbook
and record the presentation.

## How the labels are constructed

Goodreads provides shelf names and counts in `popular_shelves`. We retain the
original repository's exclusion list, removing shelves such as `to-read`,
`owned`, `ebook`, `romance`, and `fiction`. A book is eligible when it has a
nonempty description and at least 10 remaining shelf votes.

For an eligible book, a shelf is positive when its count is at least 10% of
the book's total remaining shelf votes. For example, if a book has 100 remaining
votes, 14 votes for `historical` make that label positive; 9 votes do not.
The 15 shelves with the most positive training examples form the fixed label
vocabulary for validation and test.

These are the group's heuristic labels. The inherited exclusion list does not
remove every non-content shelf. **A zero means the shelf did not pass the rule,
not that a human confirmed the book lacks that content.** We retain books with
no positive target labels; forcing a label onto every book would change the task.

We also retain `manga`, which appeared among the original top 15 shelves. It
describes a medium rather than a romance subgenre, so the target should be
described as reader shelf usage rather than a uniform genre taxonomy.

The complete file contains 335,449 books, of which
304,837 have nonempty descriptions. The filtering rule
retains **299,789 books**. Each book has an average of
0.438 positive target labels, and
65.6% have none of the 15 selected labels.

## Training, validation, and test partitions

We split by `work_id`, keeping editions of the same work in one partition.
The outer split preserves the group's original test book IDs using seed 0.
An inner grouped split with seed 1 reserves validation data from the original
development partition.

| Partition | Books | Purpose |
| --- | --- | --- |
| Training | 191,692 | Fit the vocabulary, IDF, and classifier weights |
| Validation | 47,687 | Select the model, C, and thresholds |
| Test | 60,410 | Evaluate after model selection is frozen |

There is no work-group overlap between partitions. We select the label vocabulary
and fit both TF-IDF representations using training data only. The choice is
written to `reports/full/selection.json` before test scoring.

We do not refit the selected classifier on training plus validation data. Such a
refit could change its score scale and invalidate the validation-tuned thresholds.

The earlier plan's approximately 0.443 macro-F1 came from the original two-way
split, which used more books for training. Use the matched baselines from this
run to assess improvements; differences between the old and new workflows cannot
all be attributed to the model.

## Why these models are appropriate

### Word TF-IDF

TF-IDF represents each description as a sparse vector. Terms receive more weight
when they are frequent in a particular description but less common across books.
Word unigrams and bigrams retain individual words and phrases such as `vampire`,
`duke`, and `high school`.

The word representation uses at most 50,000 features, a minimum document frequency
of 5, English stop-word removal, sublinear term frequency, and L2 normalization.
The vocabulary and IDF are fitted on training descriptions only. Shelf fields,
ratings, author IDs, and titles are not model inputs. Work IDs are used for
splitting, and shelf counts are used to construct targets.

### One-vs-rest logistic regression

One-vs-rest trains one binary classifier for each of the 15 shelves. Each learns
a linear score from the same description vector. The sigmoid converts a score
to a probability-shaped output under the training objective. With
`class_weight='balanced'`, minority-class examples receive larger loss weights;
these outputs should not be treated as calibrated population probabilities.

Logistic regression works efficiently with sparse text and exposes interpretable
feature weights. Its limitations include limited understanding of long-range
meaning and the absence of an explicit model of dependencies between labels.

### Linear SVM

Linear SVM also learns one linear function per shelf, using squared hinge loss
to encourage separation between positive and negative examples. It is suitable
for high-dimensional sparse text. Its outputs are decision scores rather than
probabilities. We use `LinearSVC`, which avoids constructing a large nonlinear
kernel matrix.

### Training objectives and model selection

For label j, let `s_ij = w_j^T x_i + b_j`, `z_ij = 2*y_ij - 1`, and `a_ij` be
the training class weight. With the default intercept scaling of 1, liblinear
also regularizes the intercept; `w_tilde_j` includes both the coefficients and
the intercept. The objectives are:

```text
LR:  0.5 * ||w_tilde_j||^2 + C * sum_i a_ij * log(1 + exp(-z_ij * s_ij))
SVM: 0.5 * ||w_tilde_j||^2 + C * sum_i a_ij * max(0, 1 - z_ij * s_ij)^2
```

For balanced weighting, `a_ij = n / (2 * n_class)` within each training binary
problem. Both models use L2 regularization. A larger C weakens regularization
relative to fitting the training data.

**Training minimizes these regularized losses. Model and threshold selection use
validation F1.** The classifier does not directly optimize F1 during fitting.

### Adding character features

The richer representation adds at most 60,000 within-word character n-grams of
length 3 to 5 using `char_wb`. Word and character blocks are L2-normalized
separately and concatenated with fixed weights of 1.0 and 0.5, respectively.
Character fragments can capture morphology, spellings, names, and non-English
text, while increasing memory and computation.

The largest training feature matrix has 191,692 rows
and 110,000 columns. A dense float64 version would
require approximately
157.1 GiB.
The implementation therefore retains sparse matrices and limits parallel
classifiers. Prediction costs approximately `O(K * nnz(X))`, where K is the
number of labels and `nnz(X)` is the number of nonzero input features.

### Candidate comparison and measured efficiency

We declared eight configurations: balanced word LR with C=0.25, 1, or 4;
unweighted word LR with C=1; balanced word SVM with C=0.25 or 1; and balanced
word-plus-character LR and SVM with C=1.

Selection uses validation macro-F1 with per-label thresholds. Ties are broken
by validation micro-F1 and then candidate name. The selected candidate is
**`wordchar_lr_C1`**, with validation macro-F1
**0.4873**.

| candidate | macro_f1 | train_seconds | feature_seconds | n_features |
| --- | --- | --- | --- | --- |
| word_lr_balanced_C1 | 0.465 | 16.774 | 31.431 | 50000 |
| word_lr_balanced_C025 | 0.457 | 12.167 | 31.431 | 50000 |
| word_lr_balanced_C4 | 0.458 | 21.629 | 31.431 | 50000 |
| word_lr_unweighted_C1 | 0.468 | 10.543 | 31.431 | 50000 |
| word_svm_C025 | 0.453 | 25.394 | 31.431 | 50000 |
| word_svm_C1 | 0.426 | 78.085 | 31.431 | 50000 |
| wordchar_lr_C1 | 0.487 | 258.459 | 149.603 | 110000 |
| wordchar_svm_C1 | 0.456 | 924.317 | 149.603 | 110000 |

`macro_f1` in this table is measured on validation data. `train_seconds` includes
training and one validation scoring pass. `feature_seconds` is the shared cost
of fitting training features and transforming validation text for a representation;
it should not be charged separately to every candidate using that representation.
Runtime depends on the machine and parallelism. These results support the
selected configuration within the tested grid, rather than a claim that one
classifier is always better for text classification.

## What threshold tuning does

The model produces a score before making a binary decision. The original
repository predicted a label when its score was greater than zero. This
experiment consistently predicts a label when `score >= threshold`, including
when scores are tied.

We compare three policies:

1. **Default:** use zero for every label.
2. **Global:** choose one threshold maximizing validation micro-F1 across all
   book-label pairs.
3. **Per-label:** choose one threshold per shelf maximizing its validation F1.
   Independent per-label optimization corresponds to the macro-F1 objective.

The exact search sorts validation scores and uses cumulative true positives to
compute F1 at every distinct score boundary. All examples with the same score
are processed together. Ties in F1 keep the higher threshold; a shelf with no
validation positives predicts none. The cost is `O(n log n)` per label.

Thresholds change the binary predictions, but not the fitted scores or their
ranking. ROC-AUC and average precision therefore remain the same across
threshold policies for a given model. We do not tune thresholds on test data
or force every book to receive a top-1 label.

## Full-data results and interpretation

The table below evaluates the original 15-label task on the held-out test set.
The prior baseline thresholds training prevalences at 0.5. The keyword baseline
uses literal shelf-name matches after simple punctuation and space normalization.

| candidate | policy | micro_f1 | macro_f1 | macro_auc | macro_ap |
| --- | --- | --- | --- | --- | --- |
| word_lr_balanced_C1 | default | 0.4661 | 0.4427 | 0.9286 | 0.4151 |
| word_lr_balanced_C1 | global | 0.4997 | 0.4651 | 0.9286 | 0.4151 |
| word_lr_balanced_C1 | per_label | 0.4843 | 0.4626 | 0.9286 | 0.4151 |
| wordchar_lr_C1 | default | 0.4945 | 0.4682 | 0.9344 | 0.4399 |
| wordchar_lr_C1 | global | 0.5242 | 0.4841 | 0.9344 | 0.4399 |
| wordchar_lr_C1 | per_label | 0.5047 | 0.4843 | 0.9344 | 0.4399 |
| label_prior | probability_0.5 | 0.0000 | 0.0000 | 0.5000 | 0.0292 |
| literal_keyword | literal_match | 0.1077 | 0.1062 | 0.5360 | 0.0515 |

On the same training partition, tuning thresholds increases word LR macro-F1
from 0.4427 to 0.4626. The selected model achieves
**micro-F1 0.5047, macro-F1 0.4843, and
macro-AUC 0.9344**.

Relative to default-threshold word LR, the macro-F1 gain is
0.0417, or approximately
4.17 percentage points and
9.4% relative improvement.
Using 300 paired bootstrap replicates that resample test
work groups, the 95% interval for that gain is
[0.0370, 0.0469]. This conditions on the fitted models and
captures test-work sampling variation, not variation from retraining or model
selection.

There is also an operating-point trade-off. For the selected classifier, the
global threshold achieves test micro-F1 0.5242, higher
than the per-label policy, with macro-F1 0.4841.
The primary configuration follows the validation macro-F1 criterion declared
before test evaluation. Per-label thresholds are not better for every metric.

Micro-F1 pools true positives, false positives, and false negatives across labels,
so frequent shelves have more influence. Macro-F1 averages each shelf's F1
equally. ROC-AUC evaluates ranking, rather than the final label decisions.
Average precision provides a useful additional view when positives are rare.
Accuracy is not a primary metric because negative targets dominate.

## How similar shelf labels are handled

The primary task retains the original 15 labels for continuity with the group's
analysis. Related names such as `historical` and `historical-romance` do not
necessarily mean the same thing.

We separately study a 12-label target by merging these pairs:

| Original shelves | Merged label |
| --- | --- |
| historical, historical-romance | historical |
| contemporary, contemporary-romance | contemporary |
| paranormal, paranormal-romance | paranormal |

**Construct the original binary targets first, then combine each pair using
logical OR.** We do not add shelf counts: the same reader could use both shelves,
so adding votes would introduce an extra overlap assumption. OR also avoids
creating a positive simply because two individually sub-threshold counts add
up to 10%.

The merged experiment uses the selected architecture, fits a new classifier on
training data, and tunes new thresholds on validation. Its test micro-F1 is
0.5320 and macro-F1 is 0.5048.
These scores belong to a broader target and cannot be presented as a direct
gain on the original 15-label task. Training-label overlap statistics are in
`reports/full/alias_analysis.csv`.

## Code walkthrough

The [executed notebook](../modeling.ipynb) and [HTML export](../modeling.html)
display the actual functions and training loop, with explanatory text and
recorded outputs. The expensive training was run through `python -m modeling.run`;
the explanatory notebook reads those results rather than retraining eight models
whenever it is opened.

| Code | What to explain |
| --- | --- |
| `core.py`: `read_books`, `select_labels`, `make_targets` | Eligibility, training-only label selection, and binary targets |
| `core.py`: `split_books` | Keeping editions together and preserving the original test set |
| `core.py`: `make_features` | Sparse TF-IDF, word/character blocks, and fixed feature weights |
| `run.py`: `classifier`, `fit_checked` | OvR classifiers, L2 regularization, class weights, convergence checks, and controlled parallelism |
| `core.py`: `best_f1_threshold`, `tune_thresholds` | Exact threshold search, tied scores, and validation-only tuning |
| `run.py`: `main` | Candidate comparison, freezing selection, and final test evaluation |
| `core.py`: `merge_targets` | OR-based merging as a separate target |
| `predict.py`: `predict` | Applying the saved vectorizer, classifier, and thresholds to a new description |

Positive feature weights show associations that raise a shelf's score. They
are not causal explanations. Inspect `reports/full/top_features.csv` for examples.
The notebook includes a synthetic description for an inference demonstration.
`error_review_queue.csv` supplies candidate false-positive and false-negative
cases for human reading; it does not contain human-verified explanations.

## Limitations and validity checks

Although work groups do not overlap, the test set contains
279
books whose lowercased, whitespace-normalized descriptions also occur in training
or validation. A diagnostic excluding these books retains
60,131 test examples and gives selected-model macro-F1
0.4842. This is a diagnostic subset, not a benchmark
retrained with all duplicate descriptions grouped together. Near duplicates
have not been completely removed.

Other limitations are:

- Shelf votes are noisy proxy labels, and the 10% / 10-vote rule has not been
  validated against human genre annotations.
- Authors and series can occur across partitions. The test does not establish
  generalization to new authors, future years, or other platforms.
- Bag-of-ngrams features have limited semantic understanding, and OvR does not
  explicitly model label relationships.
- Per-label thresholds may overfit smaller validation labels. Results come from
  one grouped partition and a finite candidate grid.

## Suggested four-minute presentation

| Time | Content and code to show |
| --- | --- |
| 0:00-0:40 | Description input, 15 binary outputs, and one vote-share labeling example |
| 0:40-1:35 | TF-IDF, independent LR classifiers, class imbalance, and regularization; show `make_features` and `classifier` |
| 1:35-2:30 | Default versus tuned thresholds; show `best_f1_threshold` and explain the validation partition |
| 2:30-3:20 | Validation comparison and selected model; explain the runtime and memory trade-offs |
| 3:20-4:00 | Separate label-merging experiment, one result table, and the main validity limitations |

Before presenting, run a prediction, inspect the leading feature weights for one
shelf, and read an error example. Explain the choices in your own words and keep
the claims consistent with the code and recorded results.

## Files and reproduction

| File or directory, relative to the repository root | Purpose |
| --- | --- |
| `modeling.ipynb`, `modeling.html` | Executed English modeling section with embedded outputs and figures |
| `modeling/core.py` | Data, targets, splitting, features, thresholds, and metrics |
| `modeling/run.py` | Full experiment, selection freeze, merged target, and bootstrap |
| `modeling/predict.py` | Prediction using a saved fitted model |
| `artifacts/full/selected_model.joblib` | Local selected classifier, vectorizer, labels, and thresholds |
| `artifacts/full/test_predictions.npz` | Test predictions and explicit book/label ordering |
| `reports/full/` | CSV tables, JSON decisions, split IDs, and PNG figures |
| `reports/full/verification.json` | Independent checks of exported outputs, split isolation, and metrics |
| `modeling/HANDOFF.md` | Instructions for merging with the shared workbook |

Use Python 3.11+ and run these commands from the repository root:

```bash
python -m pip install -r requirements-modeling.txt
python -m scripts.download_romance
python -m unittest discover -s tests -v
python -m modeling.run --jobs 4
python -m scripts.build_modeling_report
python -m scripts.verify_modeling
python -m modeling.predict "A young woman falls in love with a vampire."
```

The recorded full experiment took approximately 33.2
minutes. On the original machine, `/home/cauthor/anaconda3/bin/python3` had the
dependencies; `/usr/bin/python3` did not. Other machines should use their own
configured Python environment. Reduce `--jobs` on machines with less available
RAM. Read [README.md](../README.md) for the setup details.

The original `code.py` is unchanged. Its name shadows Python's standard-library
`code` module and can interfere with IPython startup. The report builder handles
this collision; for interactive use, launch `PYTHONSAFEPATH=1 jupyter lab`.

Raw data and fitted artifacts are excluded from Git. The HTML and saved notebook
outputs can be read without retraining. Executing the notebook or prediction CLI
requires the local fitted artifacts, obtained by running the pipeline or sharing
those artifacts separately. Only load trusted local joblib files.

After integration, execute the combined group notebook from a fresh kernel and
export the final `workbook.html`. The group must also record the presentation
and supply its actual video URL.

## References

- [Official Goodreads dataset page](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html): dataset context and academic-use conditions.
- [scikit-learn text feature extraction](https://scikit-learn.org/1.6/modules/feature_extraction.html#text-feature-extraction).
- [scikit-learn logistic regression](https://scikit-learn.org/1.6/modules/linear_model.html#logistic-regression).
- [scikit-learn support vector machines](https://scikit-learn.org/1.6/modules/svm.html#classification).
- [scikit-learn threshold tuning](https://scikit-learn.org/1.6/modules/classification_threshold.html).
- Wan and McAuley (2018), [Item Recommendation on Monotonic Behavior Chains](https://cseweb.ucsd.edu/~jmcauley/pdfs/recsys18b.pdf).
- Wan et al. (2019), [Fine-Grained Spoiler Detection from Large-Scale Review Corpora](https://aclanthology.org/P19-1248/).

The last two papers provide dataset context. They study user-behavior-chain
recommendation and review spoiler detection, respectively. Their task scores
are not directly comparable with our description-to-shelf F1.

# Evaluation

This section uses the saved test results for the original 15-shelf task. It does not refit models or change the split. The primary configuration is word-plus-character TF-IDF with one-vs-rest logistic regression, `C=1`, balanced class weights, and per-label thresholds chosen on validation macro-F1 (`wordchar_lr_C1`, policy `per_label`). Decision scores are not probabilities. F1 uses the binary predictions; ROC-AUC and average precision use the scores.

The 12-label OR-merge is a different target. Its test micro-F1 is 0.5320 and its macro-F1 is 0.5048. Those numbers stay out of the 15-label comparison below.

## Why these metrics

A book may carry several of the 15 shelves, or none. Of the 299,789 eligible books, 65.6% have no positive label among the 15. On the test set the most common positive shelf, `m-m`, covers 3,892 of 60,410 books. Predicting every shelf negative would be accurate on the great majority of book-shelf pairs and would score a micro-F1 and macro-F1 of 0, which is what the training-prior baseline does when the decision rule is "predict positive only if the training rate is at least 0.5." No shelf is that common, so accuracy would hide a useless classifier. It is not the primary metric.

Micro-F1 pools every book-shelf decision, so the frequent shelves dominate the headline number. Macro-F1 gives each shelf the same weight, which is why a shelf with F1 0.23 pulls the average down even if `m-m` looks strong. Model selection used validation macro-F1 for that reason. ROC-AUC measures ranking and does not change when the threshold changes. The primary model and its global-threshold variant therefore share macro-AUC 0.9344. Average precision is lower, 0.4399, because a high rank is harder to turn into precise predictions when positives are rare. AUC says the scores order books well. AP and F1 say the operating point still admits many false positives.

## Aggregate comparison

Validation, before any test number was used for selection, ranked `wordchar_lr_C1` with per-label thresholds first on macro-F1 (0.4873). The same candidate's global threshold had higher validation micro-F1 (0.5211) and lower macro-F1 (0.4772). The test table shows the same pattern. The global threshold is reported as a second operating point. It was not swapped in after seeing the test set.

| Configuration | Micro-F1 | Macro-F1 | Macro-AUC | Macro-AP |
| --- | ---: | ---: | ---: | ---: |
| Label prior, predict positive only above 0.5 | 0.0000 | 0.0000 | 0.5000 | 0.0292 |
| Literal keyword match | 0.1077 | 0.1062 | 0.5360 | 0.0515 |
| Word LR, default threshold | 0.4661 | 0.4427 | 0.9286 | 0.4151 |
| Word LR, per-label thresholds | 0.4843 | 0.4626 | 0.9286 | 0.4151 |
| Word + character LR, per-label thresholds (primary) | 0.5047 | 0.4843 | 0.9344 | 0.4399 |
| Word + character LR, global threshold | 0.5242 | 0.4841 | 0.9344 | 0.4399 |

Source: `reports/full/test_results.csv`. The keyword baseline from the original plan is the second row. The primary model more than quadruples its macro-F1. Against the matched word-only logistic regression with its own tuned thresholds, the gain is 0.020 micro-F1 and 0.022 macro-F1. A paired bootstrap over test works, conditional on the already fitted models, puts that macro-F1 gap between 0.018 and 0.026 (`reports/full/bootstrap.json`, 300 resamples). Dropping the 279 test books whose exact normalized description also appears in training or validation leaves macro-F1 at 0.4842. That check is a diagnostic. It does not remove shared authors, series, or near-duplicate paraphrases.

The primary operating point predicts 0.61 labels per test book, with micro-precision 0.433 and micro-recall 0.605. The global threshold predicts fewer labels (0.50 per book) and trades recall for precision, which is why its micro-F1 is higher and its macro-F1 is essentially tied. The presentation should say which goal the threshold serves. Per-label thresholds were chosen to raise validation macro-F1. They are the primary result because that choice was fixed before test evaluation.

## Per-shelf view

Primary-model test F1, from `reports/full/per_label_wordchar_lr_C1.csv`:

| Shelf | Test positives | Precision | Recall | F1 | ROC-AUC | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| manga | 964 | 0.883 | 0.878 | 0.880 | 0.997 | 0.917 |
| historical-romance | 3,741 | 0.633 | 0.775 | 0.697 | 0.974 | 0.697 |
| m-m | 3,892 | 0.586 | 0.820 | 0.683 | 0.973 | 0.653 |
| bdsm | 926 | 0.565 | 0.566 | 0.566 | 0.962 | 0.529 |
| historical | 2,307 | 0.439 | 0.707 | 0.541 | 0.959 | 0.500 |
| paranormal | 2,596 | 0.389 | 0.621 | 0.478 | 0.930 | 0.414 |
| paranormal-romance | 1,494 | 0.395 | 0.603 | 0.477 | 0.945 | 0.399 |
| chick-lit | 1,720 | 0.438 | 0.501 | 0.467 | 0.929 | 0.434 |
| young-adult | 1,304 | 0.432 | 0.498 | 0.462 | 0.952 | 0.415 |
| romantic-suspense | 876 | 0.387 | 0.513 | 0.441 | 0.953 | 0.368 |
| new-adult | 1,549 | 0.429 | 0.365 | 0.394 | 0.923 | 0.350 |
| fantasy | 625 | 0.357 | 0.374 | 0.366 | 0.932 | 0.302 |
| erotica | 983 | 0.275 | 0.331 | 0.300 | 0.914 | 0.234 |
| contemporary-romance | 1,635 | 0.244 | 0.331 | 0.281 | 0.849 | 0.211 |
| contemporary | 1,833 | 0.158 | 0.425 | 0.231 | 0.827 | 0.174 |

`manga` is the strong shelf because the description often names the medium or uses Japanese name fragments, and those features receive large positive weight (`reports/full/top_features.csv`). `historical-romance` and `m-m` are strong for a different reason: the blurbs repeatedly use a small vocabulary (`duke`, `earl`, `lady`, `London`; `gay`, `his`, `he`) and the negative weights are the opposite pronouns or modern words. Precision on `m-m` is 0.586, so a large share of predicted `m-m` books still fail the 10% rule.

`contemporary` and `contemporary-romance` are the weak pair. Their ROC-AUC values, 0.827 and 0.849, are the only ones below 0.90, so the weakness is in the ranking, not only in the threshold. The largest positive weights are series and place names (`lucky harbor`, `virgin river`, `idaho`), which fire for one imprint and miss standalone category romances. `contemporary` also uses a negative threshold, −0.437, and still reaches only 0.158 precision. Lowering the bar adds predictions without making the shelf separable. The sibling shelves `historical` / `historical-romance` and `paranormal` / `paranormal-romance` show the same split of one reader idea across two names. That is why the merged 12-label experiment exists, and why its higher F1 cannot be compared with this table as if the task were unchanged.

## Annotated cases

`reports/full/error_review_queue.csv` holds up to three confidence-ranked false positives and three confidence-ranked false negatives per shelf. Confidence here means a decision score far above the threshold for a false positive, or far below it for a false negative. The file is not a random sample of errors. The notes below read 10 of those rows. Five are the top-ranked case for that shelf and error type. The other five stay inside the queue and were chosen so the quote would be one English book whose blurb illustrates a different failure: a single novel rather than an anthology, or a blurb that states the relationship or the plot directly. Shelf fractions were recomputed from the official romance file with the project's exclusion list. The file's sha256 matches `reports/full/data_audit.json`. Full notes are in `reports/manual_error_analysis.csv`.

**Exact shelf name versus the words in the blurb.** *Mishounen Produce, Vol. 02* (13330144) is a manga false positive, score 7.71 against a threshold of 0.70. The blurb is a high-school story and says Kako models her friend on "a character in a manga she likes." Readers used `mangas`, `manga-manhwa`, and `shoujo`. The exact name `manga` has 0 of 83 content votes, so the rule marks the book negative. *Boy Meets Boy Meets Boy* (29497723) is the same pattern for `m-m`. The blurb says "male/male/male menages," while the votes are split across `m-m` (15), `menage` (12), `mmm` (8), `mm` (7), and `m-m-m` (6). The exact shelf is 7.2% of 209 content votes. The model agrees with the text. The negative label is the one-name rule.

**The 10% cutoff on an otherwise clear romance.** *Confessions of a Viscount* (87629) is a historical-romance false positive, score 7.30 against 0.83. The blurb names a viscount, the ton, and a London rooftop. `historical-romance` is 9 of 95 content votes, 9.5%. *Christmas in Lucky Harbor* (10841305) is a contemporary false positive for the sibling of a shelf it does clear: `contemporary` is 29 of 352 votes (8.2%) and `contemporary-romance` is 42 (11.9%). `lucky harbor` is a top positive feature for both. The false positive is the alias the merge experiment was built to measure.

**A real shelf that the description does not announce.** *Silver* (6670295) is a manga false negative, score −3.74, with `manga` at 28 of 77 content votes (36%). The stored text is an English aristocratic revenge plot and never says the book is a manga. *After Hours* (1445610) is an `m-m` false negative. Grace overhears a neighbor and the blurb asks whom he is speaking to. It never says that partner is a man. `m-m` is 7 of 46 votes. `her` and `she` are the largest negative features for that shelf, so the score follows the pronouns on the page. *The Notebook* (6950627) is a chick-lit false negative with a large count that is still a narrow share: 1,454 of 12,548 content votes, 11.6%. The blurb follows Noah, "a rural Southerner recently returned form the Second World War." It has none of the comic, Bridget-Jones vocabulary that dominates the positive weights.

**A positive label that the blurb contradicts.** *Dustin's Novel* (29909341) is a contemporary false negative, score −6.73. The positive label is 2 of 15 content votes. The description calls the book the last volume of a time-travel trilogy "with a touch of sci-fi." *Emma: The Wild and Wanton Edition* (17433645) clears historical-romance by 5 of 49 votes (10.2%). The blurb sells an erotic rewrite of Austen, and `sexy` is a large negative weight for historical-romance. *Lelang Pengantin* (2674675) has an Indonesian title and an English Harlequin blurb about a Montana rancher. `contemporary-romance` is 4 of 22 votes. The miss is not a failure to read Indonesian. The shelf's largest positive weights are series names this standalone book does not contain.

## What the errors support

Three limits follow from these cases and should be stated with the metrics.

Shelf votes are a noisy proxy for genre. Several "errors" are books whose descriptions match the shelf while the vote share falls just short of 10%, or whose votes are spread across spellings. Treating every false positive as a model hallucination would overstate the failure.

The description often omits the fact the shelf records. Manga adaptations are stored as the prose story. An `m-m` shelf can sit on a blurb that uses only `she`. A linear model of that text will miss the label, and a higher-capacity model of the same blurb has the same missing evidence.

On the training books, the alias pairs are real overlaps rather than two names for one set of books. `historical` is positive together with `historical-romance` on 6,467 books, and alone on only 1,032. `contemporary` and `contemporary-romance` co-occur on 2,272 training books, while 3,143 have only `contemporary-romance` and 2,859 have only `contemporary`. The merged 12-label score therefore changes the target, especially for the contemporary pair. These counts were recomputed from `data/goodreads_books_romance.json.gz` with the saved training book IDs.

The weak contemporary shelves are a modeling limit. Their AUC is lower, their positive features collapse onto a few series, and the per-label threshold cannot repair a ranking that mixes sibling shelf names. The manual sample is too small, and too deliberately chosen from the tails of the score, to estimate how often each failure mode occurs. It is large enough to show that the aggregate F1 mixes cutoff artifacts, alias splits, missing evidence, and genuine ranking errors.

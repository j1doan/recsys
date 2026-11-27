"""Build workbook.ipynb from the saved reports and the EDA figures."""
import base64
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output

ROOT = Path(__file__).resolve().parents[1]


def png(path):
    data = base64.b64encode(Path(path).read_bytes()).decode('ascii')
    return new_output(
        output_type='display_data',
        data={'image/png': data, 'text/plain': Path(path).name},
    )


def stream(text):
    if not text.endswith('\n'):
        text += '\n'
    return new_output(output_type='stream', name='stdout', text=text)


def code(source, text='', images=()):
    outputs = []
    if text:
        outputs.append(stream(text))
    outputs.extend(png(ROOT / p) for p in images)
    cell = new_code_cell(source)
    cell['outputs'] = outputs
    cell['execution_count'] = 1
    return cell


def main():
    cells = []
    cells.append(new_markdown_cell("""# Predicting Goodreads romance shelves from the book description

This is the written report for CSE 258 Assignment 2. It is meant to be read while the presentation plays. The modeling code has already been run on the full romance metadata. This notebook does not refit those models. It records the task, the data, the selected model, the evaluation, and the related work.

The local file used for the data checks is `data/goodreads_books_romance.json.gz`. Its sha256 is `19006689989a90d1b83ede16c6ef2338f65f8f432956515d7c6489fe7ac94d01`, the same digest stored in `reports/full/data_audit.json`. Test metrics are copied from `reports/full/test_results.csv` and the per-shelf table.
"""))

    cells.append(new_markdown_cell("""## 1. Predictive task

**Task.** Given only a book's description, predict which of 15 reader shelves apply. A book may have several of these shelves, or none. The shelves are reader tags such as `m-m`, `historical-romance`, and `manga`, not a hand-built genre ontology.

**Inputs and outputs.** The input is the description string. The output is 15 binary labels. A shelf is positive when it accounts for at least 10% of that book's content-shelf votes, after status and format shelves such as `to-read`, `ebook`, and the generic tags `romance` and `fiction` are removed. A book also needs at least 10 remaining shelf votes. The 15 names are chosen from the training books only, in the order stored in `reports/full/selection.json`.

**How the model is judged.** The split keeps every edition of a work on the same side: 191,692 training books, 47,687 validation books, and 60,410 test books. Validation chooses the model and the thresholds. The test set is used once, after that choice. The reported scores are micro-F1, macro-F1, macro ROC-AUC, and average precision. Accuracy is not the headline metric, because 65.6% of eligible books have none of the 15 labels. A classifier that predicts all negatives would look accurate and would score zero F1.

**Baselines.** Two baselines use no learned weights. The label prior predicts a shelf only when at least half of training books have it. None do, so its F1 is zero. The keyword baseline predicts a shelf when the shelf name appears in the description. The learned comparison is a word-only logistic regression with the same training split. A separate 12-label run merges three alias pairs. That run is a different target and is not mixed into the 15-label table.

**Whether a prediction is valid.** Shelf votes are a noisy proxy, not a checked genre label. Validity is checked in three ways: the test books were not used to pick the model, editions of one work are not split across train and test, and ten high-confidence mistakes were read against the publisher description and the raw shelf counts (`reports/manual_error_analysis.csv`).
"""))

    cells.append(new_markdown_cell("""## 2. Data, collection, and cleaning

The file is the romance subset of the UCSD Goodreads crawl, collected in late 2017 from public shelves. User IDs in the broader release are anonymized. The authors ask that the data be used for academic work and not redistributed or used commercially. The romance file has book metadata only. This project does not use the interaction file or the review file, so there is no user history to feed a recommender.

`scripts/eda_romance.py` recounts the file. The counts match the modeling audit.
"""))

    cells.append(code(
        "import json\n"
        "stats = json.load(open('reports/eda/eda_stats.json'))\n"
        "print('raw', stats['raw_books'])\n"
        "print('with description', stats['nonempty_descriptions'])\n"
        "print('eligible', stats['eligible_books'])\n"
        "print('share with none of the 15 labels', round(stats['zero_label_fraction'], 4))\n"
        "print('ratings median/mean/max',\n"
        "      stats['ratings_count']['quantiles']['0.5'],\n"
        "      round(stats['ratings_count']['mean'], 1),\n"
        "      stats['ratings_count']['max'])\n"
        "print('similar_books inside this file', round(stats['similar_books']['inside_share'], 4))",
        "raw 335449\n"
        "with description 304837\n"
        "eligible 299789\n"
        "share with none of the 15 labels 0.6560\n"
        "ratings median/mean/max 52.0 448.0 2078406.0\n"
        "similar_books inside this file 0.8384\n",
    ))

    cells.append(new_markdown_cell("""Of 335,449 raw books, 30,612 have an empty description and are dropped. Another 5,048 have a description but fewer than 10 content-shelf votes, which leaves 299,789 eligible books. The development books in the original two-way plan are now split into training and validation. The test book IDs from that plan are unchanged.

Eligible descriptions have a median of 147 words. The body of the distribution sits well below 400 words, with a long tail of catalog text and box-set blurbs.

The 15 shelves are uneven. `m-m` and `historical-romance` each cover about 19,000 eligible books. `fantasy` covers 3,105. `manga` is a medium that readers shelved inside a romance dump, which is why its text is easy to recognize and why it is a strange romance label.

196,649 eligible books, 65.6%, have zero positive labels under the 10% rule. 75,513 have one, 26,982 have two, and only 645 have three or more. Macro-F1 has to care about the rare shelves. Micro-F1 will be pulled toward the common ones, and toward the huge negative class.

Rating counts are a heavy tail: median 52, mean 448, maximum about 2.08 million. 1,499 books, 0.45%, have zero ratings. Publication year is missing on 101,800 books. The recorded extremes are year 1 (one book) and year 65535 (two books); 83 books are dated before 1800. Language is `eng` on 160,125 books and empty on 104,396. These fields are used only to describe the file. The classifier never sees year, rating, language, or `similar_books`. About 83.8% of `similar_books` links point at another row in this file, so a link-prediction task would have been possible. It is not this task.
"""))

    cells.append(code(
        "from IPython.display import Image, display\n"
        "display(Image('reports/eda/description_lengths.png'))\n"
        "display(Image('reports/eda/label_support.png'))\n"
        "display(Image('reports/eda/labels_per_book.png'))\n"
        "display(Image('reports/eda/ratings_tail.png'))\n"
        "display(Image('reports/eda/publication_years.png'))",
        '',
        [
            'reports/eda/description_lengths.png',
            'reports/eda/label_support.png',
            'reports/eda/labels_per_book.png',
            'reports/eda/ratings_tail.png',
            'reports/eda/publication_years.png',
        ],
    ))

    cells.append(new_markdown_cell("""## 3. Modeling

Each description is a bag of word and character n-grams. Shelves do not enter the vectorizer. They are only the targets. `work_id` is used only to keep editions together. The model is one logistic regression per shelf (`OneVsRestClassifier`), with `C=1` and balanced class weights. Balancing matters because every shelf is rare. The decision score is a linear function of TF-IDF features. It is not a calibrated probability.

Eight configurations were fit: word versus word-plus-character TF-IDF, two values of `C`, class weighting on or off, and logistic regression versus a linear SVM where that comparison was run. The full grid and timings are in `reports/full/validation_search.csv` and in `modeling/run.py`. Selection used validation macro-F1 with per-label thresholds. The winner is `wordchar_lr_C1`. Its validation macro-F1 is 0.487. The same model's global threshold scores higher validation micro-F1 and lower macro-F1, so the operating point depends on which F1 the project promised to optimize. Thresholds are chosen on validation only. The model is not refit on training plus validation, because a refit would move the scores relative to those thresholds.

Character n-grams help shelves whose cue is a pronoun or a name fragment (`her`, `his`, pieces of Japanese names). Word bigrams help series titles such as `lucky harbor`. A linear model is the right comparison point for this course: it is fast on 191,692 documents, the weights can be read, and it is a standard text-classification baseline. Its limit is that it cannot tell "a manga adaptation whose blurb retells the prose novel" from a prose novel, and it treats `contemporary` and `contemporary-romance` as unrelated outputs even when the blurbs share the same series name.

The original `code.py` exploration used a two-way split and a default threshold. Those older F1 numbers are not the ones in the test table below. The pipeline here keeps that file unchanged and adds a validation cut.
"""))

    cells.append(code(
        "from IPython.display import Image, display\n"
        "display(Image('reports/full/figures/validation_search.png'))\n"
        "display(Image('reports/full/figures/test_ablation.png'))\n"
        "display(Image('reports/full/figures/per_shelf_f1.png'))\n"
        "display(Image('reports/full/figures/modeling_pipeline.png'))",
        '',
        [
            'reports/full/figures/validation_search.png',
            'reports/full/figures/test_ablation.png',
            'reports/full/figures/per_shelf_f1.png',
            'reports/full/figures/modeling_pipeline.png',
        ],
    ))

    cells.append(new_markdown_cell("""## 4. Evaluation

### Why these metrics

Micro-F1 pools every book-shelf decision, so `m-m` and `historical-romance` dominate it. Macro-F1 averages the 15 shelves, which is why `contemporary` at F1 0.23 pulls the headline down. Model selection therefore used validation macro-F1. ROC-AUC measures ranking and does not change when the threshold moves: the primary model and its global-threshold variant share macro-AUC 0.9344. Average precision is 0.440. The gap between AUC and average precision is the sparse-positive problem. The scores order books well, and a high rank is still full of false positives. The prior baseline makes the accuracy point concrete. Predicting a label only when its training rate is at least one half yields F1 0, because no shelf is that common.

### Test comparison

The rows below are the matched test results. The primary row is word-plus-character logistic regression with per-label thresholds. The global-threshold row is a second operating point. It was not selected after seeing the test set. On validation it already had higher micro-F1 and lower macro-F1.

| Configuration | Micro-F1 | Macro-F1 | Macro-AUC | Macro-AP |
| --- | ---: | ---: | ---: | ---: |
| Label prior, positive only if the training rate is at least 0.5 | 0.000 | 0.000 | 0.500 | 0.029 |
| Literal keyword match | 0.108 | 0.106 | 0.536 | 0.051 |
| Word LR, default threshold | 0.466 | 0.443 | 0.929 | 0.415 |
| Word LR, per-label thresholds | 0.484 | 0.463 | 0.929 | 0.415 |
| Word + character LR, per-label thresholds (primary) | 0.505 | 0.484 | 0.934 | 0.440 |
| Word + character LR, global threshold | 0.524 | 0.484 | 0.934 | 0.440 |

The keyword baseline from the original plan is the second row. The primary model raises macro-F1 from 0.106 to 0.484. Against the matched word-only logistic regression with its own tuned thresholds, the gain is about 0.020 micro-F1 and 0.022 macro-F1. A paired bootstrap over test works, conditional on the fitted models, puts that macro-F1 gap between 0.018 and 0.026 (`reports/full/bootstrap.json`, 300 resamples). Removing 279 test books whose exact description also appears in training or validation leaves macro-F1 at 0.484. That check does not remove shared authors or series.

The primary threshold predicts 0.61 labels per test book, with micro-precision 0.433 and micro-recall 0.605. The global threshold predicts 0.50 labels per book and buys precision with recall. Say which goal the threshold serves. This project fixed macro-F1 before looking at the test set.

The 12-label merge, which ORs `historical` with `historical-romance`, `contemporary` with `contemporary-romance`, and `paranormal` with `paranormal-romance`, has test micro-F1 0.532 and macro-F1 0.505. Those numbers describe fewer, broader labels.

### Per shelf

On the primary model, `manga` has F1 0.880 (precision 0.883, recall 0.878), `historical-romance` 0.697, and `m-m` 0.683. `contemporary` has F1 0.231 and `contemporary-romance` 0.281. Those two also have the weakest ranking: ROC-AUC 0.827 and 0.849. The other thirteen shelves are all above 0.91 AUC. Lowering the `contemporary` threshold below zero, to −0.44, still leaves precision at 0.158. The shelf is hard to rank, not just hard to threshold.

The strong shelves have a repeated vocabulary. `manga` weights the word `manga` and Japanese name fragments. `historical-romance` weights `lady`, `earl`, `duke`, and `London`. `m-m` weights `his` and `gay`, and puts a large negative weight on `her`. The weak shelves' largest positive weights are series names: `lucky harbor`, `virgin river`, `idaho`. A linear model that memorizes an imprint will miss a standalone Harlequin whose blurb never uses those names.

On the training books the alias pairs are not duplicate names for one set of books. `historical` co-occurs with `historical-romance` on 6,467 books and appears alone on 1,032. `contemporary` and `contemporary-romance` co-occur on 2,272 books, while 3,143 have only the `-romance` name and 2,859 have only `contemporary`. Merging them changes the question.

### Ten mistakes, read against the description

`reports/full/error_review_queue.csv` stores up to three confidence-ranked false positives and three false negatives per shelf. Confidence means a score far above the threshold, or far below it. It is not a random sample of errors. Ten of those rows were read from the raw file. Shelf fractions were recomputed with the same exclusion list. The notes are in `reports/manual_error_analysis.csv`.

Four patterns show up.

**The words match the shelf, and the exact name misses the 10% rule.** *Mishounen Produce, Vol. 02* is a manga false positive. The blurb is a school story and says the heroine copies "a character in a manga she likes." Readers used `mangas`, `manga-manhwa`, and `shoujo`. The exact shelf `manga` has 0 of 83 content votes. *Boy Meets Boy Meets Boy* is an `m-m` false positive. The blurb says "male/male/male menages," and the votes are split across `m-m`, `mm`, `mmm`, and `menage`, so `m-m` itself is 7.2%. *Confessions of a Viscount* is a historical-romance false positive at 9 of 95 votes, 9.5%, with a viscount, the ton, and a London rooftop in the blurb.

**A sibling shelf is positive and this name is not.** *Christmas in Lucky Harbor* is a contemporary false positive. `contemporary` is 8.2% of content votes and `contemporary-romance` is 11.9%. `lucky harbor` is a top feature for both. The model is firing the alias.

**The shelf is real and the blurb does not say so.** *Silver* is a manga false negative with `manga` at 36% of content votes. The stored text is an English aristocratic revenge plot and never says manga. *After Hours* is an `m-m` false negative told from Grace's point of view. The blurb asks whom the neighbor is talking to and does not say the partner is a man. `her` and `she` are the largest negative features for `m-m`. *The Notebook* is a chick-lit false negative: 1,454 of 12,548 content votes, 11.6%, while the blurb follows Noah after the Second World War.

**The positive label is a handful of votes on a different plot.** *Dustin's Novel* is a contemporary false negative from 2 of 15 votes. The description calls it the last book of a time-travel trilogy. *Emma: The Wild and Wanton Edition* clears historical-romance by 5 of 49 votes and sells an erotic Austen rewrite. `sexy` is a negative weight for historical-romance. *Lelang Pengantin* has an Indonesian title and an English Harlequin blurb about a Montana rancher. The miss is not a failure to read Indonesian. The positive `contemporary-romance` features are series names this book does not contain.

### What the errors limit

Some false positives are cutoff and alias artifacts. Counting every one of them as a model hallucination overstates the failure. Some false negatives are missing evidence: the description never states the medium or the partner's gender. The contemporary shelves are a real ranking failure, and the manual sample is too small, and too concentrated in the tails, to estimate how often each pattern occurs.
"""))

    cells.append(code(
        "import csv\n"
        "with open('reports/manual_error_analysis.csv', encoding='utf-8') as f:\n"
        "    rows = list(csv.DictReader(f))\n"
        "print(f'{len(rows)} reviewed cases')\n"
        "for row in rows:\n"
        "    print(f\"{row['shelf']:22} {row['error_type']:15} {row['book_id']:>8}  {row['title']}\")",
        "10 reviewed cases\n"
        "manga                  false_positive   13330144  Mishounen Produce, Vol. 02\n"
        "manga                  false_negative    6670295  シルバー 1 (Silver #1)\n"
        "historical-romance     false_positive      87629  Confessions of a Viscount\n"
        "historical-romance     false_negative   17433645  Emma: The Wild and Wanton Edition\n"
        "m-m                    false_positive   29497723  Boy Meets Boy Meets Boy (Mad About the Brit Boys, #4)\n"
        "m-m                    false_negative    1445610  After Hours\n"
        "contemporary           false_positive   10841305  Christmas in Lucky Harbor (Lucky Harbor, #1-2)\n"
        "contemporary           false_negative   29909341  Dustin's Novel\n"
        "contemporary-romance   false_negative    2674675  Lelang Pengantin\n"
        "chick-lit              false_negative    6950627  The Notebook\n",
    ))

    cells.append(new_markdown_cell("""## 5. Related work

The dataset page asks users to cite two papers. Both use Goodreads. Neither predicts shelves from a description, so their scores are not a benchmark for the F1 numbers above.

**Wan and McAuley (2018)** study item recommendation when a user leaves several kinds of feedback that form a chain. A later, more explicit action implies the earlier ones. On Goodreads the chain they use is shelve, then read, then rate, then recommend. Their model, chainRec, scores items for a user by respecting that order. The goal is to rank books a user might interact with. Our input is one book's description and our output is a set of shelf names. We never see a user, so we cannot run their model, and a recall number from a recommender is not a micro-F1 on 15 labels.

Mengting Wan and Julian McAuley. 2018. Item Recommendation on Monotonic Behavior Chains. In *Proceedings of the 12th ACM Conference on Recommender Systems (RecSys '18)*. ACM, 86–94. https://doi.org/10.1145/3240323.3240369

**Wan, Misra, Nakashole, and McAuley (2019)** detect spoiler sentences in book reviews. They release sentence-level spoiler marks and a neural tagger. Spoilers are book-specific, vary by user, and tend to fall later in the review. That is a review-text task. Our file does not contain reviews. A spoiler F1 and a shelf F1 answer different questions about the same public crawl.

Mengting Wan, Rishabh Misra, Ndapa Nakashole, and Julian McAuley. 2019. Fine-Grained Spoiler Detection from Large-Scale Review Corpora. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. Association for Computational Linguistics, 2605–2610. https://aclanthology.org/P19-1248/

**Wang and Manning (2012)** is the text-classification comparison. They show that linear bags of words and bigrams, with multinomial Naive Bayes or a linear SVM, are strong sentiment and topic classifiers, and that bigrams help. Our model is in that family: TF-IDF word and character n-grams and a linear classifier, here logistic regression because the target is multi-label and the class weights are easy to set. Their paper also says why a linear bag of words fails when the meaning depends on what the blurb leaves out. "Not an inhumane monster" is their example of composition. Our *Silver* and *After Hours* cases are the same kind of miss: the shelf is a fact about the book, and the description never states it. We do not claim their sentiment accuracies as a target. Sentiment on movie reviews is one label per document. We have 15 sparse labels and a 10% vote rule.

Sida Wang and Christopher D. Manning. 2012. Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. In *Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*. Association for Computational Linguistics, 90–94. https://aclanthology.org/P12-2018/
"""))

    nb = new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'pygments_lexer': 'ipython3'},
    })
    path = ROOT / 'workbook.ipynb'
    nbformat.write(nb, path)
    print(path, 'cells', len(cells))


if __name__ == '__main__':
    main()

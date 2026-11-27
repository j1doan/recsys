# Assignment 2 continuation plan

Updated November 26, 2026. This continues the original [group plan](modeling/PROJECT_PLAN.md), using the same four work areas. It records the completed modeling contribution and the remaining integration work. A task marked pending means its finished output is not yet included in this repository; check with its owner before duplicating work done elsewhere.

## Start here

1. Read [MODELING_GUIDE_EN.md](modeling/MODELING_GUIDE_EN.md) for the task, algorithms, and results.
2. View the saved outputs in [modeling.ipynb](modeling.ipynb), or download [modeling.html](modeling.html) and open it in a browser.
3. Read [HANDOFF.md](modeling/HANDOFF.md) before combining data, predictions, or notebook sections.
4. Confirm your work area and the person responsible for merging the final shared notebook.

Working handoff repository: [CauthorChu/goodreads-romance-modeling](https://github.com/CauthorChu/goodreads-romance-modeling). Original group repository: [j1doan/recsys](https://github.com/j1doan/recsys). Agree on one final workbook and one integration owner across the repositories.

**Immediate priority for the next available teammate: the Evaluation work area, especially manual error analysis, unless another member is already handling it.** The model experiments and numerical evaluation outputs are ready to reuse.

## Keep the shared task and evaluation protocol consistent

- **Task:** predict 15 reader-assigned shelves from description text; a book may have several labels or none.
- **Data:** full Goodreads romance metadata, 335,449 books; 299,789 pass the description and vote filters.
- **Targets:** use the original exclusion list, at least 10 remaining shelf votes per book, and at least a 10% vote share for a positive shelf. Select the top 15 shelf names from training data only.
- **Split:** group by `work_id`; training 191,692, validation 47,687, test 60,410. The original test book IDs are preserved. The original plan's 239,379 development books are now divided into training and validation.
- **Selected model:** word-plus-character TF-IDF with one-vs-rest logistic regression, `C=1`, balanced class weights, and per-label thresholds chosen on validation data.
- **Selection objective:** validation macro-F1. Model fitting optimizes regularized classification loss; threshold selection optimizes validation F1.
- **Primary task:** retain all 15 labels. The separate 12-label OR-merge experiment combines the historical, contemporary, and paranormal pairs. Its scores describe a different target.

Use the saved split IDs and label names rather than independently creating new partitions or a new label ordering. Keep the existing test results fixed while preparing the analysis; any further changes to models or targets should be clearly identified as additional exploratory experiments.

## Work allocation and current status

| Original work area | Owner | Available now | Remaining handoff |
| --- | --- | --- | --- |
| Modeling | CauthorChu | Label-merge decision, threshold search, eight model configurations, code walkthrough, fitted local models, executed notebook and HTML | Integrate the section and prepare its approximately four-minute explanation |
| Evaluation | To assign or confirm | Prior and keyword baselines, aggregate and per-shelf metrics, comparison plots, bootstrap intervals, repeated-text diagnostic, error-review queue | Read error cases, explain results and metric trade-offs, write the final evaluation narrative |
| EDA and data context | To assign or confirm | Verified modeling sample counts and label support; preliminary observations in the original plan and `code.py` | Check raw metadata, produce EDA plots, document data collection and cleaning, write the final data section |
| Related work, slides, and video | To assign or confirm | Dataset-paper links and implementation references; modeling presentation outline | Read and summarize relevant research, assemble slides, coordinate recording and submission |

The integration owner can be one of these members. Confirm names and the recording format as a group; this file does not assign work to unnamed classmates.

## 1. Modeling handoff: completed experiments to reuse

Completed work includes:

- Eight LR/SVM configurations covering regularization, class weighting, and word versus word-plus-character TF-IDF.
- Default, global, and per-label thresholds, with model selection and threshold tuning performed on validation data only.
- A separate 12-label sensitivity experiment using OR on the original binary targets, without adding potentially overlapping shelf votes.
- Seven passing unit tests, independent metric verification, and an executed notebook with four embedded figures.
- A prediction CLI, feature-weight tables, shared split IDs, and integration documentation.

The main pipeline does not refit on training plus validation after tuning, because that could change the score scale used by the thresholds. The original `code.py` remains unchanged.

### Matched test results for the shared presentation

| Configuration | Micro-F1 | Macro-F1 | Macro-AUC |
| --- | --- | --- | --- |
| Word LR, default threshold | 0.4661 | 0.4427 | 0.9286 |
| Word LR, per-label thresholds | 0.4843 | 0.4626 | 0.9286 |
| Word + character LR, per-label thresholds: primary configuration | 0.5047 | 0.4843 | 0.9344 |
| Word + character LR, global threshold: secondary operating point | 0.5242 | 0.4841 | 0.9344 |

Source: [test_results.csv](reports/full/test_results.csv). Compare these matched rows rather than attributing differences from the original plan's older run entirely to modeling changes. Global thresholds have higher test micro-F1 here; the primary configuration follows the validation macro-F1 criterion chosen before test evaluation.

For the 12-label target, micro-F1 is 0.5320 and macro-F1 is 0.5048. Present this separately from the original 15-label comparison.

## 2. Evaluation owner: interpretation and manual error analysis

### Reuse these existing outputs

| File under `reports/full/` | Use |
| --- | --- |
| `test_results.csv` | Prior, literal-keyword, reference LR, selected-model, threshold-policy, and merged-target results |
| `validation_search.csv` | Compare all eight candidates and explain how the model was selected |
| `per_label_wordchar_lr_C1.csv` | Selected-model F1, precision, recall, ROC-AUC, AP, thresholds, and positive counts |
| `per_label_word_lr_balanced_C1.csv` | Matched reference LR with tuned thresholds |
| `figures/` | Existing comparison plots for the notebook and slides |
| `bootstrap.json` | Paired test-work bootstrap intervals, conditional on the fitted models |
| `unseen_description_sensitivity.json` | Diagnostic excluding exact descriptions seen in training or validation |
| `error_review_queue.csv` | High-confidence false-positive and false-negative cases to read manually |
| `top_features.csv` | Features associated with each shelf score |

The keyword baseline requested in the original plan is already implemented and evaluated. Prioritize interpretation over duplicating that implementation.

### Remaining tasks

- Explain micro-F1, macro-F1, ROC-AUC, and AP, including why accuracy is not a useful primary metric for these sparse labels.
- Discuss strong and weak shelves. For the primary model, `manga` has F1 0.8803, `historical-romance` 0.6969, and `m-m` 0.6834; `contemporary` has 0.2305 and `contemporary-romance` 0.2807.
- Read both false positives and false negatives across strong and weak labels. A suggested starting set is 6-10 cases. The queue contains up to three confidence-ranked examples of each error type per shelf, so describe that selection method rather than treating it as a representative random sample.
- Retrieve descriptions by `book_id` from the raw dataset or the local prepared cache. Inspect text evidence and shelf vote shares; distinguish plausible explanations from confirmed facts.
- Save the human review in a proposed `reports/manual_error_analysis.csv`, with columns such as `book_id`, `shelf`, `error_type`, `text_evidence`, `interpretation`, and `reviewer`. A first reading of 10 confidence-ranked cases is now in that file. The queue itself remains the unreviewed candidate list.
- Write the evaluation section with one aggregate comparison, the per-shelf view, a few annotated cases, and limitations. The draft for the shared notebook is [evaluation/EVALUATION.md](evaluation/EVALUATION.md).

**Completion criterion:** a readable evaluation section supported by the existing results and actual manual inspection, ready for the shared notebook and approximately four minutes of presentation. The draft above is ready to paste. It still needs the integration owner to place it in the shared notebook, and the group to confirm who presents it.

## 3. EDA owner: data context, plots, and cleaning notes

- Describe the source and academic-use conditions from the [official dataset page](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html).
- Explain the raw-to-eligible sample flow: 335,449 raw books, 304,837 with descriptions, and 299,789 eligible books. Confirm the details in `reports/full/data_audit.json`.
- Prepare a compact set of useful plots: description lengths, positive-label support, zero/one/multiple target labels, and the heavy tail of rating counts. Add a training-label co-occurrence view if it helps explain overlapping shelves.
- Verify the original plan's observations about invalid publication years, zero ratings, and missing values on the raw metadata. Explain any filtering or transformations used for the plots. These metadata fields are not inputs to the current text model.
- Explain that 65.6% of eligible books have no positive label among the selected 15, and that `manga` is a reader shelf describing a medium rather than a romance subgenre.
- Use the raw metadata for publication-year, rating, and language analysis. `artifacts/full/prepared.joblib` retains only modeling fields and is insufficient for those broader metadata plots.

The original `code.py` contains preliminary EDA ideas, but its old model results use the earlier two-way split. Keep the final modeling comparisons tied to the current reports.

**Completion criterion:** an English data/EDA section with clear figures, sample counts, and preprocessing notes, suitable for approximately four minutes. Raw data stays outside Git.

## 4. Related-work, slides, and video owner

### Related work

- Read Wan and McAuley (2018), [Item Recommendation on Monotonic Behavior Chains](https://cseweb.ucsd.edu/~jmcauley/pdfs/recsys18b.pdf), for the Goodreads recommendation context.
- Read Wan et al. (2019), [Fine-Grained Spoiler Detection from Large-Scale Review Corpora](https://aclanthology.org/P19-1248/), for the book-review context.
- Add a primary research source relevant to text classification with sparse features or linear models. Verify the paper and bibliographic details before citing it.
- Explain each paper's relationship to our task. The two dataset papers study different prediction problems; do not directly compare their task scores with our shelf-classification F1.
- Produce the actual related-work discussion and citations. Collected links alone are not the completed literature section.

### Presentation and recording

Keep the original plan's approximately 20-minute allocation:

| Section | Minutes | Coordination |
| --- | --- | --- |
| Task, evaluation plan, baselines, validity | 5 | Coordinator with Evaluation owner |
| Data context and EDA | 4 | EDA owner |
| Modeling | 4 | Modeling owner |
| Evaluation and errors | 4 | Evaluation owner |
| Related work | 3 | Related-work owner |

Use the modeling guide's four-minute outline and explain the actual code and results. Confirm who speaks each section or whether one person records the combined presentation. Record a playable MP4, provide the actual shared video URL, and check access from a logged-out browser.

**Completion criterion:** cited related-work prose, a coordinated slide/code walkthrough, and a watchable video of approximately 20 minutes. The current repository does not include the finished group video.

## Integration contract and reproduction

Use these conventions when combining sections:

- `reports/full/splits.csv.gz` maps book/work IDs to partitions. Join records by ID rather than assuming another table has the same row order.
- Use `reports/full/selection.json` for the label order and thresholds. Align label columns by name.
- The local `artifacts/full/test_predictions.npz` contains `y_true`, `scores`, `predictions`, `thresholds`, `labels`, `book_ids`, and `work_ids`. Its explicit book order differs from the split CSV ordering.
- Use binary predictions for F1 and decision scores for AUC/AP. Decision scores are not calibrated probabilities.
- Keep zero-target-label books. Fit vocabularies on training data only and tune on validation only.
- There are 279 test books with an exact normalized description also seen in training/validation. Excluding them leaves 60,131 books and macro-F1 0.4842. This is a diagnostic, not complete removal of near duplicates or shared authors/series.

### A fresh clone has reports, but no raw data or fitted artifacts

Reading saved notebook outputs and HTML does not require retraining. Running inference, re-executing the modeling notebook, or verifying prediction arrays requires local artifacts. Obtain them from the modeling owner or reproduce them:

```bash
git clone https://github.com/CauthorChu/goodreads-romance-modeling.git
cd goodreads-romance-modeling
python -m pip install -r requirements-modeling.txt
python -m scripts.download_romance
python -m unittest discover -s tests -v
python -m modeling.run --jobs 4
python -m scripts.build_modeling_report
python -m scripts.verify_modeling
```

Use Python 3.11+. The full experiment took about 33 minutes on the recorded 32 GiB machine; reduce `--jobs` when less memory is available. The unchanged `code.py` conflicts with a Python standard-library module name. The report builder handles this; use `PYTHONSAFEPATH=1 jupyter lab` for interactive notebook work.

The existing verifier checks the standalone modeling files and local artifacts. It does not validate the eventual combined group notebook or the video; the integration owner must check those separately.

Use a branch for each work area, such as `eval-analysis` or `eda-analysis`, and a PR for review. If you lack write access, coordinate collaborator access or contribute through a fork. Keep datasets, fitted artifacts, environment files, and caches excluded from Git.

## Remaining schedule from the original plan

| Target date | Work to finish |
| --- | --- |
| Thu Nov 26 | Confirm role owners and integration owner; start EDA and manual error review |
| Fri Nov 27 | Finish initial error analysis and read related-work sources |
| Sat Nov 28 | Assemble the shared notebook and draft slides; confirm speakers |
| Sun Nov 29 | Run the combined notebook from a fresh kernel, export HTML, and rehearse |
| Mon Nov 30 | Record, upload, verify the video link, and aim to submit early |
| Tue Dec 1 | Original plan's submission target and buffer; confirm the actual deadline on Gradescope |
| Dec 2 onward | Check when peer grading opens and follow its actual instructions and deadline |

The exact Gradescope deadlines and whether peer grading is submitted individually or as a group still need confirmation. These dates carry forward the group's planning targets, not independently verified course deadlines.

## Final integration checklist

- [ ] Role owners and one final-workbook owner are confirmed.
- [ ] All five presentation sections are included, with consistent task and label definitions.
- [ ] EDA figures and manually reviewed errors are integrated.
- [ ] Related-work claims have verified citations and appropriate task comparisons.
- [ ] The combined notebook runs from a fresh kernel with its required local dependencies/artifacts.
- [ ] The final `workbook.html` contains visible results and begins with `<!DOCTYPE html>`.
- [ ] `video_url.txt` contains one real accessible video link; a Drive video is MP4 and larger than 1 MB.
- [ ] The group verifies the deadline and submits the required files on Gradescope.
- [ ] Follow the later peer-grading instructions and submit `responses.txt` when required. The original plan notes that an empty file is used first to obtain assignments.

Completing the modeling section does not complete the group's other sections, video, or later peer grading. Use this plan to finish and integrate those remaining contributions.

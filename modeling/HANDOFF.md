# Modeling handoff

The executed `modeling.ipynb` and self-contained `modeling.html` are the modeling
contribution to the shared notebook. They are not the group's complete submission.
The assignment still needs the other members' task/EDA/evaluation/related-work
sections, an approximately 20-minute recorded presentation and its real video URL,
and the later peer-grading response. No fake video link or peer-grading file is supplied.

## Final choices

- Main task: original 15 shelves, original exclusion list and 10% / 10-vote rule.
- Label vocabulary: training data only; final order is in `reports/full/selection.json`.
- Split by work_id: train 191,692, validation 47,687,
  test 60,410. Outer seed 0 preserves the group's test IDs.
- Selected model: `wordchar_lr_C1`; selection by validation macro-F1.
- Thresholds: per-shelf F1 optima on validation only. No train+validation refit.
- Secondary task: 12 labels, OR merge of three alias pairs. Report separately.

## Integration contract

The cleanest integration is to adopt these split IDs for all group models.
`reports/full/splits.csv.gz` maps book_id/work_id to split in original eligible-row order.
`reports/full/label_support.csv` gives the names and supports. Always align labels
by name, not by an assumed index from the older global top-K list.
`artifacts/full/test_predictions.npz` arrays use the explicit `book_ids` order
stored inside that NPZ. Do not assume it is sorted or the same order as the split CSV.

For an evaluation-only handoff, use `y_true`, `scores`, `predictions`, `thresholds`,
`labels`, `book_ids`, and `work_ids` from that NPZ. `scores` are decision scores,
not calibrated probabilities. Compute AUC/AP from scores and F1 from predictions.
Keep zero-target-label books. Default- and tuned-threshold evaluations of the
same fitted model have identical AUC/AP.

`test_results.csv` contains only reference LR and the validation-selected model,
their predeclared default/global/per-label policies, prior and keyword baselines,
and the separate merged task. Other candidates are compared on validation in
`validation_search.csv`. Never use the test table to choose a different candidate.
The global threshold optimizes validation micro-F1; the per-label thresholds
optimize validation macro-F1. These are distinct operating goals.

`per_label_*.csv`, `top_features.csv`, `alias_analysis.csv`, `bootstrap.json`, and
`figures/` support evaluation and slides. `error_review_queue.csv` supplies 3
high-confidence FP/FN IDs per shelf for human review; it contains no human analysis.
Descriptions can be located in the local dataset or `artifacts/full/prepared.joblib`.

## Validity and interpretation

This run reserves validation, so the original plan's two-way-split scores are not
a matched comparison. Compare rows of the new test table with each other.
Raw shelf targets are noisy proxies, not human verified genre truth. The fixed
exclusion list is inherited rather than a complete ontology.
There are 279 test books whose exact normalized description
appears in train/validation; a diagnostic excluding these is in
`unseen_description_sensitivity.json`. No claim of complete near-duplicate,
author, series, temporal or out-of-domain separation is made.
Bootstrap intervals resample test work groups and condition on the fitted models.

## Reproduce and export

Run the commands in README using the dependency-equipped Python interpreter.
`python -m modeling.run` retrains on the full local data; the report builder runs
the explanatory notebook from a fresh kernel using those recorded results.
It does not silently train 8 models again. All outputs and images are embedded
in HTML, whose first bytes are `<!DOCTYPE html>`.
The unchanged upstream `code.py` shadows Python's standard-library module.
The builder preloads the standard library and sets `PYTHONSAFEPATH=1` for its
fresh kernel. Use Python 3.11+ and `PYTHONSAFEPATH=1 jupyter lab` for interactive
notebook work in this repository.
After copying the section into the shared workbook, execute that combined
notebook from a fresh kernel and export the group's final **workbook.html**.
Do not rename a notebook to .html. Record and validate the real video separately.

Large data and fitted artifacts are intentionally ignored by Git. Preserve the
original `code.py`. The local clone's upstream base commit is
`4eaaebdf8d89fd178105a5bcc1c976930a8fb284`.

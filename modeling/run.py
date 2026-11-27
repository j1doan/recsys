"""Full-data experiment. Run from the repo root: python -m modeling.run.

All model and threshold choices are frozen using validation macro-F1 before
any test scoring. Models stay fitted on train (no threshold-invalidating refit).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import subprocess
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import parallel_backend
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from .core import (ALIASES, DATA_URL, best_f1_threshold, make_features,
                   make_targets, merge_targets, metrics, per_label_metrics,
                   read_books, select_labels, sha256_file, split_books,
                   tune_thresholds)

# Small, predeclared search; validation macro-F1 is the selection objective.
CANDIDATES = [
    dict(name='word_lr_balanced_C1', features='word', model='lr', C=1., balanced=True),
    dict(name='word_lr_balanced_C025', features='word', model='lr', C=.25, balanced=True),
    dict(name='word_lr_balanced_C4', features='word', model='lr', C=4., balanced=True),
    dict(name='word_lr_unweighted_C1', features='word', model='lr', C=1., balanced=False),
    dict(name='word_svm_C025', features='word', model='svm', C=.25, balanced=True),
    dict(name='word_svm_C1', features='word', model='svm', C=1., balanced=True),
    dict(name='wordchar_lr_C1', features='word_char', model='lr', C=1., balanced=True),
    dict(name='wordchar_svm_C1', features='word_char', model='svm', C=1., balanced=True),
]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + '\n', encoding='utf-8')


def classifier(config, jobs):
    common = dict(C=config['C'], class_weight='balanced' if config['balanced'] else None,
                  random_state=0, max_iter=3000)
    base = (LogisticRegression(solver='liblinear', **common)
            if config['model'] == 'lr' else LinearSVC(dual='auto', **common))
    return OneVsRestClassifier(base, n_jobs=jobs)


def fit_checked(clf, x, y):
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        with parallel_backend('threading'), threadpool_limits(limits=1):
            clf.fit(x, y)
    return clf


def threshold_variants(y, scores):
    return {'default': np.zeros(y.shape[1]),
            'global': np.full(y.shape[1], best_f1_threshold(y, scores)),
            'per_label': tune_thresholds(y, scores)}


def keyword_scores(texts, labels):
    """Literal shelf-name baseline, fixed without looking at validation/test."""
    normalized = [' ' + re.sub(r'[^\w]+', ' ', s.lower()) + ' ' for s in texts]
    return np.array([[' ' + s.replace('-', ' ') + ' ' in t for s in labels]
                     for t in normalized], dtype=float)


def paired_cluster_bootstrap(y, candidate, reference, groups, repetitions=300):
    """Paired work-level resampling preserves dependence between editions."""
    _, inverse = np.unique(groups, return_inverse=True)
    n_groups, k = inverse.max() + 1, y.shape[1]
    aggregates = []
    for p in (candidate, reference):
        a = np.zeros((n_groups, 3, k), dtype=np.int64)
        for j, values in enumerate((y & p, (1-y) & p, y & ~p)):
            np.add.at(a[:, j, :], inverse, values)
        aggregates.append(a)
    rng, draws = np.random.default_rng(42), []
    for _ in range(repetitions):
        idx = rng.integers(n_groups, size=n_groups)
        values = []
        for a in aggregates:
            tp, fp, fn = a[idx].sum(axis=0)
            denom = 2 * tp + fp + fn
            values.extend([2 * tp.sum() / max(denom.sum(), 1),
                           np.divide(2*tp, denom, out=np.zeros(k), where=denom>0).mean()])
        draws.append([*values[:2], values[0]-values[2], values[1]-values[3]])
    names = ['selected_micro_f1', 'selected_macro_f1', 'delta_micro_f1', 'delta_macro_f1']
    return {n: {'low': float(np.quantile(np.array(draws)[:, j], .025)),
                'high': float(np.quantile(np.array(draws)[:, j], .975))}
            for j, n in enumerate(names)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/goodreads_books_romance.json.gz'))
    parser.add_argument('--output', type=Path, default=Path('reports/full'))
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts/full'))
    parser.add_argument('--limit', type=int, help='Smoke tests only; never label as full data.')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    out, art = args.output, args.artifacts
    out.mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)
    # The manifest is the completion marker. A failed rerun must not leave a
    # previous successful manifest alongside newly written partial reports.
    (out/'run_manifest.json').unlink(missing_ok=True)
    start = time.perf_counter()
    print('Reading romance metadata...', flush=True)
    df, audit = read_books(args.data, args.limit)
    split = split_books(df)
    tr, va, te = (split[s] for s in ('train', 'validation', 'test'))
    labels = select_labels(df, tr)
    y = make_targets(df, labels)
    joblib.dump({'books': df, 'splits': split, 'labels': labels, 'y': y},
                art / 'prepared.joblib', compress=1)
    assignment = np.empty(len(df), dtype=object)
    for s, ix in split.items():
        assignment[ix] = s
    pd.DataFrame({'book_id': df.book_id, 'work_id': df.work_id,
                  'split': assignment}).to_csv(out/'splits.csv.gz', index=False)
    audit.update({
        'split_sizes': {s: len(ix) for s, ix in split.items()},
        'split_work_counts': {s: int(df.group.iloc[ix].nunique()) for s, ix in split.items()},
        'labels': labels,
        'label_vocabulary_source': 'training partition only',
        'mean_labels_per_book': float(y.sum(1).mean()),
        'zero_label_fraction': float((y.sum(1)==0).mean()),
        'group_overlap': 0,
        'dataset_sha256': sha256_file(args.data),
        'dataset_bytes': args.data.stat().st_size,
        'data_url': DATA_URL,
    })
    hashes = {s: set(df.description_hash.iloc[ix]) for s, ix in split.items()}
    audit['exact_description_overlap'] = {
        'train_validation_unique_texts': len(hashes['train'] & hashes['validation']),
        'test_books_with_text_in_train_or_validation': int(df.description_hash.iloc[te]
            .isin(hashes['train'] | hashes['validation']).sum()),
    }
    pd.DataFrame({'shelf': labels, **{s+'_positives': y[ix].sum(0)
                                    for s, ix in split.items()}}).to_csv(out/'label_support.csv', index=False)
    write_json(out/'data_audit.json', audit)
    print('Data:', {s: len(ix) for s, ix in split.items()}, labels, flush=True)

    # Feature fitting uses training descriptions exclusively. Neither labels,
    # shelf counts, titles, nor author IDs can enter these matrices.
    validation_rows, bundles = [], {}
    for kind in ('word', 'word_char'):
        print(f'Fitting {kind} features...', flush=True)
        vec, feature_start = make_features(kind), time.perf_counter()
        xtr = vec.fit_transform(df.description.iloc[tr])
        xva = vec.transform(df.description.iloc[va])
        feature_seconds = time.perf_counter() - feature_start
        joblib.dump(vec, art/f'vectorizer_{kind}.joblib', compress=1)
        print(f'Features {xtr.shape}, nnz={xtr.nnz:,}, {feature_seconds:.1f}s', flush=True)
        for config in [c for c in CANDIDATES if c['features'] == kind]:
            print('Training', config['name'], flush=True)
            t = time.perf_counter()
            clf = fit_checked(classifier(config, args.jobs), xtr, y[tr])
            scores = clf.decision_function(xva)
            train_seconds = time.perf_counter()-t
            variants = threshold_variants(y[va], scores)
            for policy, thresholds in variants.items():
                validation_rows.append({'candidate': config['name'], 'policy': policy,
                    **config, **metrics(y[va], scores, thresholds),
                    'train_seconds': train_seconds, 'feature_seconds': feature_seconds,
                    'n_features': xtr.shape[1]})
            bundle = {'classifier': clf, 'config': config, 'labels': labels,
                      'thresholds': variants['per_label'], 'threshold_variants': variants}
            joblib.dump(bundle, art/(config['name']+'.joblib'), compress=1)
            bundles[config['name']] = bundle
            print({policy: round(metrics(y[va], scores, threshold)['macro_f1'], 4)
                   for policy, threshold in variants.items()}, flush=True)
            pd.DataFrame(validation_rows).to_csv(out/'validation_search.csv', index=False)
        del xtr, xva, vec
        gc.collect()

    search = pd.DataFrame(validation_rows)
    eligible = search[search.policy == 'per_label'].sort_values(
        ['macro_f1', 'micro_f1', 'candidate'], ascending=[False, False, True])
    winner_name = eligible.iloc[0]['candidate']
    winner = bundles[winner_name]
    decision = {'selection_metric': 'validation macro-F1, per-label thresholds',
                'secondary_tiebreak': 'validation micro-F1, then candidate name',
                'selected_candidate': winner_name,
                'selected_config': winner['config'],
                'selected_validation_macro_f1': float(eligible.iloc[0].macro_f1),
                'labels': labels, 'thresholds': winner['thresholds'].tolist(),
                'refit_on_train_plus_validation': False,
                'reason_no_refit': 'Keep validation-tuned thresholds on the fitted score scale.',
                'test_used_for_selection': False}
    # Persist the choice BEFORE scoring any held-out test description.
    write_json(out/'selection.json', decision)
    print('FROZEN SELECTION:', winner_name, flush=True)
    vec = joblib.load(art/f'vectorizer_{winner["config"]["features"]}.joblib')
    joblib.dump({**winner, 'vectorizer': vec}, art/'selected_model.joblib', compress=1)

    # Separate coarser-label sensitivity experiment, with the same selected
    # architecture and independent validation thresholds. Never a test-selected model.
    ym, merged_labels = merge_targets(y, labels)
    merge_start = time.perf_counter()
    print('Training merged-label sensitivity model...', flush=True)
    merged_clf = fit_checked(classifier(winner['config'], args.jobs),
                             vec.transform(df.description.iloc[tr]), ym[tr])
    merged_val_scores = merged_clf.decision_function(vec.transform(df.description.iloc[va]))
    merged_thresholds = tune_thresholds(ym[va], merged_val_scores)
    joblib.dump({'classifier': merged_clf, 'vectorizer': vec, 'labels': merged_labels,
                 'thresholds': merged_thresholds}, art/'merged_model.joblib', compress=1)
    merge_fit_seconds = time.perf_counter()-merge_start

    # Test starts here. Fixed prior and literal keyword baselines also enter.
    test_rows, test_scores = [], {}
    reference_name = CANDIDATES[0]['name']
    for name in dict.fromkeys([reference_name, winner_name]):
        b = bundles[name]
        v = joblib.load(art/f'vectorizer_{b["config"]["features"]}.joblib')
        score_start = time.perf_counter()
        s = b['classifier'].decision_function(v.transform(df.description.iloc[te]))
        inference_seconds = time.perf_counter()-score_start
        test_scores[name] = s
        for policy, thresholds in b['threshold_variants'].items():
            test_rows.append({'candidate': name, 'policy': policy, 'task': 'original_15',
                              'inference_seconds': inference_seconds,
                              **metrics(y[te], s, thresholds)})
        per_label_metrics(y[te], s, b['thresholds'], labels).to_csv(
            out/f'per_label_{name}.csv', index=False)
    prior = np.tile(y[tr].mean(0), (len(te), 1))
    test_rows.append({'candidate':'label_prior', 'policy':'probability_0.5',
                      'task':'original_15', **metrics(y[te], prior, .5)})
    keyword = keyword_scores(df.description.iloc[te], labels)
    test_rows.append({'candidate':'literal_keyword', 'policy':'literal_match',
                      'task':'original_15', **metrics(y[te], keyword, .5)})
    ms = merged_clf.decision_function(vec.transform(df.description.iloc[te]))
    test_rows.append({'candidate': winner_name, 'policy':'per_label',
                      'task':f'merged_{len(merged_labels)}_sensitivity',
                      **metrics(ym[te], ms, merged_thresholds)})
    per_label_metrics(ym[te], ms, merged_thresholds, merged_labels).to_csv(
        out/'per_label_merged.csv', index=False)
    pd.DataFrame(test_rows).to_csv(out/'test_results.csv', index=False)

    s = test_scores[winner_name]
    selected_pred = s >= winner['thresholds']
    baseline_default = test_scores[reference_name] >= 0
    baseline_tuned = test_scores[reference_name] >= bundles[reference_name]['thresholds']
    print('Computing paired work bootstrap...', flush=True)
    bootstrap = {'repetitions':300, 'unit':'work_id group', 'seed':42,
                 'vs_reference_default':paired_cluster_bootstrap(
                     y[te], selected_pred, baseline_default, df.group.iloc[te].to_numpy()),
                 'vs_reference_tuned':paired_cluster_bootstrap(
                     y[te], selected_pred, baseline_tuned, df.group.iloc[te].to_numpy())}
    write_json(out/'bootstrap.json', bootstrap)
    clean = ~df.description_hash.iloc[te].isin(hashes['train'] | hashes['validation']).to_numpy()
    write_json(out/'unseen_description_sensitivity.json', {
        'test_books':int(clean.sum()), 'excluded_seen_text_books':int((~clean).sum()),
        'selected':metrics(y[te][clean], s[clean], winner['thresholds']),
        'reference_default':metrics(y[te][clean], test_scores[reference_name][clean], 0),
        'reference_tuned':metrics(y[te][clean], test_scores[reference_name][clean],
                                  bundles[reference_name]['thresholds']),
        'note':'Diagnostic subset only; not a separately retrained deduplicated benchmark.'})
    np.savez_compressed(art/'test_predictions.npz', y_true=y[te], scores=s,
                        predictions=selected_pred, thresholds=winner['thresholds'],
                        labels=np.array(labels), book_ids=df.book_id.iloc[te].to_numpy(dtype=str),
                        work_ids=df.work_id.iloc[te].to_numpy(dtype=str))

    # Useful handoff tables. Error rows are IDs + outcomes; teammates can review
    # the actual descriptions locally rather than accepting automatic judgments.
    errors = []
    for j, label in enumerate(labels):
        for error_type, mask in [('false_positive', (y[te, j]==0) & selected_pred[:, j]),
                                 ('false_negative', (y[te, j]==1) & ~selected_pred[:, j])]:
            indices = np.flatnonzero(mask)
            indices = indices[np.argsort(-np.abs(s[indices,j]-winner['thresholds'][j]))[:3]]
            for q in indices:
                b = df.iloc[te[q]]
                errors.append({'shelf':label, 'error_type':error_type, 'book_id':b.book_id,
                               'title':b.title, 'score':float(s[q,j]),
                               'threshold':float(winner['thresholds'][j]),
                               'shelf_vote_fraction':b.shelves.get(label,0)/b.votes})
    pd.DataFrame(errors).to_csv(out/'error_review_queue.csv', index=False)
    terms = vec.get_feature_names_out()
    top_terms = []
    for label, estimator in zip(labels, winner['classifier'].estimators_):
        w = estimator.coef_.ravel()
        for sign, indices in [('positive',np.argsort(w)[-12:][::-1]),
                              ('negative',np.argsort(w)[:8])]:
            for rank, j in enumerate(indices, 1):
                top_terms.append({'shelf':label, 'direction':sign, 'rank':rank,
                                  'feature':terms[j], 'coefficient':w[j]})
    pd.DataFrame(top_terms).to_csv(out/'top_features.csv', index=False)
    alias_rows = []
    for a, b in ALIASES.items():
        if a in labels and b in labels:
            aa, bb = y[tr,labels.index(a)].astype(bool), y[tr,labels.index(b)].astype(bool)
            alias_rows.append({'shelf_a':a, 'shelf_b':b, 'train_a':int(aa.sum()),
                               'train_b':int(bb.sum()), 'intersection':int((aa&bb).sum()),
                               'union':int((aa|bb).sum()),
                               'positive_jaccard':float((aa&bb).sum()/max((aa|bb).sum(),1))})
    pd.DataFrame(alias_rows).to_csv(out/'alias_analysis.csv', index=False)
    write_json(out/'merge_decision.json', {
        'main_task':'retain original 15 labels for continuity with group work',
        'auxiliary_task':'OR positive indicators of the three predeclared alias pairs',
        'merged_labels':merged_labels, 'aliases':ALIASES,
        'validation_metrics':metrics(ym[va], merged_val_scores, merged_thresholds),
        'fit_seconds':merge_fit_seconds,
        'caveat':'Broader labels change the target; the F1 difference is not a model improvement.'})
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable'
    write_json(out/'run_manifest.json', {
        'source_repo':'https://github.com/j1doan/recsys', 'upstream_commit':commit,
        'python':platform.python_version(), 'numpy':np.__version__,
        'scipy':scipy.__version__, 'pandas':pd.__version__, 'sklearn':sklearn.__version__,
        'jobs':args.jobs, 'inner_threads':1, 'wall_seconds':time.perf_counter()-start,
        'candidates':CANDIDATES, 'data_limit':args.limit,
        'label_selection':'train only', 'split_seeds':{'outer':0,'inner':1},
        'bootstrap_seed':42, 'selection_frozen_before_test':True})
    print(pd.DataFrame(test_rows).to_string(index=False), flush=True)
    print(f'Done in {time.perf_counter()-start:.1f}s. Reports: {out}', flush=True)


if __name__ == '__main__':
    main()

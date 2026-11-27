"""Tests target leakage boundaries and exact threshold/merge semantics."""
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from modeling.core import (best_f1_threshold, make_targets, merge_targets,
                           select_labels, split_books, tune_thresholds)


class ModelingTests(unittest.TestCase):
    def test_threshold_matches_exhaustive_tied_score_search(self):
        rng = np.random.default_rng(0)
        for _ in range(60):
            y = rng.integers(2, size=31)
            s = rng.integers(-4, 5, size=31).astype(float)
            t = best_f1_threshold(y, s)
            candidates = np.r_[np.unique(s), np.nextafter(s.max(), np.inf)]
            optimum = max(f1_score(y, s >= q, zero_division=0) for q in candidates)
            self.assertAlmostEqual(f1_score(y, s >= t, zero_division=0), optimum)

    def test_no_positive_label_predicts_none(self):
        s = np.array([1.,1.,2.])
        self.assertFalse((s >= best_f1_threshold(np.zeros(3), s)).any())

    def test_per_label_thresholds_optimize_macro_f1(self):
        y = np.array([[1,0],[0,1],[0,0],[1,1]])
        s = np.array([[.3,-2.],[-1.,2.],[-2.,-3.],[.5,1.]])
        ts = tune_thresholds(y,s)
        choices = [np.r_[np.unique(s[:,j]), np.nextafter(s[:,j].max(),np.inf)]
                   for j in range(2)]
        optimum = max(f1_score(y,s >= np.array([a,b]),average='macro',zero_division=0)
                      for a in choices[0] for b in choices[1])
        self.assertAlmostEqual(f1_score(y,s >= ts,average='macro',zero_division=0),optimum)

    def test_train_only_label_selection(self):
        df = pd.DataFrame({'shelves':[{'train':10},{'test_only':1000}], 'votes':[10,1000]})
        self.assertEqual(select_labels(df,np.array([0]),k=1),['train'])

    def test_label_fraction_boundary_and_negatives(self):
        df = pd.DataFrame({'shelves':[{'a':1},{'a':1},{}], 'votes':[10,11,20]})
        np.testing.assert_array_equal(make_targets(df,['a']),[[1],[0],[0]])

    def test_merge_or_never_sums_reader_votes(self):
        y = np.array([[1,0,1],[0,1,0],[0,0,0]],dtype=np.uint8)
        merged, labels = merge_targets(y,['historical','historical-romance','manga'])
        self.assertEqual(labels,['historical','manga'])
        np.testing.assert_array_equal(merged,[[1,1],[1,0],[0,0]])

    def test_editions_stay_in_same_partition(self):
        df = pd.DataFrame({'group':np.repeat([f'w{i}' for i in range(50)],3)})
        splits = split_books(df)
        group_sets = [set(df.group.iloc[x]) for x in splits.values()]
        for a in range(3):
            for b in range(a):
                self.assertFalse(group_sets[a]&group_sets[b])


if __name__ == '__main__':
    unittest.main()

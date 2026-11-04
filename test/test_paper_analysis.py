"""Check conservation semantics and the weighted gene-bootstrap statistic."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from process_input.build_conservation_scalar_sidecar import _one
from scripts.refresh_recomb2027_results import Ranker, components


class Track:
    def chroms(self):
        return {"chr1": 4}

    def values(self, chrom, start, end, numpy=True):
        assert chrom == "chr1"
        return np.asarray([np.nan, -20., 5., 30.])[start:end]


def test_conservation_defined_mean_and_clipping():
    # Undefined sites contribute zero to the maximum, but not the mean's count.
    at, mean, maximum = _one(Track(), {"chr1"}, "1", 3, clip=10.)
    assert at == .5 and maximum == 1.
    np.testing.assert_allclose(mean, 1 / 6, rtol=1e-7)
    assert _one(Track(), {"chr1"}, "chrX", 3) == (0., 0., 0.)


def test_weighted_ties_match_reference_metrics():
    y = np.asarray([0, 1, 1, 0, 1, 0])
    score = np.asarray([.1, .4, .4, .8, .9, .9])
    weights = np.asarray([2, 0, 3, 1, 2, 1])
    np.testing.assert_allclose(Ranker(y, score).score(weights), [
        roc_auc_score(y, score, sample_weight=weights),
        average_precision_score(y, score, sample_weight=weights),
    ])


def test_gene_components_are_transitive():
    group = components(np.asarray(["A|B", "B|C", "C", "D", "", ""]))
    assert group[0] == group[1] == group[2]
    assert group[3] != group[0] and group[4] != group[5]

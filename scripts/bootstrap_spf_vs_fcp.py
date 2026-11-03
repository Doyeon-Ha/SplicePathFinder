"""Gene-clustered bootstrap CIs and a paired test for SPF vs FC_P.

Scores are fixed across resamples, so we sort once per model and express each
bootstrap replicate as a weight vector over rows (gene g drawn k times -> every
row of g gets weight k). Weighted AUROC and weighted average precision are then
O(n) per replicate instead of an O(n log n) re-sort.
"""
import numpy as np, pandas as pd

d = np.load('/tmp/paired.npz')
y, a, b, fold = d['y'].astype(np.int8), d['a'], d['b'], d['fold']

B = './data/eval/clinvar_pb_2026_split/coord_splice_foldsets/'
t = pd.concat([pd.read_csv(B + f'fold{k}_test.tsv', sep='\t',
                           usecols=['global_idx', 'gene_symbol', 'label'])
               for k in range(5)]).sort_values('global_idx')
assert np.array_equal(t['label'].values.astype(int), y.astype(int)), "label mismatch"
gene = t['gene_symbol'].fillna('__nogene').values
ug, inv = np.unique(gene, return_inverse=True)
G = len(ug)


class Ranker:
    """Weighted AUROC / average precision for one fixed score vector."""

    def __init__(self, s, y):
        o = np.argsort(s, kind='mergesort')
        self.o, self.y = o, y[o]
        ss = s[o]
        self.gid = np.r_[0, np.cumsum(np.diff(ss) != 0)]
        self.ng = self.gid[-1] + 1
        self.pos, self.neg = (self.y == 1), (self.y == 0)

    def auroc(self, w):
        ws = w[self.o]
        pg = np.bincount(self.gid, weights=ws * self.pos, minlength=self.ng)
        ng = np.bincount(self.gid, weights=ws * self.neg, minlength=self.ng)
        Wp, Wn = pg.sum(), ng.sum()
        if Wp == 0 or Wn == 0:
            return np.nan
        below = np.r_[0.0, np.cumsum(ng)[:-1]]
        return float((pg * (below + 0.5 * ng)).sum() / (Wp * Wn))

    def ap(self, w):
        # descending score = reverse group order
        ws = w[self.o]
        pg = np.bincount(self.gid, weights=ws * self.pos, minlength=self.ng)[::-1]
        ng = np.bincount(self.gid, weights=ws * self.neg, minlength=self.ng)[::-1]
        Wp = pg.sum()
        if Wp == 0:
            return np.nan
        tp, fp = np.cumsum(pg), np.cumsum(ng)
        keep = (pg + ng) > 0
        tp, fp, pgk = tp[keep], fp[keep], pg[keep]
        prec = tp / np.maximum(tp + fp, 1e-12)
        return float((prec * pgk).sum() / Wp)          # == sum dRecall * Precision


RA, RB = Ranker(a, y), Ranker(b, y)
ones = np.ones(len(y))
print(f"point estimates  SPF  AUROC {RA.auroc(ones):.4f}  AUPRC {RA.ap(ones):.4f}")
print(f"                 FC_P AUROC {RB.auroc(ones):.4f}  AUPRC {RB.ap(ones):.4f}")
print(f"genes {G}  rows {len(y)}  pos {int(y.sum())}\n")

print("=== per-fold spread (checklist 4.11) ===")
for nm, R in (('SPF', RA), ('FC_P', RB)):
    ro, pr = [], []
    for k in range(5):
        wk = (fold == k).astype(float)
        ro.append(R.auroc(wk)); pr.append(R.ap(wk))
    print(f"  {nm:5s} AUROC {np.mean(ro):.4f} +- {np.std(ro, ddof=1):.4f}"
          f"   AUPRC {np.mean(pr):.4f} +- {np.std(pr, ddof=1):.4f}")
    print(f"        per fold ROC {['%.4f' % v for v in ro]}  PRC {['%.4f' % v for v in pr]}")

NB = 2000
rng = np.random.default_rng(42)
ra = np.empty(NB); rb = np.empty(NB); pa = np.empty(NB); pb = np.empty(NB)
for i in range(NB):
    cnt = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
    w = cnt[inv]
    ra[i], rb[i] = RA.auroc(w), RB.auroc(w)
    pa[i], pb[i] = RA.ap(w), RB.ap(w)

q = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
print(f"\n=== gene-clustered bootstrap, {NB} replicates (4.11 CIs) ===")
for nm, r, p in (('SPF', ra, pa), ('FC_P', rb, pb)):
    print(f"  {nm:5s} AUROC 95% CI [{q(r)[0]:.4f}, {q(r)[1]:.4f}]"
          f"   AUPRC 95% CI [{q(p)[0]:.4f}, {q(p)[1]:.4f}]")

print("\n=== paired difference SPF - FC_P (4.12) ===")
for nm, dd in (('dAUROC', ra - rb), ('dAUPRC', pa - pb)):
    lo, hi = q(dd)
    pv = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
    ptxt = f"< {1/NB:.4f}" if pv == 0 else f"= {pv:.4g}"
    print(f"  {nm} = {dd.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  two-sided p {ptxt}")

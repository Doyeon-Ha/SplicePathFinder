"""Fixed-threshold operating characteristics."""
import numpy as np

def counts(y, pos):
    """(tp, fp, fn, tn) for a boolean call vector."""
    tp = int(np.sum(pos & (y == 1))); fp = int(np.sum(pos & (y == 0)))
    return tp, fp, int(np.sum(~pos & (y == 1))), int(np.sum(~pos & (y == 0)))

def op(tp, fp, fn, tn):
    """Operating characteristics from a confusion matrix.

    LR+/LR- are reported because they are the only prevalence-free quantities here:
    ClinVar's pathogenic fraction is an ascertainment artefact, so PPV/NPV describe
    this benchmark, not a clinic.
    """
    P, N = tp + fn, fp + tn
    sens = tp / P if P else float("nan")
    spec = tn / N if N else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"n": P + N, "n_pos": P, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sens": sens, "spec": spec, "ppv": ppv, "npv": npv,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
            "mcc": ((tp * tn - fp * fn) / den) if den else float("nan"),
            "youden_j": sens + spec - 1,
            "lr_pos": (sens / (1 - spec)) if spec < 1 else float("inf"),
            "lr_neg": ((1 - sens) / spec) if spec > 0 else float("nan"),
            "flagged_frac": (tp + fp) / (P + N)}

def at(y, s, t):
    m = np.isfinite(s)
    return op(*counts(y[m], s[m] >= t)) | {"threshold": float(t), "coverage": float(m.mean())}

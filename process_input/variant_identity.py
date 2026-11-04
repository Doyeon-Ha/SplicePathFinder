"""Variant identity and row-order checks shared by preparation and evaluation."""
import hashlib
import numpy as np


def decode(values):
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x)
                       for x in values], dtype=object)


def variant_keys(table):
    return np.asarray([
        f'{c.removeprefix("chr")}:{int(p)}:{r}:{a}'
        for c, p, r, a in zip(decode(table['chrom'][:]), table['pos'][:],
                              decode(table['ref'][:]), decode(table['alt'][:]))
    ], dtype=object)


def row_digest(keys):
    h = hashlib.sha256()
    for key in keys:
        h.update((str(key) + '\n').encode())
    return h.hexdigest()


def require_equal(expected, actual, name):
    if len(expected) != len(actual):
        raise ValueError(f'{name}: {len(actual)} rows, expected {len(expected)}')
    # A few long alleles must not pad every cohort key to the longest width.
    different = np.flatnonzero(np.asarray(expected, dtype=object) !=
                               np.asarray(actual, dtype=object))
    if len(different):
        raise ValueError(f'{name}: row identity/value mismatch at row {int(different[0])}')

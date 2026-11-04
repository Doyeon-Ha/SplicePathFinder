"""Join ClinVar variant/submission summaries into a cohort-row-aligned gene table.

All matched submissions contribute gene symbols, including submissions that do
not contribute to the aggregate classification. Missing submitted genes remain
missing. Preferred gene is parsed from the first matching GRCh38 Name.
"""
import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
import h5py
from scripts.build_clinvar_recordgene_strata import record_gene
from process_input.variant_identity import variant_keys, row_digest

MISSING = {'', '.', '-', 'na', 'n/a', 'nan', 'none', 'null', 'not provided', 'not specified'}
csv.field_size_limit(sys.maxsize)


def rows(path):
    with gzip.open(path, 'rt') as f:
        header = next(line.lstrip('#').rstrip('\n').split('\t') for line in f
                      if line.lstrip('#').startswith(('AlleleID\t', 'VariationID\t')))
        # NCBI summaries are literal tab-separated records; quote characters in
        # submitter text must not consume subsequent records as CSV continuations.
        yield from csv.DictReader(f, fieldnames=header, delimiter='\t', quoting=csv.QUOTE_NONE)


def build(variant_h5, variant_summary, submission_summary, output):
    with h5py.File(variant_h5) as f:
        keys = variant_keys(f).tolist()
    where = {k: i for i, k in enumerate(keys)}
    if len(where) != len(keys):
        raise ValueError('Duplicate cohort identities')
    ids, names = [set() for _ in keys], {}
    id_to_row = {}
    for r in rows(variant_summary):
        if r['Assembly'] != 'GRCh38':
            continue
        key = ':'.join([r['Chromosome'].removeprefix('chr'), r['PositionVCF'],
                        r['ReferenceAlleleVCF'], r['AlternateAlleleVCF']])
        i = where.get(key)
        if i is None:
            continue
        vid = r['VariationID']
        if vid in id_to_row and id_to_row[vid] != i:
            raise ValueError('A VariationID maps to multiple cohort identities')
        ids[i].add(vid)
        id_to_row[vid] = i
        names.setdefault(i, r['Name'])
    if any(not x for x in ids):
        raise ValueError('variant_summary does not cover every cohort row')
    genes = [set() for _ in keys]
    for r in rows(submission_summary):
        i = id_to_row.get(r['VariationID'])
        if i is not None:
            genes[i].update(g.strip() for g in re.split(r'[|;,]', r['SubmittedGeneSymbol'])
                            if g.strip().lower() not in MISSING)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, 'wt', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['key', 'submitted_genes', 'preferred_gene'])
        for i, key in enumerate(keys):
            w.writerow([key, '|'.join(sorted(genes[i])), record_gene(names[i])])
    inputs = {}
    for path in [variant_summary, submission_summary]:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        inputs[str(path)] = h.hexdigest()
    report = dict(rows=len(keys), row_key_sha256=row_digest(keys), source_sha256=inputs)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant-h5', required=True)
    p.add_argument('--variant-summary', required=True)
    p.add_argument('--submission-summary', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    print(json.dumps(build(a.variant_h5, a.variant_summary, a.submission_summary, a.out), indent=2))


if __name__ == '__main__':
    main()

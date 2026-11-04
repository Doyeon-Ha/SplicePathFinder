"""Render a backbone's saved refresh results at their final 6.5-inch paper width (Slurm only)."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--only-pr', action='store_true',
                        help='Refresh the precision-recall figure without redrawing case figures')
    parser.add_argument('--only-cases', action='store_true',
                        help='Preserve the author case plots without requiring an SPF head')
    args = parser.parse_args()
    if args.only_pr and args.only_cases:
        parser.error('--only-pr and --only-cases are mutually exclusive')
    src, out = args.refresh_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['STIXGeneral', 'Nimbus Roman', 'DejaVu Serif'],
        'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
        'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    })
    metadata = {'Creator': None, 'Producer': None, 'CreationDate': None}
    colors = {'SPF': '#2a78d6', 'Pangolin': '#eb6834', 'SpliceAI': '#1baf7a'}
    styles = {'SPF': '-', 'Pangolin': (0, (5.5, 1.8)),
              'SpliceAI': (0, (1.2, 1.4, 4.5, 1.4))}
    summary = json.loads((src/'clinvar_metrics.json').read_text())
    tag = summary.get('tag', 'ep12')
    report = {'curve_average_precision': {}}
    if not args.only_cases:
        frame = np.load(src/'analysis_frame.npz')
        cuts = json.loads((src/'spf_clinical_cutoffs.json').read_text())['reported']['cutoffs']
        metrics = summary['rows']
        panels = [('deep-intronic', 'deep-intronic (≥100 bp)'),
                  ('SR-acc', 'SR-acc (acceptor splice region)'), ('prox-mid', 'prox-mid (intron 9–99)')]
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.12), layout='constrained')
        for ax, (key, title) in zip(axes, panels):
            mask = frame['m_'+key]
            y = frame['y'][mask]
            ax.axhline(y.mean(), color='#8a8a86', lw=.7, ls=(0, (3, 2)))
            curve_handles, mark_handles = [], []
            report['curve_average_precision'][key] = {}
            for model in colors:
                scores = frame['spf' if model == 'SPF' else model][mask]
                assert np.isfinite(scores).all(), 'Saved frame must retain no-calls as zero.'
                precision, recall, thresholds = precision_recall_curve(y, scores)
                ap = float(average_precision_score(y, scores))
                assert abs(ap - metrics[key][model]['auprc']) < 1e-10
                report['curve_average_precision'][key][model] = ap
                curve_handles.append(ax.step(recall, precision, where='post',
                    color=colors[model], ls=styles[model], lw=1.7 if model == 'SPF' else 1.2,
                    label=model)[0])
                if model == 'SPF':
                    for threshold, marker in zip(cuts, ['o', 's', '^']):
                        i = min(int(np.searchsorted(thresholds, threshold)), len(recall)-1)
                        mark_handles.append(ax.plot(recall[i], precision[i], marker,
                            ms=5.4, mfc='white', mec=colors[model], mew=1.2,
                            label=f't = {threshold:g}')[0])
            ax.set(xlim=(-.01, 1.01), ylim=(-.02, 1.04), xlabel='Recall')
            ax.set_xticks([0, .5, 1])
            ax.set_title(f'{title}\nn={len(y):,} · {100*y.mean():.1f}% pathogenic', pad=4)
            ax.grid(True, lw=.6, color='#e3e2dd')
            ax.set_axisbelow(True)
        axes[0].set_ylabel('Precision')
        handles = [h for pair in zip(curve_handles, mark_handles) for h in pair]
        fig.legend(handles, [h.get_label() for h in handles], ncol=3,
            loc='outside lower center', frameon=False, handlelength=3,
            columnspacing=2, labelspacing=.4)
        fig.savefig(out/'spf_pr_hard_strata.pdf', metadata=metadata)
        plt.close(fig)

    if args.only_pr:
        (out/'pr_validation.json').write_text(json.dumps(report, indent=2) + '\n')
        return

    cases = json.loads((src/'cases/case_scores.json').read_text())
    with h5py.File(src/'cases/cache.h5') as cache:
        for i, case in enumerate(cases):
            ref, alt = cache['ref_probs'][i], cache['alt_probs'][i]
            x = np.arange(len(ref)) - 4375
            rb1 = case['gene'] == 'RB1'
            lo, hi = (-150, 75) if rb1 else (-32, 123)
            points = [-103, -1] if rb1 else [11, 76]
            mask = (x >= lo) & (x <= hi) & (cache['coverage'][i] > 0)
            # Author's case layout: alternate-allele bars, signed acceptor/donor
            # axis, pale pseudoexon shading, red variant, and an exon schematic.
            # Only probabilities and the resulting axis scale depend on the checkpoint.
            fig = plt.figure(figsize=(3.125, 2.65))
            ax = fig.add_axes([.16, .24, .83, .74])
            upper = .42 if rb1 else .72
            tick = .3 if rb1 else .5
            peak = float(alt[mask, :2].max())
            if peak >= upper:
                upper = float(np.ceil(peak * 1.15 * 10) / 10)
                tick = .5 if upper < 1 else 1.
            acc_color, don_color = '#1c7198', '#200b18'
            ax.axvspan(*points, color='#fff4d3', zorder=0)
            for channel, sign, color in [(0, 1, acc_color), (1, -1, don_color)]:
                ax.vlines(x[mask], 0, sign*alt[mask, channel], color=color, linewidth=.65)
            ax.spines['bottom'].set_position(('data', 0))
            ax.spines['bottom'].set_color('#20242c')
            ax.spines['left'].set_color('#20242c')
            ax.set(xlim=(lo,hi),ylim=(-upper,upper),xticks=[],yticks=[-tick,0,tick])
            ax.set_yticklabels([f'{tick:g}','0',f'{tick:g}'])
            ax.grid(axis='y',color='#cbd0d9',linewidth=.65,zorder=0)
            ax.set_ylabel('Splice-site probability',labelpad=1)
            ax.text(.012,.80,'acceptor',rotation=90,transform=ax.transAxes,
                    ha='left',va='center')
            ax.text(.012,.12,'donor',rotation=90,transform=ax.transAxes,
                    ha='left',va='center')
            for channel,sign,p,color,label in [(0,1,points[0],acc_color,"cryptic 3′ss"),
                                               (1,-1,points[1],don_color,"cryptic 5′ss")]:
                y=sign*float(alt[p+4375,channel])
                ax.plot(p,y,'o',ms=4.5,mec='white',mew=.7,color=color,zorder=5)
                # Keep annotation callouts clear of the measured probability bars.
                tx=p+4 if rb1 or channel==0 else p+2
                ty=upper*.83 if channel==0 else -upper*.72
                align='left'
                if not rb1 and channel==1:
                    tx=hi-1;align='right'
                ax.text(tx,ty,f'{label}\n{p:+d} bp',color=color,ha=align,va='center')
            vy=upper*.045 if rb1 else -upper*.045
            ax.plot(0,vy,'o',color='#cc0000',ms=4.5,zorder=6)
            ax.plot([0,0],[vy,upper*.19 if rb1 else -upper*.19],
                    color='#cc0000',ls='--',lw=.7)
            tx=hi-1 if rb1 else lo+1
            ax.text(tx,upper*.29 if rb1 else -upper*.34,
                    'variant\n'+case['hgvs'],color='#cc0000',
                    ha='right' if rb1 else 'left',va='center',fontsize=10)
            ax.text(hi, -upper*.08,'(bp)',ha='right',va='top')
            exon = fig.add_axes([.01,.02,.98,.14]);exon.set_axis_off()
            exon.set(xlim=(0,1),ylim=(0,1))
            for x0,x1,label,color in [(0,.18,'Exon 23' if rb1 else 'Exon 4','#c8d4ec'),
                 (.30,.72,f'pseudoexon {points[1]-points[0]+1} bp','#efaa61'),
                 (.84,1,'Exon 24' if rb1 else 'Exon 5','#c8d4ec')]:
                exon.add_patch(FancyBboxPatch((x0,.12),x1-x0,.75,
                    boxstyle='round,pad=0,rounding_size=.04',fc=color,ec='#8c9198',lw=.7))
                exon.text((x0+x1)/2,.49,label,ha='center',va='center',fontsize=10)
            for left,right in [(.18,.30),(.72,.84)]:
                exon.plot([left,right],[.49,.49],color='#777d86',lw=.7)
                exon.text((left+right)/2,.49,'//',ha='center',va='center',
                          bbox={'facecolor':'white','edgecolor':'none','pad':0})
            fig.savefig(out/f"{case['gene'].lower()}_{tag}.pdf", metadata=metadata)
            plt.close(fig)
    inputs = ['clinvar_metrics.json', 'cases/case_scores.json', 'cases/cache.h5']
    if not args.only_cases:
        inputs += ['analysis_frame.npz', 'spf_clinical_cutoffs.json']
    report['input_sha256'] = {str(src/f): hashlib.sha256((src/f).read_bytes()).hexdigest()
                              for f in inputs}
    report['figure_width_inches'] = 6.5
    report['font_size_pdf_points'] = 10
    report['case_width_inches'] = 3.125
    report['case_tracks'] = 'alternate-allele probabilities, author bar-plot layout'
    (out/'figure_report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()

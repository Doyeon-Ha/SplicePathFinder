"""Formatting for the paper result grids."""

STRATA = ['canonical', 'ess', 'SR-all', 'SR-don', 'SR-acc', 'acc-br', 'prox-mid', 'deep-intronic', 'deep-acc', 'overall', 'SNV', 'indel']
DISPLAY = {'SplicePath': r'$\delta$max$_{SP}$', 'SPF': r'\textbf{SPF}', 'minus_region': r'$-\mathrm{region}$', 'Cons6_MLP': 'Cons6 MLP', 'Cons6_logreg': 'Cons6 logistic', 'SPF_SRtrain': 'SPF (SR train)', 'ST': 'ST', 'SSNet': 'SSNet'}
DISPLAY.update({'FC_P': r'FC$_P$', 'FC_S': r'FC$_S$', 'FC_ST': r'FC$_{ST}$',
                'FC_SN': r'FC$_{SN}$', 'FC_SP': r'FC$_{SP}$',
                'minus_bidir': '$-$bidir', 'minus_hybrid': '$-$hybrid',
                'FC_cons': r'\shortstack{FC\\$+$cons.}',
                'SPF_nocons': r'\shortstack{SPF\\$-$cons.}',
                'Cons6_MLP': r'\shortstack{Cons6\\MLP}',
                'Cons6_logreg': r'\shortstack{Cons6\\logistic}',
                'SPF_SRtrain': r'\shortstack{SPF\\SR train}'})
# Supplementary arms; the position-fixed table shows only FCu columns, so they reuse FC_X.
DISPLAY.update({f'FCu_{b}': DISPLAY[f'FC_{b}'] for b in ['P', 'S', 'ST', 'SN', 'SP']})
DISPLAY.update({f'FCc_{b}': DISPLAY[f'FC_{b}'].replace('FC$', 'FC$^c') for b in ['P', 'S', 'ST', 'SN', 'SP']})
DISPLAY.update({'Evo2': 'Evo 2', 'CaduceusPh': 'Caduceus-Ph', 'DNABERT2': 'DNABERT-2',
                'SPF_Pangolin': r'\shortstack{SPF head\\Pangolin}',
                'SPF_SpliceAI': r'\shortstack{SPF head\\SpliceAI}'})


def num(v):
    return '--' if v is None else f'{v:.3f}'

def table(rows, models, caption, labels, parts=None, wide=True):
    parts = parts or list(rows)
    env = 'table*' if wide else 'table'
    clinical = labels[0] == 'tab:clinvar-cv5-region-fine'
    if clinical:
        models = ['Pangolin','FC_P','SpliceAI','FC_S','ST','FC_ST','SSNet','FC_SN',
                  'SplicePath','FC_SP','SPF']
    elif labels[0] == 'tab:backbone-ablation-short':
        models = ['SplicePath','minus_region','minus_bidir','minus_hybrid']
    elif labels[0] in ['tab:deltareader-ablation-short','tab:deltareader-ablation']:
        models = ['FC_SP','FC_cons','SPF_nocons'] + models
    spacing = '2pt' if clinical or len(models)>5 else '3pt'
    if labels[0] == 'tab:clinvar-cv5-region-fine-dnalm':
        spacing = '1.6pt'
    font = r'\fontsize{9}{10.5}\selectfont' if clinical else r'\normalsize'
    lines = [rf'\begin{{{env}}}[t]\centering',
             '{'+font+rf'\setlength{{\tabcolsep}}{{{spacing}}}',
             r'\begin{tabular}{@{}lrcc'+'c'*len(models)+r'@{}}', r'\toprule']
    if clinical:
        lines += [r' & & & & \multicolumn{8}{c}{Third-party base} & \multicolumn{3}{c}{\textbf{SplicePath} (ours)} \\',
                  r'\cmidrule(lr){5-12}\cmidrule(lr){13-15}']
    elif labels[0] in ['tab:deltareader-ablation-short','tab:deltareader-ablation']:
        lines += [r' & & & & \textbf{A1} & \textbf{A2} & \textbf{A3} & \textbf{All} & '+
                  rf'\multicolumn{{{len(models)-4}}}{{c}}{{Controls}} \\']
    lines += ['Stratum & $n$ & pos/neg & & '+
              ' & '.join(DISPLAY.get(m,m) for m in models)+r' \\',r'\midrule']
    for p in parts:
        if p == 'overall' and p != parts[0]:
            lines.append(r'\midrule')
        m = next(iter(rows[p].values()))
        label = p.replace('_',r'\_')
        if m['n'] and m['n_pos']/m['n'] >= .9:
            label += r'$^{\triangle}$'
        for metric, short in [('auroc','ROC'),('auprc','PRC')]:
            values = [rows[p].get(k,{}).get(metric) for k in models]
            best = max((v for v in values if v is not None), default=None)
            cells = [('--' if v is None else
                      (r'\textbf{' if metric == 'auroc' else r'\underline{')+num(v)+'}'
                      if num(v) == num(best) else num(v)) for v in values]
            prefix = (label+f" & {m['n']:,} & {m['n_pos']:,}/{m['n']-m['n_pos']:,}"
                      if metric == 'auroc' else ' & & ')
            lines.append(prefix+' & '+short+' & '+' & '.join(cells)+r' \\')
        lines.append(r'\addlinespace[1pt]')
    caption += (r' ROC/PRC denote AUROC/AUPRC. Bold marks the highest displayed AUROC; '
                r'underlining marks the highest displayed AUPRC. '
                r'$^{\triangle}$ denotes at least 90\% pathogenic variants.')
    if any(any(k not in rows[p] for k in models) for p in parts):
        caption += r' -- denotes a comparison awaiting completion under the matched protocol.'
        if any(k.startswith('FC') for k in models):
            caption += r' FC denotes a fully connected probe.'
    elif any(k.startswith('FC') for k in models):
        caption += r' FC denotes a fully connected probe.'
    lines += [r'\bottomrule\end{tabular}}',r'\caption{'+caption+'}', *[r'\label{'+l+'}' for l in labels],rf'\end{{{env}}}']
    return '\n'.join(lines)+'\n'

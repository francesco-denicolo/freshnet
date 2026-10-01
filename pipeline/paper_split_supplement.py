"""
paper_split_supplement.py — Second referee report, #7 (length): move appendices A (dataset
characterisation), B (robustness), D (HPO/implementation) and F (supplementary results) of the
MDPI paper into a separate Supplementary Material document, rewriting every cross-reference.

  main  -> supp : \\ref{key} of a moved label becomes its S-number read from supplementary.aux
                  ("Appendix~\\ref{app:x}" -> "Supplementary Section~S2"); \\eqref -> "(Sn)"
  supp  -> main : \\ref{key} of a label that stays in the paper becomes its number read from
                  template.aux (plain numbers = main article, stated at the top of the supplement)
Citations used only in the moved text get a \\nocite in the main paper (MDPI: references in
supplementary files must also appear in the main reference list).

Usage: python paper_split_supplement.py <overleaf_dir>    (works in place; run on a copy first)
"""
import os, re, sys, subprocess

ROOT = sys.argv[1]
MAIN = os.path.join(ROOT, 'template.tex')
SUPP = os.path.join(ROOT, 'supplementary.tex')
MOVE = ['app:eda', 'app:robustness', 'app:impl', 'app:resfig']      # order = S1..S4


def build(name):
    for cmd in (['pdflatex', '-interaction=nonstopmode', name],
                ['bibtex', name],
                ['pdflatex', '-interaction=nonstopmode', name],
                ['pdflatex', '-interaction=nonstopmode', name]):
        subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def labels_from_aux(name):
    out = {}
    for line in open(os.path.join(ROOT, name + '.aux'), encoding='utf-8', errors='ignore'):
        m = re.match(r'\\newlabel\{([^}]*)\}\{\{([^}]*)\}', line)
        if m and not m.group(1).endswith('@cref'):
            out[m.group(1)] = m.group(2)
    return out


src = open(MAIN, encoding='utf-8').read()
start_app = src.index('\\appendixstart')
end_app = src.index('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%% Optional', start_app)
app = src[start_app:end_app]
heads = [m.start() for m in re.finditer(r'^\\section\{', app, flags=re.M)] + [len(app)]
blocks = {}
for a, b in zip(heads, heads[1:]):
    blk = app[a:b]
    lab = re.search(r'\\label\{(app:[^}]*)\}', blk.split('\n')[0]).group(1)
    blocks[lab] = blk
assert all(k in blocks for k in MOVE), blocks.keys()

moved_text = ''.join(blocks[k] for k in MOVE)
moved_labels = set(re.findall(r'\\label\{([^}]*)\}', moved_text))
kept_app = app[:heads[0]] + ''.join(v for k, v in blocks.items() if k not in MOVE)
main_new = src[:start_app] + kept_app + src[end_app:]

# ---- citations used only in the moved text -> \nocite in the main paper
cite_re = re.compile(r'\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]*)\}')
keys = lambda t: {k.strip() for g in cite_re.findall(t) for k in g.split(',')}
only_supp = sorted(keys(moved_text) - keys(main_new))
if only_supp:
    main_new = main_new.replace('\\bibliography{references}',
                                '\\nocite{' + ','.join(only_supp) + '}  % cited in the Supplementary Material\n\\bibliography{references}')

# ---- pass 1: compile the main paper without the moved blocks to number its own labels
open(MAIN, 'w', encoding='utf-8').write(main_new)
build('template')
main_lab = labels_from_aux('template')

# ---- build the supplement, resolving its references to the main paper
pre = src[:re.search(r'^\\begin\{document\}', src, flags=re.M).start()]
pre = re.sub(r'\\Title\{[^\n]*\}', lambda m: '\\Title{Supplementary Material for ``The Conditional Value of '
             'Latent-Demand Recovery in Retail Forecasting: A Large-Scale Benchmark under Stock-out Censoring\'\'}', pre)
pre = re.sub(r'\\abstract\{[^\n]*\}', lambda m: '\\abstract{This document collects the supporting material of the article: '
             'additional dataset characterisation (Section~S1), the robustness checks (Section~S2), the hyperparameter '
             'optimisation and implementation details (Section~S3) and supplementary results (Section~S4). Section, table, '
             'figure and equation numbers without the prefix S refer to the main article.}', pre)
body = moved_text
body = re.sub(r'\bThis appendix\b', 'This section', body)
body = re.sub(r'\bthis appendix\b', 'this section', body)


def to_main(m):
    cmd, key = m.group(1), m.group(2)
    if key in moved_labels:
        return m.group(0)
    if key not in main_lab:
        raise KeyError(f'supplement references unknown label {key}')
    if cmd == 'eqref':
        return f'({main_lab[key]})'
    return main_lab[key] + (' of the main article' if key.startswith('app:') else '')


body = re.sub(r'\\(ref|eqref)\{([^}]*)\}', to_main, body)
supp = (pre + '\\begin{document}\n'
        '\\renewcommand{\\thesection}{S\\arabic{section}}\n'
        '\\renewcommand{\\thetable}{S\\arabic{table}}\n'
        '\\renewcommand{\\thefigure}{S\\arabic{figure}}\n'
        '\\renewcommand{\\theequation}{S\\arabic{equation}}\n\n'
        + body +
        '\n\\begin{adjustwidth}{-\\extralength}{0cm}\n\\reftitle{References}\n\\bibliography{references}\n'
        '\\end{adjustwidth}\n\\end{document}\n')
open(SUPP, 'w', encoding='utf-8').write(supp)
build('supplementary')
supp_lab = labels_from_aux('supplementary')
missing = [k for k in moved_labels if k not in supp_lab]
assert not missing, missing

# ---- pass 2: point the main paper's references to the S-numbers
SEC = {k for k in moved_labels if k.startswith('app:') or k.startswith('subsec') or k.startswith('sec')}


def appendix_phrase(m):
    k1, k2 = m.group(2), m.group(4)
    if k1 in moved_labels and (k2 is None or k2 in moved_labels):
        if k2:
            return f'Supplementary Sections~{supp_lab[k1]} and~{supp_lab[k2]}'
        return f'Supplementary Section~{supp_lab[k1]}'
    return m.group(0)


main_new = re.sub(r'(Appendix|Appendices)~\\ref\{([^}]*)\}(\s*and~\\ref\{([^}]*)\})?', appendix_phrase, main_new)


def to_supp(m):
    cmd, key = m.group(1), m.group(2)
    if key not in moved_labels:
        return m.group(0)
    return f'({supp_lab[key]})' if cmd == 'eqref' else supp_lab[key]


main_new = re.sub(r'\\(ref|eqref)\{([^}]*)\}', to_supp, main_new)
left = [k for k in moved_labels if re.search(r'\\(?:ref|eqref)\{' + re.escape(k) + r'\}', main_new)]
assert not left, left
# ---- MDPI "Supplementary Materials" statement, generated from the actual S-numbers
def rng(prefix, sec):
    ks = sorted({supp_lab[k] for k in moved_labels if k.startswith(prefix) and k in sec_of and sec_of[k] == sec},
                key=lambda x: int(x[1:]))
    return ks[0] if len(ks) == 1 else (f'{ks[0]}--{ks[-1]}' if ks else '')
sec_of = {}
for k in MOVE:
    for l in re.findall(r'\\label\{([^}]*)\}', blocks[k]):
        sec_of[l] = k
items = []
for k, desc in zip(MOVE, ['additional dataset characterisation', 'robustness checks',
                          'hyperparameter optimisation and implementation details', 'supplementary results']):
    parts = [p for p in (('Tables~' if '--' in rng('tab:', k) else 'Table~') + rng('tab:', k) if rng('tab:', k) else '',
                         ('Figures~' if '--' in rng('fig:', k) else 'Figure~') + rng('fig:', k) if rng('fig:', k) else '') if p]
    items.append(f'{desc} (Section~{supp_lab[k]}' + (', ' + ', '.join(parts) if parts else '') + ')')
stmt = ('\\supplementary{The following supporting information can be downloaded at: \\linksupplementary{s1}, '
        'Supplementary Material: ' + '; '.join(items) + '.}')
old_stmt = '%\\supplementary{The following supporting information can be downloaded at:  \\linksupplementary{s1}, Figure S1: title; Table S1: title; Video S1: title.}'
assert main_new.count(old_stmt) == 1
main_new = main_new.replace(old_stmt, stmt)

open(MAIN, 'w', encoding='utf-8').write(main_new)
build('template')
print('statement:', stmt)

print('moved labels:', len(moved_labels))
print('S-numbers:', {k: supp_lab[k] for k in sorted(moved_labels)})
print('nocite added:', only_supp)

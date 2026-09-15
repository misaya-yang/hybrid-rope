"""Prepare an unapplied manuscript proposal; never writes manuscript sources.

Outputs stay next to this script. The patch and exact before/after review are
generated from the current source so the author can approve a concrete change.
"""
from pathlib import Path
import difflib
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
ROOT = PAPER.parent
changes = {}


def read(path):
    return (PAPER/path).read_text()


def change(path, text):
    before = read(path) if (PAPER/path).exists() else ''
    if before != text:
        changes[path] = (before, text)


abstract = r'''Rotary position embeddings (RoPE) typically tie their frequency placement to a single base. We study frequency allocation as a way to improve model quality within a chosen context range. Controlled experiments show that redistributing interior frequencies improves performance even when the frequency range is fixed. Full sine--cosine geometry characterizes positional overlap and exposes rankings missed by cosine-only proxies. Coordinate interventions distinguish these positional directions from the model's learned use of them. We introduce TailSpline by formulating and solving a discrete allocation problem for the transition into the extended low-frequency tail. The resulting closed-form rule redistributes distance scaling while retaining the standard rotary operator, with no weight updates or calibration. TailSpline substantially outperforms MrRoPE-Pro on RULER at both intermediate and target extension lengths, with a small observed native-window trade-off. A complementary Cosh transport demonstrates allocation benefits through paired learning and extrapolation experiments. Together, the analysis and experiments make internal frequency placement a practical dimension of RoPE design.
'''
change('sections/00_abstract.tex', abstract)
change('title_abstract.txt', read('title_abstract.txt').splitlines()[0]+'\n\n'+abstract.replace('--', '–'))

intro = read('sections/01_intro.tex')
old = '''Complete rotary pairs expose finite-window direction overlap, while
coordinate and table interventions show how models use the supplied frequencies.'''
new = '''Complete rotary pairs expose finite-window direction overlap. Task gains
with lower positional effective rank, together with coordinate and table
interventions, distinguish this structure from its learned use.'''
assert intro.count(old) == 1
change('sections/01_intro.tex', intro.replace(old, new))

theory = read('sections/03_theory.tex')
corollary = re.search(r'\\begin\{corollary\}.*?\\end\{corollary\}', theory, flags=re.S).group()
head, _ = theory.split(r'\paragraph{Design implication.}', 1)
head = head.replace(r'\section{How Allocation Changes Positional Structure}',
                    r'\section{Positional Structure and Content Use}')
new_end = r'''\subsection{Shared positional dependence and content use}
Each rotary block remains invertible even when its positional functions
overlap. Shared dependence on distance can therefore coexist with distinct
content comparisons. A slow-block construction makes this distinction
explicit (Appendix~\ref{sec:content-coordinate-retention}), complementing
prior observations of positional and semantic frequency use
\citep{barbero2025round}.

The complete frozen Llama tables provide a concrete example. At $16/32$K,
TailSpline has full-pair effective ranks $8.28/10.08$, compared with
MrPro's $8.74/10.21$ under uniform separations, while improving clean
RULER by $3.39/11.72$ points (Table~\ref{tab:clean-length-main}).
The formulas, full precision and alternative separation measures are in
Appendix~\ref{sec:allocation-rank-quality}.
These gains establish the value of controlling distance dependence through
allocation even when normalized positional diversity decreases. The model's
use of the supplied frequencies determines their practical value.

\paragraph{Frequency changes and coordinate assignment.}
Changing frequencies also differs from reassigning a fixed spectrum.
In the no-alias interval $(0,\pi)$, position-independent invertible Q/K
maps preserve the full integer-position rotary kernel exactly if and only
if the frequency multisets agree (Appendix~\ref{sec:discrete-kernel-proof}).
This criterion complements the coordinate interventions and STRING's
structural characterization \citep{schenck2025string}.
'''
change('sections/03_theory.tex', head+new_end)

proofs = read('appendix/a1_proofs.tex')
anchor = '\\label{sec:discrete-kernel-proof}\n'
assert proofs.count(anchor) == 1
change('appendix/a1_proofs.tex', proofs.replace(anchor, anchor+corollary+'\n\n', 1))

method = read('sections/04_mature.tex')
old = '''Their unit sum allocates exactly $\\log s$ of extra span across the band. Beyond it, constant displacement $m=1$ preserves native adjacent gaps: the extra gap returns to zero. The remaining choice is how to reach this junction.'''
new = r'''Their unit sum allocates exactly $\log s$ of extra span across the band.
At pair $q$, a fixed exponent $m_q$ assigns wavelength growth $s^{m_q}$;
the endpoints fix the total span, while the interior profile determines
how this growth is distributed across channels.
The outer bands preserve two distance references: high frequencies retain
$R_{\omega^N}(d)$, while the low-frequency tail satisfies
$R_{\omega^N/s}(sd)=R_{\omega^N}(d)$. At the tail, constant displacement
$m=1$ preserves native adjacent gaps, so the extra gap returns to zero.'''
assert method.count(old) == 1
method = method.replace(old, new)
old = '''MrRoPE-Pro uses increasing increments, $m_q=q(q+1)/[n(n+1)]$. We instead smooth the transition into the fully interpolated tail, whose subsequent increments are zero:'''
new = r'''MrRoPE-Pro uses increasing increments, $m_q=q(q+1)/[n(n+1)]$.
We formulate a discrete tail-connection problem by penalizing variation
in the extra log gaps and their mismatch with the fully interpolated tail:'''
assert method.count(old) == 1
method = method.replace(old, new)
old = '''These guarantees solve the declared objectives; task value is tested below.'''
new = r'''TailSpline also redistributes scaling across the entire transition:
its wavelength multiplier is $s^{m_q^{\rm TS}}$, with
$m_q^{\rm TS}\ge m_q^{\rm Pro}$ at every interior pair. On the Llama
$s=4$ grid, the largest TailSpline/MrPro wavelength ratio is $1.766$.
Appendix~\ref{sec:allocation-scale-response} derives this ordering and
compares the response to scale with YaRN's frequency blend.'''
assert method.count(old) == 1
method = method.replace(old, new)
start = method.index(r'\paragraph{What the comparison isolates.}')
end = method.index(r'\paragraph{Installation.}', start)
method = method[:start]+r'''\paragraph{What the comparison isolates.}
TailSpline and MrPro retain the same outer bands and endpoints, so their
comparison tests complete internal allocations. At fixed support, total
log-frequency displacement is a statistic of $z$. The control $C$ also
matches this statistic to study the remaining shape difference
(Appendices~\ref{sec:tailspline-dose-control} and~\ref{sec:current-controls}).

'''+method[end:]
change('sections/04_mature.tex', method)

related = read('sections/02_related.tex')
old = '''Our integer-position corollary uses rotation spectra to separate a change in frequencies from a fixed change of basis.'''
new = r'''Our integer-position criterion uses rotation spectra to separate a frequency change from a fixed change of basis (Appendix~\ref{sec:discrete-kernel-proof}).'''
assert related.count(old) == 1
change('sections/02_related.tex', related.replace(old, new))

discussion = read('sections/05_discussion.tex')
old = '''learned-coordinate effects clarify how frequency changes interact with a
pretrained model.'''
new = old
assert discussion.count(old) == 1
change('sections/05_discussion.tex', discussion.replace(old, new))

appendix = (HERE/'proposed_appendix.tex').read_text()
change('appendix/a11_allocation_response.tex', appendix)
main = read('main.tex')
anchor = '\\input{appendix/a8_profile_diagnostics}\n'
assert main.count(anchor) == 1
change('main.tex', main.replace(anchor, anchor+'\\input{appendix/a11_allocation_response}\n'))

guide = read('appendix/a0_guide.tex')
anchor = '''Cosh extrapolation, native-window and natural-QA evaluations provide complementary results under their stated protocols.'''
replacement = r'''Allocation response and the rank--quality comparison are developed in Appendix~\ref{sec:allocation-response}. Cosh extrapolation, native-window and natural-QA evaluations provide complementary results under their stated protocols.'''
assert guide.count(anchor) == 1
change('appendix/a0_guide.tex', guide.replace(anchor, replacement))


def active_sources(overlays):
    found = {}
    def visit(path):
        if path in found:
            return
        text = overlays[path][1] if path in overlays else read(path)
        text = re.sub(r'(?<!\\)%.*', '', text)
        found[path] = text
        for target in re.findall(r'\\(?:input|include)\{([^}]+)\}', text):
            target = target if target.endswith('.tex') else target+'.tex'
            visit(target)
    visit('main.tex')
    return found


def reference_state(sources):
    text = '\n'.join(sources.values())
    labels = re.findall(r'\\label\{([^}]+)\}', text)
    refs = set(re.findall(r'\\(?:eqref|ref|pageref|autoref)\{([^}]+)\}', text))
    cited = set()
    for value in re.findall(r'\\cite\w*\*?(?:\[[^]]*\])*\{([^}]+)\}', text):
        cited.update(x.strip() for x in value.split(','))
    return set(labels), refs-set(labels), cited, {x for x in labels if labels.count(x)>1}


baseline = reference_state(active_sources({}))
proposed = reference_state(active_sources(changes))
bib_keys = set(re.findall(r'@\w+\{([^,]+),', read('refs/references.bib')))
assert not proposed[1]-baseline[1], proposed[1]-baseline[1]
assert not proposed[3]-baseline[3], proposed[3]-baseline[3]
assert not proposed[2]-bib_keys, proposed[2]-bib_keys
assert not re.search(r'\d', abstract)
assert re.search(r'\\title\{.*?\}', main, re.S).group() == re.search(r'\\title\{.*?\}', changes['main.tex'][1], re.S).group()

patch = ''.join(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
              fromfile='a/paper-2027/'+path if before else '/dev/null',
              tofile='b/paper-2027/'+path)) for path,(before,after) in changes.items())
(HERE/'candidate.patch').write_text(patch)

review = ['# 精确改稿增量（未应用）\n',
          '以下由当前稿件生成。每节给出实际删除与插入的LaTeX，新增附录另列完整文件。\n',
          '阅读说明与取舍见[审核入口](README.md)，全部理论身份见[理论汇总](THEORY_SYNTHESIS.md)。\n']
for path, (before, after) in changes.items():
    review.append(f'\n## {path}\n')
    if not before:
        review.append('完整新增内容见[附录草稿](proposed_appendix.tex)。\n')
    else:
        diff = ''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),n=3))
        review.append('```diff\n'+diff+'```\n')
(HERE/'MANUSCRIPT_DELTA.md').write_text('\n'.join(review))

stats = {'status':'UNAPPLIED_AUTHOR_REVIEW', 'changed_paths':list(changes),
         'abstract_words_before':len(read('sections/00_abstract.tex').split()),
         'abstract_words_after':len(abstract.split()), 'abstract_has_digits':False,
         'new_unresolved_references':sorted(proposed[1]-baseline[1]),
         'new_duplicate_labels':sorted(proposed[3]-baseline[3]),
         'unresolved_citations':sorted(proposed[2]-bib_keys),
         'base_sha256':{p:hashlib.sha256(b.encode()).hexdigest() for p,(b,a) in changes.items() if b},
         'proposed_sha256':{p:hashlib.sha256(a.encode()).hexdigest() for p,(b,a) in changes.items()},
         'protected_artifacts':{p:hashlib.sha256((PAPER/p).read_bytes()).hexdigest()
                                for p in ('main.pdf','exponent-allocation-source.zip')},
         'compiled':False,'page_count_verified':False,'patch_applied':False}
(HERE/'proposal_manifest.json').write_text(json.dumps(stats,indent=2)+'\n')
print(json.dumps({k:v for k,v in stats.items() if not k.endswith('sha256') and k!='protected_artifacts'},indent=2))

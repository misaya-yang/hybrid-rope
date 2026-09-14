"""Rebuild manuscript figures/tables from existing, recorded experiments.

This performs no model evaluation. Run from any directory:
    python paper-2027/figs/make_exponent_revision_figures.py
"""
from __future__ import annotations
import hashlib
import re
import argparse
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
OUT = PAPER / "figs"
PORTABLE_INPUTS = OUT / "figure_inputs.json"
BLUE, ORANGE, INK, GRID = "#0072B2", "#D55E00", "#20262B", "#DFE4E8"
TASK_NAMES = {
    "hotpotqa": "HotpotQA", "2wikimqa": "2WikiMQA", "qasper": "Qasper",
    "narrativeqa": "NarrativeQA", "multifieldqa_en": "MultiFieldQA",
}
RULER_NAMES = {
    "niah_single_1": "Single-1", "niah_single_2": "Single-2",
    "niah_single_3": "Single-3", "niah_multikey_1": "MultiKey-1",
    "niah_multikey_2": "MultiKey-2", "niah_multikey_3": "MultiKey-3",
    "niah_multivalue": "MultiValue", "niah_multiquery": "MultiQuery",
    "vt": "Variable tracking", "cwe": "Common words", "fwe": "Frequent words",
    "qa_1": "QA-1", "qa_2": "QA-2",
}
SOURCES = {
    "range": "paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.json",
    "llama": "data/curated/llama8b_causal_source_use_s42_20260714.json",
    "frozen": "data/curated/frozen_fixed_support_mature_20260823.json",
    "index": "paper-2027/research/attention-aware-retrofit/evidence/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RECEIPT_20260901.json",
    "qa": "docs/research/ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.json",
    "bm_definition": "docs/research/ROPE_MRPRO_BM_CANDIDATE_20260908.json",
    "olmo": "docs/research/ROPE_OLMO_BM_RESULT_20260908.json",
    "qwen3": "docs/research/ROPE_BM_TRANSFER_RESULT_20260908.json",
    "qwen7": "docs/research/ROPE_QWEN7_BM_RESULT_20260908.json",
    "slot_report": "paper-2027/research/attention-aware-retrofit/results/coupling-transfer/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md",
    "coadapt50": "paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md",
    "coadapt151": "paper-2027/research/attention-aware-retrofit/evidence/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULTS_20260823.json",
    "coordinate128": "paper-2027/research/attention-aware-retrofit/evidence/K128_COORDINATE_CONFIRMATION_RECEIPT_20260901.json",
    "coordinate32": "paper-2027/research/attention-aware-retrofit/evidence/K32_PAIRED_CROSSING_CONFIRMATION_RECEIPT_20260901.json",
}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.linewidth": .65,
})

def load_data():
    if all((ROOT / v).is_file() for v in SOURCES.values()):
        data = {k: (json.loads((ROOT / v).read_text()) if v.endswith(".json")
                    else (ROOT / v).read_text()) for k, v in SOURCES.items()}
    else:
        data = json.loads(PORTABLE_INPUTS.read_text())["data"]
    rows = data["qa"]["rows"]
    assert len(rows) == 778 == len({r["row_id"] for r in rows})
    for stratum, tasks in data["qa"]["by_stratum"].items():
        extended = stratum == "extended"
        for task, aggregate in tasks.items():
            subset = [r for r in rows if r["row_id"].startswith(task + "_")
                      and (r["input_tokens"] > 4096) == extended]
            assert len(subset) == aggregate["n"]
            for method, field in [("MrPro", "baseline"), ("BM", "candidate")]:
                assert abs(np.mean([r[field] for r in subset]) - aggregate[method]) < 1e-12
        for method in ("MrPro", "BM"):
            macro = np.mean([r[method] for r in tasks.values()])
            assert abs(macro - data["qa"]["task_equal_macro"][stratum][method]) < 1e-12
    for length, arms in data["index"]["scores"].items():
        for arm, values in arms.items():
            assert abs(np.mean(list(values["per_task"].values())) - values["macro"]) < 1e-12
    return data

def write_portable_inputs(data):
    """Export only the numbers/identities consumed by this figure builder."""
    if not all((ROOT / v).is_file() for v in SOURCES.values()):
        return
    keep = lambda value, keys: {k: value[k] for k in keys}
    compact = {
        "range": keep(data["range"], ["fixed_training_range", "target_matched_range"]),
        "llama": data["llama"],
        "index": keep(data["index"], ["tasks", "scores"]),
        "qa": keep(data["qa"], ["by_stratum", "task_equal_macro"]),
        "bm_definition": {},
        "slot_report": "\n".join(line for line in data["slot_report"].splitlines()
            if "All four 20-row" in line or "candidate's 1x PG-19" in line or "versus `3.104234`" in line),
        "coadapt50": "\n".join(r for r in data["coadapt50"].splitlines()
                                 if r.startswith("| Geo |") or r.startswith("| EVQ |")),
        "coadapt151": {"small_model_crossing": keep(data["coadapt151"]["small_model_crossing"],
            ["checkpoint_seeds", "anchors", "tail_tokens", "mean_tail_nll_two_seed_length1024"])},
        "olmo": {"experiments": {"seed_replication": {"arms": {}}}},
        "qwen3": {"models": {"qwen3": {"arms": {}}}},
        "qwen7": {"result": keep(data["qwen7"]["result"], ["baseline", "candidate"])},
    }
    compact["qa"]["rows"] = [keep(r, ["row_id", "input_tokens", "baseline", "candidate"])
                              for r in data["qa"]["rows"]]
    for name in ["target_olmo", "source_qwen_comparison"]:
        compact["bm_definition"][name] = keep(data["bm_definition"][name],
            ["low", "high", "N", "scale", "gain", "exponents_intended", "radix_increments"])
    for name, path in [("olmo", ["experiments", "seed_replication", "arms"]),
                       ("qwen3", ["models", "qwen3", "arms"])]:
        src, dst = data[name], compact[name]
        for key in path: src, dst = src[key], dst[key]
        for arm in ["MrPro", "MrProBM"]: dst[arm] = keep(src[arm], ["summary"])
    for name, contrast in [("coordinate32", "primary_physical_minus_index"),
                           ("coordinate128", "primary_index_minus_physical")]:
        compact[name] = keep(data[name], ["scores", "profile_selection_performed", "old_pilot_pooled", contrast])
        compact[name]["arm_identities"] = {
            arm: keep(data[name]["arm_identities"][arm],
                ["attention_scaling", "checkpoint_weight_sha256", "data_manifest_sha256",
                 "input_cell_sha256", "native_sha256_float32", "runner_sha256", "tokenizer_sha256"])
            for arm in ["physical_x", "normalized_index"]}
    sources = {k: {"file": Path(v).name, "sha256": hashlib.sha256((ROOT/v).read_bytes()).hexdigest()}
               for k,v in SOURCES.items()}
    PORTABLE_INPUTS.write_text(json.dumps({"schema": 1, "sources": sources, "data": compact}, indent=2)+"\n")

def build_coadaptation(data):
    loss50, ppl50 = np.zeros((2,2)), np.zeros((2,2))
    for i, weights in enumerate(["Geo","EVQ"]):
        for j, table in enumerate(["Geo","EVQ"]):
            rows = [r for r in data["coadapt50"].splitlines()
                    if r.startswith(f"| {weights} | {table} |")]
            assert len(rows) == 1
            cells = [c.strip() for c in rows[0].split("|")]
            loss50[i,j], ppl50[i,j] = float(cells[3]), float(cells[4])
    assert np.allclose(np.log(ppl50), loss50, rtol=0, atol=7e-4)
    crossing = data["coadapt151"]["small_model_crossing"]
    assert crossing["checkpoint_seeds"] == [137,256]
    assert crossing["anchors"] == 32 and crossing["tail_tokens"] == 128
    rows = crossing["mean_tail_nll_two_seed_length1024"]
    loss151 = np.array([[rows[w][t] for t in ["fmrope_derived","cosh_derived"]]
                       for w in ["fmrope_weights","anchored_cosh_weights"]])
    cmap = LinearSegmentedColormap.from_list("swap_penalty",["#FFFDFB","#F3C3A9",ORANGE])
    fig, axes = plt.subplots(1,2,figsize=(7.25,2.7),gridspec_kw={"wspace":.70})
    for ax,loss,values,title,labels,digits in [
        (axes[0],loss50,ppl50,"(a) 50M: perplexity",["Geo","Cosh"],2),
        (axes[1],loss151,loss151,"(b) 151.9M: tail NLL at 1K",["FMR-derived","Cosh-derived"],3)]:
        penalty = loss - np.diag(loss)[:,None]
        assert penalty[0,1] > 0 and penalty[1,0] > 0
        im=ax.imshow(penalty,cmap=cmap,vmin=0,vmax=2.5,aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(j,i,f"{values[i,j]:.{digits}f}",ha="center",va="center",
                        color="white" if penalty[i,j]>1.8 else INK,fontsize=13)
        ax.set(xticks=[0,1],xticklabels=labels,yticks=[0,1],
               yticklabels=["Geo-trained","Cosh-trained"] if ax is axes[0]
                          else ["FMR-trained","Cosh-trained"],
               xlabel="Installed runtime table")
        ax.set_title(title,loc="left",pad=10)
        ax.tick_params(length=0,pad=6)
        ax.set_xticks([-.5,.5,1.5],minor=True)
        ax.set_yticks([-.5,.5,1.5],minor=True)
        ax.grid(which="minor",color="white",linewidth=2)
        ax.tick_params(which="minor",length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
    axes[0].set_ylabel("Frozen weights")
    fig.subplots_adjust(left=.14,right=.875,bottom=.23,top=.84)
    cax=fig.add_axes([.91,.23,.014,.61])
    colorbar=fig.colorbar(im,cax=cax,ticks=[0,1,2])
    colorbar.set_label(r"Swap penalty ($\Delta$NLL)",fontsize=8)
    colorbar.ax.tick_params(labelsize=8,length=2)
    finish(fig,"fig_weight_table_crossing")

def finish(fig, name):
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight", pad_inches=.04)
    fig.savefig(OUT / (name + ".png"), dpi=220, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)

def axis_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=.5)
    ax.set_axisbelow(True)

def build_overview(data):
    """Fixed support only: the training window is an observed point, not an inset."""
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.45))
    ax = axes[0]
    k, K, tau = np.arange(32), 32, 4.
    u = (k + .5) / K
    q = 1 - np.arcsinh((1-u)*np.sinh(tau))/tau
    z = (q-q[0])/(q[-1]-q[0]); geom = k/(K-1)
    assert np.all(z <= geom+1e-12)
    for xs, y, color, marker in [(geom,.7,BLUE,"o"),(z,.25,ORANGE,"D")]:
        ax.hlines(y,0,1,color=GRID,lw=1)
        ax.scatter(xs,np.full(K,y),color=color,s=12,marker=marker,zorder=3)
        ax.scatter([0,1],[y,y],s=30,facecolor="white",edgecolor=INK,zorder=4)
    ax.text(.03,.82,"Uniform allocation",color=BLUE,fontsize=9)
    ax.text(.03,.37,"Redistributed interiors",color=ORANGE,fontsize=9)
    ax.set(xlim=(-.04,1.04),ylim=(0,1),xlabel="Normalized exponent",xticks=[0,.5,1])
    ax.set_yticks([]);ax.spines[["left","right","top"]].set_visible(False)
    ax.set_title("(a) Same endpoints and pair count",loc="left")
    lengths=[256,512,1024,2048]; block=data["range"]["fixed_training_range"]
    vals=np.array([[block[str(L)]["seed_values"][str(seed)] for L in lengths] for seed in [42,137,256]])
    assert np.all(vals[:,0]>0) and np.all(vals[:,1:]<0)
    means=vals.mean(0)
    for L,mean in zip(lengths,means):
        assert abs(mean-block[str(L)]["mean_nll_difference"])<1e-8
    ax=axes[1];ax.axhline(0,color=INK,lw=.8)
    for i,v in enumerate(vals):ax.plot(range(4),v,color="#A9B0B7",lw=1,marker="o",ms=3,label="Three seeds" if i==0 else None)
    ax.plot(range(4),means,color=ORANGE,lw=2,marker="D",ms=4,label="Seed mean")
    ax.set(xticks=range(4),xticklabels=["1x","2x","4x","8x"],ylim=(-.57,.12),
           xlabel="Evaluation / training length",ylabel="Cosh minus Geo tail NLL")
    ax.set_title("(b) Paired fixed-support intervention",loc="left")
    ax.legend(frameon=False,fontsize=8,loc="lower right");axis_style(ax)
    fig.subplots_adjust(left=.035,right=.985,bottom=.23,top=.86,wspace=.52)
    finish(fig,"fig_evidence_overview")


def build_compatibility(data):
    """Three controls in four readable panels; each metric has its own axis."""
    fig,axes=plt.subplots(2,2,figsize=(7.25,4.3),gridspec_kw={"wspace":.52,"hspace":.95})
    ax=axes[0,0];lengths=[512,1024,2048]
    for key,color,marker,label in [("fixed_training_range",ORANGE,"D","Base kept at 256"),("target_matched_range",BLUE,"o","Base = eval. length")]:
        vals=np.array([[data["range"][key][str(L)]["seed_values"][str(seed)] for L in lengths] for seed in [42,137,256]])
        for v in vals:ax.plot(range(3),v,color=color,alpha=.25,lw=.8)
        ax.plot(range(3),vals.mean(0),color=color,marker=marker,lw=1.7,ms=3,label=label)
    ax.axhline(0,color=INK,lw=.7)
    ax.set(xticks=range(3),xticklabels=["2x","4x","8x"],xlabel="Eval. / train length",ylabel="Cosh minus Geo tail NLL",ylim=(-.57,.83))
    ax.set_title("(a) Change evaluation range",loc="left")
    ax.legend(frameon=False,fontsize=8,loc="upper left");axis_style(ax)
    ax=axes[0,1]
    rows=data["coadapt151"]["small_model_crossing"]["mean_tail_nll_two_seed_length1024"]
    vals=np.array([[rows[w][t] for t in ["fmrope_derived","cosh_derived"]] for w in ["fmrope_weights","anchored_cosh_weights"]])
    penalties=vals-np.diag(vals)[:,None]
    assert penalties[0,1]>0 and penalties[1,0]>0
    ax.imshow(penalties,cmap="Oranges",vmin=0,vmax=2.5,aspect="auto")
    for i in range(2):
        for j in range(2):ax.text(j,i,f"{vals[i,j]:.3f}",ha="center",va="center",fontsize=11,color="white" if penalties[i,j]>1.5 else INK)
    ax.set(xticks=[0,1],xticklabels=["Geo","Cosh"],yticks=[0,1],yticklabels=["Geo","Cosh"],xlabel="Runtime table",ylabel="Trained weights")
    ax.set_title("(b) Cross the tables: tail NLL",loc="left")
    ax.tick_params(length=0)
    for sp in ax.spines.values():sp.set_visible(False)
    report=data["slot_report"]
    match=re.search(r"candidate's 1x PG-19 NLL was `([0-9.]+)`,\nversus `([0-9.]+)`",report)
    assert match, "missing OLMo permutation result"
    perm,ref=map(float,match.groups())
    qref=float(re.search(r"zero, versus macro `([0-9.]+)`",report).group(1))*100
    assert abs(ref-3.104234)<1e-8 and abs(perm-6.864926)<1e-8 and qref==70
    for ax,values,title,xlabel,maximum,digits in [
        (axes[1,0],[ref,perm],"(c) Reassign slots: OLMo","PG-19 tail NLL (lower is better)",8,3),
        (axes[1,1],[qref,0],"(d) Reassign slots: Qwen","64K task macro (%) (higher is better)",100,0)]:
        ax.barh([0,1],values,color=[BLUE,ORANGE],height=.55)
        ax.set(yticks=[0,1],yticklabels=["Reference","Permuted"],xlabel=xlabel,xlim=(0,maximum));ax.invert_yaxis()
        ax.xaxis.label.set_size(9)
        for y,v in enumerate(values):ax.text(v+maximum*.025,y,f"{v:.{digits}f}",va="center",fontsize=9)
        ax.set_title(title,loc="left");ax.spines[["top","right"]].set_visible(False)
    fig.subplots_adjust(left=.13,right=.96,bottom=.14,top=.92)
    finish(fig,"fig_allocation_compatibility")


def write_main_qa_table(data):
    qa=data["qa"]["by_stratum"]["extended"]
    macro=data["qa"]["task_equal_macro"]["extended"]
    rows=[r"\begin{table}[!ht]",r"\centering\small",
          r"\caption{\textbf{A concrete redistribution improves natural QA.} Frozen OLMo, static $s=4$, whole-response token F1 (\%). Both methods share the declared band, gain and inputs. The $631$ inputs exceed the $4$K native window; task means are equally weighted. Differences use unrounded scores. The paired, task-stratified $95\%$ interval for the macro difference is $[1.32,6.29]$ points.}",
          r"\label{tab:bm-qa-main}",r"\begin{tabular}{@{}lrrrr@{}}",r"\toprule",
          r"Task & $n$ & MrRoPE-Pro & BM & Difference\\",r"\midrule"]
    for task,name in TASK_NAMES.items():
        a,b=100*qa[task]['MrPro'],100*qa[task]['BM']
        rows.append(f"{name} & {qa[task]['n']} & {a:.2f} & {b:.2f} & {b-a:+.2f}"+r" \\")
    assert sum(qa[t]['n'] for t in TASK_NAMES)==631
    ci=np.array(macro['paired_bootstrap_95_interval'])*100
    assert np.allclose(ci,[1.3160525890820804,6.291977669603137])
    rows += [r"\midrule",f"Task-equal mean & 631 & {100*macro['MrPro']:.2f} & {100*macro['BM']:.2f} & {100*macro['delta']:+.2f}"+r" \\",r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    (PAPER/'tables/table_bm_qa_main.tex').write_text('\n'.join(rows)+'\n')

def build_llama(data):
    # Published rounded PPL values from the matched adaptation report;
    # check consistency with independently stored rounded NLL.
    x = np.arange(3)
    native = np.array([6.817,108.958,991.475])
    evq = np.array([10.068,24.068,127.911])
    for i,L in enumerate([8192,16384,32768]):
        row = data["llama"]["temporal_nll"][str(L)]
        assert abs(np.log(native[i])-row["native"]) < 6e-5
        assert abs(np.log(evq[i])-row["evq"]) < 6e-5
    fig,ax=plt.subplots(figsize=(6.25,2.65))
    ax.set_yscale("log")
    ax.plot(x,native,"o-",color=BLUE,lw=1.8,ms=5,label="Native-LoRA")
    ax.plot(x,evq,"D-",color=ORANGE,lw=1.8,ms=5,label="EVQ-Cosh-LoRA")
    for vals,color,offsets in [
        (native,BLUE,[(17,-4),(0,10),(-2,10)]),
        (evq,ORANGE,[(0,10),(0,-16),(0,-16)])]:
        for i,v in enumerate(vals):
            ax.annotate(f"{v:.2f}",(i,v),xytext=offsets[i],
                        textcoords="offset points",ha="center",color=color,fontsize=9)
    ax.set(xlim=(-.18,2.18),ylim=(4.5,1800),xticks=x,xticklabels=["8K","16K","32K"],
           xlabel="Evaluation length (adaptation length: 8K)",ylabel="Perplexity")
    ticks=[10,30,100,300,1000]
    ax.set_yticks(ticks,[str(t) for t in ticks])
    ax.legend(loc="upper left",frameon=False)
    axis_style(ax)
    fig.subplots_adjust(left=.12,right=.985,bottom=.23,top=.94)
    finish(fig,"fig_8b_length_curve")

def build_qa(data):
    task_order=list(TASK_NAMES)
    qa=data["qa"]["by_stratum"]["extended"]
    labels=[f"{TASK_NAMES[t]}  (n={qa[t]['n']})" for t in task_order]+["Task-equal mean"]
    baseline=[100*qa[t]["MrPro"] for t in task_order]
    candidate=[100*qa[t]["BM"] for t in task_order]
    baseline.append(100*data["qa"]["task_equal_macro"]["extended"]["MrPro"])
    candidate.append(100*data["qa"]["task_equal_macro"]["extended"]["BM"])
    fig,ax=plt.subplots(figsize=(7.25,2.55))
    y=np.arange(6)
    ax.hlines(y,baseline,candidate,color="#ADB5BB",lw=1.5)
    ax.scatter(baseline,y,s=26,color=BLUE,label="MrRoPE-Pro",zorder=3)
    ax.scatter(candidate,y,s=28,color=ORANGE,marker="D",label="Boundary-matched",zorder=3)
    for i,(a,b) in enumerate(zip(baseline,candidate)):
        ax.annotate(f"{a:.2f}",(a,i),xytext=(-7,0),textcoords="offset points",
                    ha="right",va="center",color=BLUE,fontsize=8.5)
        ax.annotate(f"{b:.2f}",(b,i),xytext=(7,0),textcoords="offset points",
                    ha="left",va="center",color=ORANGE,fontsize=8.5)
    ax.axhline(4.5,color=GRID,lw=.8)
    ax.set(yticks=y,yticklabels=labels,xlim=(5,44),ylim=(5.5,-.6),
           xlabel="Whole-response token F1 (%)")
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="y",length=0)
    ax.grid(axis="x",color=GRID,lw=.45)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center",bbox_to_anchor=(.43,1.01),ncol=2,frameon=False)
    fig.subplots_adjust(left=.28,right=.985,bottom=.21,top=.87)
    finish(fig,"fig_bm_natural_qa")

def build_profiles(data):
    definition = data["bm_definition"]
    profiles = [definition["target_olmo"], definition["source_qwen_comparison"]]
    for profile, expected in zip(profiles, [(14, 32, 18), (23, 40, 17)]):
        assert (profile["low"], profile["high"], profile["N"]) == expected
        assert profile["scale"] == 4
        assert abs(profile["gain"] - (1 + .1*np.log(4))) < 1e-12
    fig,axes=plt.subplots(1,2,figsize=(7.25,2.35))
    for ax,profile,title in zip(axes,profiles,["OLMo: 18 transition steps","Qwen: 17 transition steps"]):
        N = profile["N"]
        q=np.arange(N+1,dtype=float)
        mr=q*(q+1)/(N*(N+1))
        bm=q*(q+1)*(3*N+2-2*q)/(N*(N+1)*(N+2))
        assert np.allclose([mr[0],bm[0],mr[-1],bm[-1]],[0,0,1,1])
        installed = np.asarray(profile["exponents_intended"])
        low, high = profile["low"], profile["high"]
        assert np.all(installed[:low + 1] == 0) and np.all(installed[high:] == 1)
        assert np.allclose(installed[low:high + 1], bm, rtol=0, atol=1e-14)
        assert np.allclose(np.diff(bm), profile["radix_increments"], rtol=0, atol=1e-14)
        ax.plot(q/N,q/N,"--",color="#78848D",label="MrRoPE-Uni")
        ax.plot(q/N,mr,"o-",ms=2.5,lw=1.3,color=BLUE,label="MrRoPE-Pro")
        ax.plot(q/N,bm,"D-",ms=2.5,lw=1.3,color=ORANGE,label="Boundary-matched")
        ax.set(xlabel=r"Position in transition band $q/N$",ylabel=r"Cumulative shift $m_q$",
               xlim=(0,1),ylim=(0,1))
        ax.set_title(title,loc="left")
        axis_style(ax)
    axes[0].legend(frameon=False,fontsize=8,loc="upper left")
    fig.subplots_adjust(left=.08,right=.985,bottom=.23,top=.86,wspace=.32)
    finish(fig,"fig_bm_exponent_profiles")

def write_tables(data):
    tasks=data["index"]["tasks"]
    s=data["index"]["scores"]
    lines=[r"\begin{table}[ht]",r"\centering\small",
        r"\caption{Qwen-$0.5$B RULER-13 per-task score (\%), $20$ examples per cell.}",
        r"\label{tab:index-full13}",r"\begin{tabular}{@{}lrrrrrr@{}}",r"\toprule",
        r"& \multicolumn{3}{c}{$32$K} & \multicolumn{3}{c}{$64$K}\\",
        r"Task & Native & Index & YaRN & Native & Index & YaRN\\",r"\midrule"]
    for t in tasks:
        vals=[100*s[L][arm]["per_task"][t] for L in ["32768","65536"]
              for arm in ["native","normalized_index","official_yarn"]]
        lines.append(RULER_NAMES[t]+" & "+" & ".join(f"{v:.2f}" for v in vals)+r" \\")
    lines.extend([r"\bottomrule",r"\end{tabular}",r"\end{table}"])
    (PAPER/"tables/table_index_full13.tex").write_text("\n".join(lines)+"\n")
    olmo=data["olmo"]["experiments"]["seed_replication"]["arms"]
    q3=data["qwen3"]["models"]["qwen3"]["arms"]
    models=[("OLMo-$1$B",olmo["MrPro"]["summary"],olmo["MrProBM"]["summary"]),
            ("Qwen-$3$B",q3["MrPro"]["summary"],q3["MrProBM"]["summary"]),
            ("Qwen-$7$B",data["qwen7"]["result"]["baseline"],data["qwen7"]["result"]["candidate"])]
    lines=[r"\begin{table}[ht]",r"\centering\small",
        r"\caption{All six-task BM/MrRoPE-Pro comparisons. Values are official RULER task scores (\%); model-specific sample counts are given in Table~\ref{tab:bm-models}.}",
        r"\label{tab:bm-full-tasks}",r"\begin{tabular}{@{}llrrrr@{}}",r"\toprule",
        r"& & \multicolumn{2}{c}{Short} & \multicolumn{2}{c}{Long}\\",
        r"Model & Task & MrPro & BM & MrPro & BM\\",r"\midrule"]
    for m,(name,base,cand) in enumerate(models):
        if m:lines.append(r"\midrule")
        lens=sorted(base["by_length"],key=int)
        tasks=list(base["by_length"][lens[0]]["task_accuracy"])
        for i,t in enumerate(tasks):
            vals=[100*a["by_length"][L]["task_accuracy"][t] for L in lens for a in (base,cand)]
            lines.append((name if i==0 else "")+" & "+RULER_NAMES[t]+" & "
                         +" & ".join(f"{v:.2f}" for v in vals)+r" \\")
    lines.extend([r"\bottomrule",r"\end{tabular}",r"\end{table}"])
    (PAPER/"tables/table_bm_tasks.tex").write_text("\n".join(lines)+"\n")
    lines=[r"\begin{table}[ht]",r"\centering\small",
        r"\caption{Natural QA under static $s=4$. Token F1 uses the whole generated response; EOS counts are recorded separately.}",
        r"\label{tab:bm-qa-complete}",r"\begin{tabular}{@{}llrrrrr@{}}",r"\toprule",
        r"Input length & Task & $n$ & MrPro F1 (\%) & BM F1 (\%) & MrPro EOS & BM EOS\\",r"\midrule"]
    for si,(stratum,pretty) in enumerate([("extended",r"$4$K--$16$K"),("within_native_length",r"$\le4$K")]):
        if si:lines.append(r"\midrule")
        for i,(t,r) in enumerate(data["qa"]["by_stratum"][stratum].items()):
            lines.append((pretty if i==0 else "")+" & "+TASK_NAMES[t]+f" & {r['n']} & {100*r['MrPro']:.2f} & {100*r['BM']:.2f} & {r['MrPro_eos']} & {r['BM_eos']}"+r" \\")
    lines.extend([r"\bottomrule",r"\end{tabular}",r"\end{table}"])
    (PAPER/"tables/table_bm_qa_all.tex").write_text("\n".join(lines)+"\n")
    c32,c128=data["coordinate32"],data["coordinate128"]
    for receipt in [c32,c128]:
        a,b=[receipt["arm_identities"][k] for k in ["physical_x","normalized_index"]]
        for field in ["attention_scaling","checkpoint_weight_sha256","data_manifest_sha256",
                      "input_cell_sha256","native_sha256_float32","runner_sha256","tokenizer_sha256"]:
            assert a[field]==b[field],field
        assert receipt["profile_selection_performed"] is False
        assert receipt["old_pilot_pooled"] is False
    entries=[]
    for L in ["32768","65536"]:
        contrast=c32["primary_physical_minus_index"][L]
        ci=contrast["paired_stratified_ci975"]
        entries.append(("Qwen-$0.5$B","$32$K" if L=="32768" else "$64$K",
                        c32["scores"][L],[-100*ci[1],-100*ci[0]]))
    entries.append(("Gemma-$2$B","$16$K",c128["scores"],
                    [100*x for x in c128["primary_index_minus_physical"]["paired_task_stratified_ci95"]]))
    lines=[r"\begin{table}[ht]",r"\centering\small",
           r"\caption{Independent-input placement comparisons, $80$ examples per task and four tasks per panel. Intervals are for index minus direct-gap, in percentage points: $97.5\%$ for each Qwen length (Bonferroni over two lengths), $95\%$ for Gemma.}",
           r"\label{tab:coordinate-confirmation}",r"\begin{tabular}{@{}llrrl@{}}",r"\toprule",
           r"Model & Length & Direct-gap (\%) & Index (\%) & Difference interval\\",r"\midrule"]
    def signed_display(value):
        return format(Decimal(str(round(value, 10))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "+.2f")
    for model,L,scores,ci in entries:
        for values in scores.values():
            assert abs(np.mean(list(values["per_task"].values()))-values["macro"])<1e-12
        a,b=[100*scores[k]["macro"] for k in ["physical_x","normalized_index"]]
        lines.append(f"{model} & {L} & {a:.2f} & {b:.2f} & $[{signed_display(ci[0])},{signed_display(ci[1])}]$"+r" \\")
    lines.extend([r"\bottomrule",r"\end{tabular}",r"\end{table}"])
    (PAPER/"tables/table_coordinate_confirmation.tex").write_text("\n".join(lines)+"\n")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--main-only",action="store_true",help="Rebuild discovery figures and main QA table only")
    args=parser.parse_args()
    data=load_data()
    write_portable_inputs(data)
    build_overview(data)
    build_compatibility(data)
    write_main_qa_table(data)
    if args.main_only:
        receipt_path=OUT/"exponent_revision_source_receipt.json"
        receipt=json.loads(receipt_path.read_text())
        receipt["sources"]=json.loads(PORTABLE_INPUTS.read_text())["sources"]
        receipt["checks"].update({"fixed_support_training_window_included":True,
            "slot_assignment_report_values_verified":True,"main_qa_table_recomputed":True})
        receipt["execution"]="Replotting recorded outcomes and generating the main QA table only; no model execution."
        receipt_path.write_text(json.dumps(receipt,indent=2)+"\n")
        print("Built fixed-support and compatibility figures plus QA table; source values checked.")
        return
    build_llama(data)
    build_qa(data)
    build_profiles(data)
    build_coadaptation(data)
    write_tables(data)
    receipt={"sources":json.loads(PORTABLE_INPUTS.read_text())["sources"],
             "checks":{"qa_rows":778,"qa_long_rows":631,"qa_short_rows":147,
                       "qa_aggregate_recomputed":True,"index_macro_recomputed":True,
                       "llama_ppl_nll_rounding_consistent":True,
                       "bm_profiles_match_recorded_definitions":True,
                       "coordinate_identity_and_macros_verified":True,
                       "coadaptation_sources_cross_checked":True},
             "execution":"Plotting and recomputing stored scores only; no model execution."}
    (OUT/"exponent_revision_source_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("Rebuilt discovery and supporting figures plus main/appendix tables from verified stored scores.")

if __name__=="__main__":
    main()

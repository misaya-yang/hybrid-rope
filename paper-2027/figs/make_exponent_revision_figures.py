"""Rebuild manuscript figures/tables from existing, recorded experiments.

This performs no model evaluation. Run from any directory:
    python paper-2027/figs/make_exponent_revision_figures.py
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper-2027"
OUT = PAPER / "figs"
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
    data = {k: (json.loads((ROOT / v).read_text()) if v.endswith(".json")
                else (ROOT / v).read_text()) for k, v in SOURCES.items()}
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
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.7),
                             gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    k, K, tau = np.arange(32), 32, 4.
    u = (k + .5) / K
    q = 1 - np.arcsinh((1-u)*np.sinh(tau))/tau
    z = (q-q[0])/(q[-1]-q[0])
    geom = k/(K-1)
    assert np.all(np.diff(z) > 0) and z[0] == 0 and z[-1] == 1
    ax.hlines([.67, .28], 0, 1, color=GRID, lw=1)
    for i in range(3, 30, 4):
        ax.plot([geom[i], z[i]], [.65, .30], color="#CCD1D5", lw=.8)
    for xs, y, color, marker in [(geom,.67,BLUE,"o"),(z,.28,ORANGE,"D")]:
        ax.scatter(xs, np.full(K,y), color=color, s=16, marker=marker,
                   edgecolor="white", linewidth=.35, zorder=3)
        ax.scatter([0,1],[y,y],s=35,facecolor="white",edgecolor=INK,zorder=4)
    ax.annotate("", (1,.96),(0,.96), arrowprops={"arrowstyle":"<->","lw":.8})
    ax.text(.5,1.04,"Same endpoints and 32 rotary pairs",ha="center",fontsize=9)
    ax.text(.03,.79,"Uniform exponent spacing",color=BLUE,fontsize=10)
    ax.text(.03,.40,"EVQ-Cosh reallocation",color=ORANGE,fontsize=10,
            bbox={"facecolor":"white","edgecolor":"none","pad":.2})
    ax.set(xlim=(-.04,1.04),ylim=(.05,1.19),xlabel=r"Normalized exponent $z_k$")
    ax.set_yticks([])
    ax.set_xticks([0,.25,.5,.75,1])
    ax.spines[["left","right","top"]].set_visible(False)
    ax.set_title("(a) Same range, different allocation",loc="left",pad=13)
    ax = axes[1]
    lengths = [256,512,1024,2048]
    x = np.arange(4)
    fixed = data["range"]["fixed_training_range"]
    seeds = sorted(fixed["256"]["seed_values"],key=int)
    vals = np.array([[fixed[str(L)]["seed_values"][s] for L in lengths] for s in seeds])
    means = vals.mean(0)
    assert np.all(vals[:,1:] < 0)
    ax.axhspan(-.55,0,color="#FAF0EB",zorder=0)
    ax.axhline(0,color=INK,lw=.8)
    for v in vals:
        ax.plot(x,v,color="#A6ADB3",lw=.9,marker="o",ms=2.5)
    ax.plot(x,means,color=ORANGE,lw=1.8,marker="D",ms=4.5,zorder=5)
    for i,v in enumerate(means):
        offset = (0,8) if i == 0 else (0,-15)
        ax.annotate(f"{v:+.3f}",(i,v),xytext=offset,textcoords="offset points",
                    ha="center",fontsize=8.5,color=ORANGE)
    ax.set(ylim=(-.55,.10),xticks=x,xticklabels=["256\n1x","512\n2x","1K\n4x","2K\n8x"],
           xlabel="Evaluation length / training length",
           ylabel="NLL difference\n(Cosh minus geometric)")
    ax.set_title("(b) Every tested extrapolation improves",loc="left",pad=13)
    ax.text(2.0,-.48,"9 / 9 seed–length comparisons",ha="center",fontsize=8.5,color=INK)
    axis_style(ax)
    fig.subplots_adjust(left=.04,right=.985,bottom=.24,top=.85,wspace=.46)
    finish(fig,"fig_evidence_overview")

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
           xlabel="Complete-output token F1 (%)")
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
    for model,L,scores,ci in entries:
        for values in scores.values():
            assert abs(np.mean(list(values["per_task"].values()))-values["macro"])<1e-12
        a,b=[100*scores[k]["macro"] for k in ["physical_x","normalized_index"]]
        lines.append(f"{model} & {L} & {a:.2f} & {b:.2f} & $[{ci[0]:+.2f},{ci[1]:+.2f}]$"+r" \\")
    lines.extend([r"\bottomrule",r"\end{tabular}",r"\end{table}"])
    (PAPER/"tables/table_coordinate_confirmation.tex").write_text("\n".join(lines)+"\n")

def main():
    data=load_data()
    build_overview(data)
    build_llama(data)
    build_qa(data)
    build_profiles(data)
    build_coadaptation(data)
    write_tables(data)
    receipt={"sources":{k:{"path":v,"sha256":hashlib.sha256((ROOT/v).read_bytes()).hexdigest()}
                         for k,v in SOURCES.items()},
             "checks":{"qa_rows":778,"qa_long_rows":631,"qa_short_rows":147,
                       "qa_aggregate_recomputed":True,"index_macro_recomputed":True,
                       "llama_ppl_nll_rounding_consistent":True,
                       "bm_profiles_match_recorded_definitions":True,
                       "coordinate_identity_and_macros_verified":True,
                       "coadaptation_sources_cross_checked":True},
             "execution":"Plotting and recomputing stored scores only; no model execution."}
    (OUT/"exponent_revision_source_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("Rebuilt five figures and four appendix tables from verified stored scores.")

if __name__=="__main__":
    main()

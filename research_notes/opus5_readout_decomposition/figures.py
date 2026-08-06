#!/usr/bin/env python3
"""Per-layer figures for the causal readout decomposition.

Every panel shows all five matched cases individually; nothing is reduced to a
single averaged curve, because n=5 and the cases disagree in informative ways.
Run ``decompose.py`` first (this script consumes ``curves.npz``).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
LAYERS = np.arange(32)
EVQ_C, GEO_C = "#1f6fb4", "#c2453a"
plt.rcParams.update(
    {"figure.dpi": 140, "font.size": 8.5, "axes.grid": True, "grid.alpha": 0.25}
)


def load():
    z = np.load(OUT / "curves.npz", allow_pickle=True)
    shas = [str(s) for s in z["shas"]]
    depths = [float(d) for d in z["depths"]]
    return z, shas, depths


def get(z, arm, sha, field, pos=0):
    return z[f"{arm}__{sha}__{field}"][pos]


def fig1_entry_and_retention(z, shas, depths):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharex=True)
    for pos, ax in enumerate(axes):
        for sha, dep in zip(shas, depths):
            ax.plot(LAYERS, get(z, "evq", sha, "delta_gold_centred", pos),
                    color=EVQ_C, lw=1.3, alpha=0.85, label="EVQ" if sha == shas[0] else None)
            ax.plot(LAYERS, get(z, "geo", sha, "delta_gold_centred", pos),
                    color=GEO_C, lw=1.1, alpha=0.7, ls="--",
                    label="Geo (control)" if sha == shas[0] else None)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvspan(22, 24, color="grey", alpha=0.12)
        ax.set_title(f"answer position {pos}" + (" (first token — generation-critical)" if pos == 0 else ""))
        ax.set_xlabel("decoder layer")
    axes[0].set_ylabel(r"gold $\tilde{\delta}_\ell$  (full $-$ gold-drop, median-centred)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle(
        "Causal gold-block evidence in the output basis, per layer — 5 matched 16K cases, "
        "grey band = layers 22–24 onset", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_entry_and_retention.png", bbox_inches="tight")
    plt.close(fig)


def fig2_qk_lag(z, shas, depths):
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    qk = z["qk_adv_hit"]
    ax.bar(LAYERS, qk, color="#8a8f98", alpha=0.55, width=0.8,
           label="per-layer QK advantage, EVQ$-$Geo mean hit@16 (32 heads)")
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("QK gold-block hit@16 advantage", color="#4a4f58")
    ax.axhline(0, color="k", lw=0.6)
    ax2 = ax.twinx()
    ax2.grid(False)
    for sha in shas:
        ax2.plot(LAYERS, get(z, "evq", sha, "delta_gold_centred"), color=EVQ_C, lw=1.2, alpha=0.8)
    mean_curve = np.mean([get(z, "evq", s, "delta_gold_centred") for s in shas], axis=0)
    ax2.plot(LAYERS, mean_curve, color="#0b3d66", lw=2.4, label=r"EVQ readout $\tilde{\delta}_\ell$ (mean, pos 0)")
    ax2.set_ylabel(r"gold $\tilde{\delta}_\ell$ in output basis", color=EVQ_C)
    ax.axvspan(11, 21, color="#e8b64c", alpha=0.16)
    ax.axvspan(22, 24, color="#4c9e6a", alpha=0.16)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.25)
    ax.text(16.0, lo * 0.72, "QK addressing\nestablished (L11–21)",
            fontsize=7.5, ha="center", va="top", color="#7a5c10")
    ax.text(27.5, lo * 0.72, "readout onset\n(L22–24)",
            fontsize=7.5, ha="center", va="top", color="#1f5c3a")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title("QK addressing precedes output-space readability by ~5–10 layers\n"
                 r"(instantaneous Spearman $-0.06$; cumulative $0.63$, KL vs cumulative $0.90$)",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_qk_readout_lag.png", bbox_inches="tight")
    plt.close(fig)


def fig3_rank(z, shas, depths):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    for ax, field, title in (
        (axes[0], "causal_delta_rank", "gold rank under the causal delta  $\\Delta z_\\ell$\n(registered rule; basis-shift invariant)"),
        (axes[1], "gold_rank_full", "gold rank under the dense readout  $z_\\ell^{full}$"),
    ):
        for sha, dep in zip(shas, depths):
            ax.semilogy(LAYERS, get(z, "evq", sha, field), color=EVQ_C, lw=1.3, alpha=0.85,
                        label="EVQ" if sha == shas[0] else None)
            ax.semilogy(LAYERS, get(z, "geo", sha, field), color=GEO_C, lw=1.1, alpha=0.65, ls="--",
                        label="Geo (control)" if sha == shas[0] else None)
        ax.axhline(1, color="k", lw=0.6)
        ax.set_xlabel("decoder layer")
        ax.set_ylabel("rank (log scale)")
        ax.set_title(title, fontsize=9)
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    axes[0].axvspan(22, 24, color="grey", alpha=0.12)
    axes[1].axvspan(25, 31, color="#c2453a", alpha=0.10)
    axes[1].text(28, axes[1].get_ylim()[1] * 0.3, "late-layer\ndemotion", fontsize=7.5,
                 ha="center", color="#8c2f26")
    fig.suptitle("Answer position 0 — the causal effect localises onto the gold token while the "
                 "readout moves it away", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_rank_trajectories.png", bbox_inches="tight")
    plt.close(fig)


def fig4_competition(shas):
    rows = [
        r
        for r in csv.DictReader(open(OUT / "final_layer_competition.csv"))
        if r["prompt_sha8"] in set(shas)  # matched pairs only, both arms aligned
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    ax = axes[0]
    width = 0.35
    for i, arm in enumerate(("evq", "geo")):
        sub = sorted(
            (r for r in rows if r["arm"] == arm and r["answer_position"] == "0"),
            key=lambda r: shas.index(r["prompt_sha8"]),
        )
        gaps = [float(r["gap_full"]) for r in sub]
        closed = [float(r["deficit_closed_by_gold_block"]) for r in sub]
        x = np.arange(len(sub)) + (i - 0.5) * width
        ax.bar(x, gaps, width, color=(EVQ_C if arm == "evq" else GEO_C), alpha=0.85,
               label=f"{arm.upper()} residual deficit")
        ax.bar(x, closed, width, bottom=gaps, color=(EVQ_C if arm == "evq" else GEO_C),
               alpha=0.35, hatch="///", edgecolor="w",
               label=f"{arm.upper()} closed by gold block")
    ax.set_xticks(np.arange(len(shas)))
    ax.set_xticklabels(shas, fontsize=7, rotation=20)
    ax.set_ylabel("logits behind the top competitor")
    ax.set_title("Final-layer competition, answer position 0\n"
                 "the gold block closes 3–10% of the deficit", fontsize=9)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for arm, c in (("evq", EVQ_C), ("geo", GEO_C)):
        fr = [[float(r["closure_fraction"]) for r in rows
               if r["arm"] == arm and r["answer_position"] == str(t) and r["closure_fraction"]]
              for t in range(3)]
        ax.plot(range(3), [np.median(f) if f else 0 for f in fr], "o-", color=c, lw=2, label=arm.upper())
        for t in range(3):
            ax.scatter([t] * len(fr[t]), fr[t], color=c, alpha=0.4, s=16)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["pos 0\n(first token)", "pos 1", "pos 2"])
    ax.set_ylabel("closure fraction  (deficit closed / deficit without block)")
    ax.set_title("The bottleneck is specific to the first answer token", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_competition.png", bbox_inches="tight")
    plt.close(fig)


def fig5_sharpening(z, shas, depths):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for sha in shas:
        axes[0].plot(LAYERS, get(z, "evq", sha, "kl_full_abl"), color=EVQ_C, lw=1.2, alpha=0.85)
        axes[0].plot(LAYERS, get(z, "geo", sha, "kl_full_abl"), color=GEO_C, lw=1.0, alpha=0.6, ls="--")
        axes[1].plot(LAYERS, get(z, "evq", sha, "entropy_full"), color=EVQ_C, lw=1.2, alpha=0.85)
        axes[2].semilogy(LAYERS, get(z, "evq", sha, "gold_prob_full"), color=EVQ_C, lw=1.2, alpha=0.85)
    axes[0].set_ylabel(r"KL$(P^{full}_\ell \| P^{drop}_\ell)$")
    axes[0].set_title("total causal effect on the output\ndistribution — peaks L27–28", fontsize=9)
    axes[1].set_ylabel("entropy of $P^{full}_\\ell$ (nats)")
    axes[1].set_title("the readout sharpens hard over the\nsame layers (9.5 → 3.5 nats)", fontsize=9)
    axes[2].set_ylabel("$P^{full}_\\ell$(gold)")
    axes[2].set_title("so gold probability peaks at L25–29\nand then falls ~17×", fontsize=9)
    for ax in axes:
        ax.set_xlabel("decoder layer")
        ax.axvspan(27, 31, color="#c2453a", alpha=0.10)
    fig.suptitle("Why the KL collapse is not attenuation of the gold signal: the distribution "
                 "concentrates on a competitor (EVQ, answer position 0)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_sharpening_vs_attenuation.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    z, shas, depths = load()
    fig1_entry_and_retention(z, shas, depths)
    fig2_qk_lag(z, shas, depths)
    fig3_rank(z, shas, depths)
    fig4_competition(shas)
    fig5_sharpening(z, shas, depths)
    for p in sorted(FIG.glob("*.png")):
        print("wrote", p.relative_to(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

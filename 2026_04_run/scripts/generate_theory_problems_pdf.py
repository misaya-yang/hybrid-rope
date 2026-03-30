"""
Generate: EVQ-Cosh 理论待解决问题 (April 2026)
Output: 2026_04_run/docs/THEORY_OPEN_PROBLEMS.pdf
"""

import os
import subprocess
import tempfile
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import black, HexColor
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    HRFlowable, KeepTogether, Table, TableStyle
)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
OUT_PDF = BASE / "docs" / "THEORY_OPEN_PROBLEMS.pdf"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# LaTeX formula rendering → PNG
# ──────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FORMULA_CACHE = {}
FORMULA_IDX = [0]

def latex_img(latex_str: str, fontsize: int = 13, dpi: int = 200,
              width: float = None) -> Image:
    """Render LaTeX string to a ReportLab Image flowable."""
    key = (latex_str, fontsize)
    if key in FORMULA_CACHE:
        path = FORMULA_CACHE[key]
    else:
        fig, ax = plt.subplots(figsize=(0.01, 0.01))
        ax.axis("off")
        text_obj = ax.text(
            0, 0, f"${latex_str}$",
            fontsize=fontsize, color="black",
            ha="left", va="baseline",
            transform=ax.transAxes
        )
        fig.savefig("/tmp/_measure.png", dpi=dpi, bbox_inches="tight",
                     pad_inches=0.03, transparent=True)
        plt.close(fig)

        # re-render with correct sizing
        fig, ax = plt.subplots(figsize=(0.01, 0.01))
        ax.axis("off")
        ax.text(0, 0, f"${latex_str}$",
                fontsize=fontsize, color="black",
                ha="left", va="baseline",
                transform=ax.transAxes)
        idx = FORMULA_IDX[0]
        FORMULA_IDX[0] += 1
        path = f"/tmp/_formula_{idx}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                     pad_inches=0.03, transparent=True)
        plt.close(fig)
        FORMULA_CACHE[key] = path

    img = Image(path)
    # Scale to fit
    if width:
        ratio = width / img.drawWidth
        img.drawWidth = width
        img.drawHeight *= ratio
    else:
        scale = 0.45
        img.drawWidth *= scale
        img.drawHeight *= scale
    return img


# ──────────────────────────────────────────────
# Styles
# ──────────────────────────────────────────────
styles = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle(
    "DocTitle", parent=styles["Title"],
    fontSize=18, leading=24, textColor=black,
    spaceAfter=6, alignment=TA_CENTER,
    fontName="Helvetica-Bold"
)

SUBTITLE_STYLE = ParagraphStyle(
    "DocSubtitle", parent=styles["Normal"],
    fontSize=11, leading=14, textColor=HexColor("#444444"),
    spaceAfter=16, alignment=TA_CENTER,
    fontName="Helvetica"
)

H1_STYLE = ParagraphStyle(
    "H1", parent=styles["Heading1"],
    fontSize=15, leading=20, textColor=black,
    spaceBefore=20, spaceAfter=8,
    fontName="Helvetica-Bold"
)

H2_STYLE = ParagraphStyle(
    "H2", parent=styles["Heading2"],
    fontSize=12, leading=16, textColor=black,
    spaceBefore=14, spaceAfter=6,
    fontName="Helvetica-Bold"
)

BODY_STYLE = ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontSize=10.5, leading=15, textColor=black,
    spaceAfter=8, alignment=TA_JUSTIFY,
    fontName="Helvetica"
)

BODY_BOLD = ParagraphStyle(
    "BodyBold", parent=BODY_STYLE,
    fontName="Helvetica-Bold"
)

LABEL_STYLE = ParagraphStyle(
    "Label", parent=BODY_STYLE,
    fontSize=9.5, leading=13,
    textColor=HexColor("#333333"),
    fontName="Helvetica-Oblique",
    spaceAfter=4
)

# ──────────────────────────────────────────────
# Content builder
# ──────────────────────────────────────────────
def build_pdf():
    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=A4,
        leftMargin=25*mm, rightMargin=25*mm,
        topMargin=25*mm, bottomMargin=25*mm
    )
    story = []
    W = doc.width  # usable width

    # ── Title ──
    story.append(Paragraph("EVQ-Cosh: Open Theoretical Problems", TITLE_STYLE))
    story.append(Paragraph("April 2026 Sprint — Theory Agenda", SUBTITLE_STYLE))
    story.append(HRFlowable(width="100%", thickness=1, color=black))
    story.append(Spacer(1, 10))

    # ════════════════════════════════════════════
    # Problem 1
    # ════════════════════════════════════════════
    story.append(Paragraph("Problem 1: Closing the Lagrange Multiplier in the Scaling Law", H1_STYLE))

    story.append(Paragraph("<b>Current status</b>", BODY_BOLD))
    story.append(Paragraph(
        "The paper's deployed scaling law is:", BODY_STYLE))
    story.append(latex_img(r"\tau^*(L) = \frac{d_{\mathrm{head}}}{\sqrt{L}}", fontsize=16))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "This is derived from a softmax transport variational objective:", BODY_STYLE))
    story.append(latex_img(
        r"\mathcal{F}(\tau) = S_{\chi^2}(\tau) - \lambda \, U(\tau, L)",
        fontsize=14))
    story.append(Spacer(1, 4))
    story.append(Paragraph("where:", BODY_STYLE))

    story.append(latex_img(
        r"S_{\chi^2}(\tau) = \frac{1}{M}\!\left[\frac{\sinh\tau \cdot \arctan(\sinh\tau)}{\tau^2} - 1\right]"
        r"\quad \text{(Pearson } \chi^2 \text{ stiffness)}",
        fontsize=12))
    story.append(Spacer(1, 4))
    story.append(latex_img(
        r"U(\tau, L) = \frac{M}{L}\int_0^1 q(L \, b^{-\phi})\,\rho_\tau(\phi)\,d\phi"
        r"\quad \text{(softmax transport utility)}",
        fontsize=12))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>The problem</b>", BODY_BOLD))
    story.append(Paragraph(
        "The Lagrange multiplier \u03bb remains a free parameter. Currently it is calibrated against "
        "empirical \u03c4* values from 99 training runs. The stiffness and utility terms together "
        "yield an L-exponent \u03b3 \u2208 [0.47, 0.51], which is consistent with the empirical "
        "\u03b3 = 0.500 \u00b1 0.03. However, \u03bb itself has no closed-form expression from first "
        "principles. This makes the scaling law semi-analytic rather than a parameter-free theorem.",
        BODY_STYLE))

    story.append(Paragraph("<b>What would a solution look like?</b>", BODY_BOLD))
    story.append(Paragraph(
        "A complete derivation would express \u03bb as a function of known quantities "
        "(d<sub>head</sub>, K, b, and the distance prior D(\u0394)), eliminating the need for "
        "empirical calibration. Specifically, this requires deriving the relative scale of the "
        "stiffness and utility terms from first principles, rather than fitting it post-hoc.",
        BODY_STYLE))

    story.append(Paragraph("<b>Possible approaches</b>", BODY_BOLD))
    story.append(Paragraph(
        "(a) Self-consistent constraint: require that the optimal \u03c4 simultaneously minimizes "
        "the collision functional under the exact kernel, not just the surrogate. This adds a "
        "second equation that may pin \u03bb. "
        "(b) Dimensional analysis on the full attention operator: the 1/L factor comes from the "
        "softmax Jacobian; the stiffness term has a known \u03c4-dependence. If both terms can be "
        "expressed in the same physical units (bits of positional information per channel), "
        "\u03bb may reduce to a dimensionless constant. "
        "(c) Large-K asymptotic expansion: in the limit K \u2192 \u221e (many frequency channels), "
        "the discrete sum becomes an integral and \u03bb may admit a saddle-point evaluation.",
        BODY_STYLE))

    story.append(Paragraph("<b>Reviewer pressure</b>", BODY_BOLD))
    story.append(Paragraph(
        "GPT reviewer (Weakness 1, Major): \"The \u03c4* law is not theoretically closed in the "
        "strong sense that some readers may infer... the paper gives a persuasive structured "
        "hypothesis plus validation, not a full derivation of the deployed formula.\" "
        "Closing \u03bb would upgrade the paper from semi-analytic to a genuine theorem.",
        BODY_STYLE))

    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#CCCCCC")))

    # ════════════════════════════════════════════
    # Problem 2
    # ════════════════════════════════════════════
    story.append(Paragraph(
        "Problem 2: Broadband Surrogate Validity Boundary", H1_STYLE))

    story.append(Paragraph("<b>Current status</b>", BODY_BOLD))
    story.append(Paragraph(
        "The entire EVQ derivation rests on a single approximation: replacing the exact "
        "phase-collision kernel with a two-parameter broadband surrogate:", BODY_STYLE))

    story.append(latex_img(
        r"K(\phi_1,\phi_2) \;\approx\; K_{\mathrm{app}}(\phi_1,\phi_2)"
        r" = \alpha\,\delta(\phi_1 - \phi_2) + \beta\,\min(\phi_1,\phi_2)",
        fontsize=12))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "where the exact kernel is:", BODY_STYLE))
    story.append(latex_img(
        r"K(\phi_1,\phi_2) = \int D(\Delta)\,\cos(\omega(\phi_1)\Delta)"
        r"\,\cos(\omega(\phi_2)\Delta)\,d\Delta",
        fontsize=12))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>The problem</b>", BODY_BOLD))
    story.append(Paragraph(
        "The surrogate is validated functionally (EVQ derived from K<sub>app</sub> reduces the "
        "exact-kernel collision score by 24\u201392% across 12 configurations). However, this "
        "validation assumes a uniform distance prior D(\u0394) \u221d 1/\u0394. The reviewer "
        "asks (Question 3): what happens under a heavy-tailed prior or an empirical "
        "attention-distance distribution measured from trained checkpoints? "
        "Additionally, the surrogate's R\u00b2 drops from >0.99 to ~0.95 at base=500K in the "
        "mid-frequency band. The validity boundary of K<sub>app</sub> is not characterized "
        "in terms of (L, b, d<sub>head</sub>).",
        BODY_STYLE))

    story.append(Paragraph("<b>What would a solution look like?</b>", BODY_BOLD))
    story.append(Paragraph(
        "A characterization of the (L, b) region where K<sub>app</sub> faithfully represents "
        "the exact kernel, ideally expressed as an inequality on a single control parameter "
        "(e.g., the collision parameter c = ln(L/2\u03c0)/ln(b)). Outside this region, a "
        "corrected surrogate or an alternative derivation would be needed.",
        BODY_STYLE))

    story.append(Paragraph("<b>Possible approaches</b>", BODY_BOLD))
    story.append(Paragraph(
        "(a) Measure attention-distance distributions from trained 454M/750M checkpoints at "
        "multiple context lengths, then recompute the exact kernel with D(\u0394) = empirical "
        "distribution instead of 1/\u0394. If EVQ still reduces the collision score, the "
        "surrogate is robust to the prior. "
        "(b) Analytically bound the approximation error ||K - K<sub>app</sub>|| as a function of "
        "(L, b) in an appropriate operator norm. "
        "(c) Extend the surrogate to three parameters (adding a quadratic term) to improve "
        "mid-band fidelity at high base values.",
        BODY_STYLE))

    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#CCCCCC")))

    # ════════════════════════════════════════════
    # Problem 3
    # ════════════════════════════════════════════
    story.append(Paragraph(
        "Problem 3: Modality-Dependent Correction Factor for DiT", H1_STYLE))

    story.append(Paragraph("<b>Current status</b>", BODY_BOLD))
    story.append(Paragraph(
        "On autoregressive text models, the scaling law works directly:", BODY_STYLE))
    story.append(latex_img(
        r"\tau^*_{\mathrm{AR}} = \frac{d_{\mathrm{head}}}{\sqrt{L}}",
        fontsize=14))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "On video DiT (bidirectional attention, diffusion), experiments show:", BODY_STYLE))
    story.append(latex_img(
        r"\tau^*_{\mathrm{DiT}} \approx 0.53 \times \tau^*_{\mathrm{AR}}",
        fontsize=14))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>The problem</b>", BODY_BOLD))
    story.append(Paragraph(
        "The 0.53 correction factor is empirical and post-hoc. A hand-waving decomposition "
        "into two factors exists: (1) bidirectional attention effectively doubles the useful "
        "context (\u00d70.71 from 1/\u221a2), and (2) diffusion noise attenuation adds another "
        "\u00d70.75. Together 0.71 \u00d7 0.75 \u2248 0.53. But this decomposition is not "
        "derived from the variational framework \u2014 it is a post-hoc rationalization.",
        BODY_STYLE))

    story.append(Paragraph("<b>What would a solution look like?</b>", BODY_BOLD))
    story.append(Paragraph(
        "A modality-aware version of Proposition 2 that takes the attention pattern (causal "
        "mask vs bidirectional) and the loss type (next-token vs diffusion MSE) as inputs, "
        "and outputs the correct \u03c4* without empirical correction. The softmax transport "
        "utility U(\u03c4, L) would need to be rederived for the bidirectional attention "
        "Jacobian, where the baseline is p<sub>0</sub> = 1/L for full attention instead of "
        "the causal triangular average.",
        BODY_STYLE))

    story.append(Paragraph("<b>Reviewer pressure</b>", BODY_BOLD))
    story.append(Paragraph(
        "GPT reviewer (Question 5): \"How stable is the reported modality correction factor "
        "across architectures, bases, and training lengths?\" Currently only one DiT "
        "architecture (129.6M, base=10K, T=32) was tested. If the factor changes substantially "
        "with architecture or base, the lack of a principled formula becomes a real limitation.",
        BODY_STYLE))

    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#CCCCCC")))

    # ════════════════════════════════════════════
    # Problem 4
    # ════════════════════════════════════════════
    story.append(Paragraph(
        "Problem 4: LoRA Phase Transition \u2014 Sharp Threshold Derivation", H1_STYLE))

    story.append(Paragraph("<b>Current status</b>", BODY_BOLD))
    story.append(Paragraph(
        "EVQ fails catastrophically under LoRA fine-tuning when rank r < d<sub>head</sub>/2. "
        "The LLaMA-8B experiment shows PPL jumps from 11.8 to 77.1 at r=16 (r/K = 25%). "
        "The theoretical explanation is that frozen weights create a coupling stiffness:", BODY_STYLE))

    story.append(latex_img(
        r"S_{\mathrm{total}} = S_{\chi^2}(\tau) + \Lambda_0\!\left(1 - \frac{r}{K}\right)"
        r"\frac{\tau^2}{d_{\mathrm{head}}}",
        fontsize=13))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "The coupling term has a different \u03c4-dependence (\u03c4\u00b2) than the intrinsic "
        "stiffness (\u03c4\u2074), causing a phase transition:", BODY_STYLE))

    story.append(latex_img(
        r"r < r_c = K = \frac{d_{\mathrm{head}}}{2} \;\Rightarrow\; \tau^* \to 0 "
        r"\;\;\text{(EVQ infeasible)}",
        fontsize=13))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>The problem</b>", BODY_BOLD))
    story.append(Paragraph(
        "The coupling coefficient \u039b<sub>0</sub> is calibrated from a single data point "
        "(LLaMA-8B r=16 PPL=77.1). The sharpness of the phase transition (r=48 still fails, "
        "r=64 suddenly works) is confirmed numerically but not derived analytically. "
        "Specifically: (a) can we derive \u039b<sub>0</sub> from the pretrained weight statistics "
        "(e.g., the Frobenius norm of W<sub>Q</sub>, W<sub>K</sub>)? (b) How does the critical "
        "rank r<sub>c</sub> depend on the number of training steps T and learning rate \u03b7? "
        "The SFT recovery timescale (coupling decays as e<sup>-T\u03b7\u03c3</sup>) is "
        "postulated but not measured.",
        BODY_STYLE))

    story.append(Paragraph("<b>What would a solution look like?</b>", BODY_BOLD))
    story.append(Paragraph(
        "A closed-form expression for \u039b<sub>0</sub> in terms of measurable weight "
        "statistics, and a prediction for the SFT recovery curve (how many steps of full "
        "fine-tuning at a given learning rate are needed to overcome the coupling). This would "
        "turn the LoRA limitation into a quantitative recommendation: \"use r \u2265 r<sub>c</sub> "
        "or fine-tune for at least T<sub>min</sub> steps.\"",
        BODY_STYLE))

    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#CCCCCC")))

    # ════════════════════════════════════════════
    # Problem 5
    # ════════════════════════════════════════════
    story.append(Paragraph(
        "Problem 5: Progressive Training Amplification Mechanism", H1_STYLE))

    story.append(Paragraph("<b>Current status</b>", BODY_BOLD))
    story.append(Paragraph(
        "Three-stage progressive training (512 \u2192 1024 \u2192 2048) shows EVQ's advantage "
        "grows monotonically:", BODY_STYLE))

    # Table for progressive results
    prog_data = [
        ["Stage", "L<sub>train</sub>", "Geo+YaRN\n@16K", "EVQ+YaRN\n@16K", "\u0394"],
        ["1", "512", "3.80", "2.48", "-34.6%"],
        ["2", "512\u21921024", "4.60", "2.21", "-52.0%"],
        ["3", "512\u21921024\u21922048", "13.17", "2.48", "-81.2%"],
    ]
    prog_table_data = []
    for row in prog_data:
        prog_table_data.append([
            Paragraph(cell, BODY_STYLE) for cell in row
        ])

    prog_table = Table(prog_table_data, colWidths=[W*0.1, W*0.28, W*0.2, W*0.2, W*0.15])
    prog_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#F0F0F0")),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(prog_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>The problem</b>", BODY_BOLD))
    story.append(Paragraph(
        "The amplification is driven by an asymmetry: Geo+YaRN degrades across stages "
        "(3.80 \u2192 13.17) while EVQ+YaRN remains flat (2.48 \u2192 2.21 \u2192 2.48). "
        "But there is no theoretical explanation for why geometric RoPE's representation "
        "degrades under progressive extension while EVQ's does not. "
        "Additionally, the tau strategy question is unresolved at the theoretical level: "
        "should tau be retargeted per stage (\u03c4 = d<sub>head</sub>/\u221aL<sub>current</sub>) "
        "or kept fixed at the initial stage's value? The paper uses retarget (Phase 17C), "
        "but EXP-4 at low token budgets showed delayed (fixed) winning by 3.5%.",
        BODY_STYLE))

    story.append(Paragraph("<b>Key sub-questions</b>", BODY_BOLD))
    story.append(Paragraph(
        "(a) Can we derive from the collision functional why Geo+YaRN PPL diverges under "
        "progressive extension? The hypothesis is that geometric allocation accumulates "
        "phase collisions as L<sub>train</sub> grows, and YaRN rescaling inherits and amplifies "
        "the suboptimal packing. But this needs a formal statement. "
        "(b) For the tau-retarget vs tau-freeze question: when tau changes between stages, the "
        "inv_freq vector changes, effectively perturbing the attention kernel. How many tokens "
        "does the model need to adapt to the new frequencies? Is there a principled "
        "warm-up schedule? "
        "(c) Is there a theoretical prediction for the amplification rate (i.e., should the "
        "advantage grow linearly, quadratically, or exponentially with the number of stages)?",
        BODY_STYLE))

    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#CCCCCC")))

    # ════════════════════════════════════════════
    # Problem 6
    # ════════════════════════════════════════════
    story.append(Paragraph(
        "Problem 6: Scaling Law Validity at L \u2265 4096", H1_STYLE))

    story.append(Paragraph("<b>Current status</b>", BODY_BOLD))
    story.append(Paragraph(
        "The scaling law \u03c4* = d<sub>head</sub>/\u221aL is validated for "
        "L \u2208 [256, 2048]. The diagnostic experiment (Exp1b) confirmed that all "
        "L-dependence in \u03c4* comes from softmax transport (the 1/L Jacobian), not "
        "from the discrete collision objective. Specifically, the static discrete objective's "
        "optimal \u03c4 is approximately constant across L:", BODY_STYLE))

    story.append(latex_img(
        r"\tau^*_{\mathrm{static}} \approx 2.80 \;\;\text{(constant across all } L \text{, "
        r"for } K=32,\; b=10000\text{)}",
        fontsize=12))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>The problem</b>", BODY_BOLD))
    story.append(Paragraph(
        "At L \u2265 4096, the formula predicts \u03c4* \u2264 1.0 (for d<sub>head</sub>=64), "
        "which is in the regime where EVQ barely departs from geometric RoPE. The theory "
        "predicts this is correct (EVQ converges back to geometric as L grows), but this "
        "has not been experimentally validated at L=4096 or L=8192 from scratch. "
        "The MLA experiment (L<sub>train</sub>=8192) used a reference length L=512 to set "
        "\u03c4=1.414 rather than the formula's prediction of \u03c4*(8192)=64/\u221a8192\u22480.71 "
        "\u2014 and it worked well. This raises the question: is the formula's prediction at "
        "large L actually correct, or does the \"effective L\" for tau selection differ from "
        "the actual training length?",
        BODY_STYLE))

    story.append(Paragraph("<b>Key sub-questions</b>", BODY_BOLD))
    story.append(Paragraph(
        "(a) At L=8192, should \u03c4 be set to d<sub>head</sub>/\u221a8192 \u2248 0.71, or is "
        "there a saturation effect where \u03c4 floors at some minimum value? "
        "(b) The MLA success with \u03c4=1.414 (reference L=512) at L<sub>train</sub>=8192 "
        "suggests the formula may need a bivariate correction \u03c4*(L, b) where base b "
        "modulates the effective L. "
        "(c) Can the softmax transport argument be extended to predict the exact L at which "
        "the formula breaks down (if it does)?",
        BODY_STYLE))

    story.append(Paragraph("<b>Experimental path</b>", BODY_BOLD))
    story.append(Paragraph(
        "Run a controlled tau sweep at L=4096 and L=8192 (125M model, quick) to directly "
        "measure where the empirical \u03c4* falls relative to the formula's prediction. "
        "If \u03c4*<sub>empirical</sub> > \u03c4*<sub>formula</sub>, the formula underestimates "
        "at large L and the bivariate correction hypothesis is supported.",
        BODY_STYLE))

    # ── Footer ──
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=black))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Document generated for EVQ-Cosh NeurIPS 2026 submission. "
        "Intended audience: advisors, collaborators, and AI assistants.",
        LABEL_STYLE))

    doc.build(story)
    print(f"PDF saved to: {OUT_PDF}")


if __name__ == "__main__":
    build_pdf()

"""Build every figure and table for the report from a finished run (no GPU needed).

    python make_report_assets.py --run runs/full --dest report

Writes report/figures/*.pdf|png, report/generated/*.tex (tables + \\newcommand numbers).

LLM disclosure: written with assistance from Claude (Anthropic), reviewed by the team.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical palette (fixed slot order) + one-hue sequential ramp for ordered lambda.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ_BLUE = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal: light -> dark = small -> large
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

MAIN_LAMBDA = 0.02
METHOD_STYLE = {  # color = training method, line style = training eps (secondary encoding)
    "none": dict(color=BLUE, marker="o", label="Standard"),
    "fgsm": dict(color=ORANGE, marker="s", label="FGSM-AT"),
    "pgd": dict(color=AQUA, marker="^", label="PGD-AT"),
}


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.label.set_color(INK2); ax.xaxis.label.set_color(INK2)
    ax.title.set_color(INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def mark_lambda(ax, lam, text=True):
    """Dashed reference lines at lambda/2 (no detector can be robust beyond it) and lambda.
    Labels sit just above the plot area so they never collide with data near 0 or 1."""
    import matplotlib.transforms as mtransforms

    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for x, lbl in [(lam / 2, r"$\lambda/2$"), (lam, r"$\lambda$")]:
        ax.axvline(x, color=MUTED, linestyle=(0, (3, 3)), linewidth=0.9, zorder=0)
        if text:
            ax.text(x, 1.01, lbl, transform=trans, color=INK2, fontsize=7, va="bottom", ha="center")


def save(fig, dest: Path, name: str):
    fig.savefig(dest / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(dest / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def with_clean(df: pd.DataFrame, attack: str) -> pd.DataFrame:
    """Rows for one attack plus the clean row at eps=0, sorted by eps."""
    part = df[(df.attack == attack) | (df.attack == "none")].copy()
    return part.sort_values("eps")


def label_for(row) -> str:
    if row["train_adv"] == "none":
        return "Standard"
    return f"{METHOD_STYLE[row['train_adv']]['label']} ($\\epsilon$={row['train_eps']:g})"


# --------------------------------------------------------------------------- figures
def fig_baseline(df, dest, controls=None):
    """H1 + H2: baseline detector under FGSM vs PGD, plus the noise / oracle controls if available."""
    base = df[(df.model == "std_l0.02")]
    series = [("fgsm", BLUE, "o", "-", "FGSM"), ("pgd", ORANGE, "s", "-", "PGD")]
    if controls is not None:
        ctl = controls[(controls.model == "std_l0.02") & (controls.eps <= df.eps.max() + 1e-9)]
        base = pd.concat([base, ctl], ignore_index=True)
        series += [("noise", MUTED, "x", (0, (2, 2)), r"random $\pm\epsilon$ noise (control)"),
                   ("oracle", INK2, "D", (0, (5, 2)), r"oracle (moves toward the other copy)")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9))
    panels = [("accuracy", "Detection accuracy"), ("TPR", "TPR (watermarked flagged)"),
              ("FPR", "FPR (clean flagged)"), ("evasion_asr", "Evasion attack success rate")]
    for ax, (col, title) in zip(axes, panels):
        for attack, color, marker, ls, lbl in series:
            part = with_clean(base, attack) if col != "evasion_asr" else base[base.attack == attack].sort_values("eps")
            ax.plot(part.eps, part[col], color=color, marker=marker, markersize=5, linewidth=2,
                    linestyle=ls, label=lbl)
        if col == "accuracy":
            ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
            ax.text(0.031, 0.5, "chance", color=INK2, fontsize=7, va="bottom", ha="right")
        mark_lambda(ax, MAIN_LAMBDA)
        ax.set_ylim(-0.02, 1.02); ax.set_xlabel(r"attack budget $\epsilon$ ($L_\infty$, pixel units)")
        ax.set_title(title, fontsize=9, pad=14)
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(rf"Standard ResNet50 detector, $\lambda$={MAIN_LAMBDA}", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, dest, "fig_h1_h2_baseline")


def fig_adv_training(df, dest):
    """H3: evasion ASR and robust accuracy for standard vs adversarially trained detectors."""
    sub = df[df.strength == MAIN_LAMBDA]
    models = sub.drop_duplicates("model")[["model", "train_adv", "train_eps"]]
    models = models.sort_values(["train_adv", "train_eps"], key=lambda s: s.map({"none": 0, "fgsm": 1, "pgd": 2}) if s.name == "train_adv" else s)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for col_i, attack in enumerate(["fgsm", "pgd"]):
        for row_i, (metric, ylab) in enumerate([("accuracy", "Robust accuracy"), ("evasion_asr", "Evasion ASR")]):
            ax = axes[row_i, col_i]
            for _, m in models.iterrows():
                st = METHOD_STYLE[m.train_adv]
                part = sub[sub.model == m.model]
                part = with_clean(part, attack) if metric == "accuracy" else part[part.attack == attack].sort_values("eps")
                ls = "-" if m.train_adv == "none" or m.train_eps >= 0.01 else (0, (4, 2))
                ax.plot(part.eps, part[metric], color=st["color"], marker=st["marker"], markersize=5,
                        linewidth=2, linestyle=ls, label=label_for(m))
            if metric == "accuracy":
                ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
            mark_lambda(ax, MAIN_LAMBDA)
            ax.set_ylim(-0.02, 1.02); ax.set_ylabel(ylab)
            ax.set_title(f"{attack.upper()} attack", fontsize=9, pad=14)
            if row_i == 1:
                ax.set_xlabel(r"attack budget $\epsilon$")
            style_axes(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, dest, "fig_h3_adv_training")


def fig_lambda(df, dest):
    """H4: watermark strength vs robustness (PGD) and fidelity."""
    sub = df[df.train_adv == "none"].copy()
    lams = sorted(sub.strength.unique())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for color, lam in zip(SEQ_BLUE[-len(lams):] if len(lams) <= 4 else SEQ_BLUE, lams):
        part = with_clean(sub[sub.strength == lam], "pgd")
        ax.plot(part.eps, part.accuracy, color=color, marker="o", markersize=5, linewidth=2,
                label=rf"$\lambda$={lam:g}")
        # where the theory says robustness must end for this lambda
        ax.plot([lam / 2], [np.interp(lam / 2, part.eps, part.accuracy)], marker="|", markersize=12,
                color=color, markeredgewidth=2)
    ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_ylim(-0.02, 1.02); ax.set_xlabel(r"PGD budget $\epsilon$"); ax.set_ylabel("Accuracy under PGD")
    ax.set_title(r"Robustness vs. watermark strength (tick = $\lambda/2$)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)

    ax = axes[1]
    fid = sub.drop_duplicates("strength").sort_values("strength")
    ax.plot(fid.strength, fid.wm_psnr, color=BLUE, marker="o", markersize=5, linewidth=2)
    for _, r in fid.iterrows():
        ax.annotate(f"{r.wm_psnr:.1f} dB", (r.strength, r.wm_psnr), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, color=INK2)
    ax.set_xscale("log"); ax.set_xticks(fid.strength); ax.set_xticklabels([f"{s:g}" for s in fid.strength])
    ax.set_xlabel(r"watermark strength $\lambda$"); ax.set_ylabel("PSNR of watermarked image (dB)")
    ax.set_title("Image fidelity", fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    save(fig, dest, "fig_h4_lambda")


def fig_examples(run: Path, dest: Path, model="std_l0.02", eps_show=(0.01, 0.03)):
    path = run / "results" / f"{model}_examples.npz"
    if not path.exists():
        return
    ex = np.load(path)
    n = min(4, ex["clean"].shape[0])
    cols = [("Original", ex["clean"], ex["p_marked_clean"]),
            (r"Watermarked ($\lambda$=0.02)", ex["marked"], ex["p_marked_marked"])]
    for e in eps_show:
        for a in ("fgsm", "pgd"):
            key = f"{a}_{e}"
            if key in ex:
                cols.append((f"{a.upper()} $\\epsilon$={e:g}", ex[key], ex[f"p_marked_{key}"]))
    fig, axes = plt.subplots(n, len(cols), figsize=(1.45 * len(cols), 1.6 * n))
    for j, (title, imgs, probs) in enumerate(cols):
        for i in range(n):
            ax = axes[i, j]
            ax.imshow(np.clip(imgs[i].transpose(1, 2, 0), 0, 1), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xlabel(f"P(wm)={probs[i]:.2f}", fontsize=7, color=INK2, labelpad=2)
            if i == 0:
                ax.set_title(title, fontsize=7, color=INK)
    fig.tight_layout()
    save(fig, dest, "fig_examples")


def fig_training(run: Path, dest: Path, df: pd.DataFrame):
    files = sorted((run / "results").glob("*_history.csv"))
    if not files:
        return
    meta = df.drop_duplicates("model").set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9))
    for f in files:
        name = f.name.replace("_history.csv", "")
        if name not in meta.index or meta.loc[name, "strength"] != MAIN_LAMBDA:
            continue
        h = pd.read_csv(f)
        m = meta.loc[name]
        st = METHOD_STYLE[m.train_adv]
        ls = "-" if m.train_adv == "none" or m.train_eps >= 0.01 else (0, (4, 2))
        lbl = label_for({"train_adv": m.train_adv, "train_eps": m.train_eps})
        axes[0].plot(h.epoch, h.val_accuracy, color=st["color"], linestyle=ls, marker=st["marker"],
                     markersize=4, linewidth=1.8, label=lbl)
        if "val_robust_accuracy" in h:
            axes[1].plot(h.epoch, h.val_robust_accuracy, color=st["color"], linestyle=ls,
                         marker=st["marker"], markersize=4, linewidth=1.8, label=lbl)
    axes[0].set_title("Validation accuracy (clean)", fontsize=9)
    axes[1].set_title(r"Validation robust accuracy at training $\epsilon$", fontsize=9)
    for ax in axes:
        ax.set_xlabel("epoch"); ax.set_ylim(0.3, 1.01); style_axes(ax)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, dest, "fig_training")


# --------------------------------------------------------------------------- tables / numbers
def pct(x):
    return "--" if pd.isna(x) else f"{100 * x:.1f}"


def table_clean(df, gen: Path):
    rows = df[df.attack == "none"].copy()
    rows["order"] = rows.train_adv.map({"none": 0, "fgsm": 1, "pgd": 2})
    rows = rows.sort_values(["strength", "order", "train_eps"])
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Detector & $\lambda$ & Acc. & TPR & FPR & Prec. \\", r"\midrule"]
    for _, r in rows.iterrows():
        lines.append(f"{label_for(r).replace('$', '$')} & {r.strength:g} & {pct(r.accuracy)} & "
                     f"{pct(r.TPR)} & {pct(r.FPR)} & {pct(r.precision)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (gen / "table_clean.tex").write_text("\n".join(lines) + "\n")


def table_robust(df, gen: Path, eps_cols=(0.005, 0.01, 0.02)):
    sub = df[df.strength == MAIN_LAMBDA].copy()
    sub["order"] = sub.train_adv.map({"none": 0, "fgsm": 1, "pgd": 2})
    models = sub.drop_duplicates("model").sort_values(["order", "train_eps"])
    head = " & ".join([f"F {e:g}" for e in eps_cols] + [f"P {e:g}" for e in eps_cols])
    lines = [r"\begin{tabular}{l" + "r" * (2 * len(eps_cols)) + "}", r"\toprule",
             r"Detector & " + head + r" \\", r"\midrule"]
    for _, m in models.iterrows():
        vals = []
        for a in ("fgsm", "pgd"):
            for e in eps_cols:
                hit = sub[(sub.model == m.model) & (sub.attack == a) & np.isclose(sub.eps, e)]
                vals.append(pct(hit.evasion_asr.iloc[0]) if len(hit) else "--")
        lines.append(f"{label_for(m)} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (gen / "table_robust.tex").write_text("\n".join(lines) + "\n")


def table_lambda(df, gen: Path, eps_cols=(0.0025, 0.005, 0.01, 0.02)):
    sub = df[df.train_adv == "none"]
    lams = sorted(sub.strength.unique())
    lines = [r"\begin{tabular}{rrrr" + "r" * len(eps_cols) + "}", r"\toprule",
             r"$\lambda$ & PSNR & SSIM & Clean & " + " & ".join(f"P {e:g}" for e in eps_cols) + r" \\",
             r"\midrule"]
    for lam in lams:
        part = sub[sub.strength == lam]
        clean = part[part.attack == "none"].iloc[0]
        vals = []
        for e in eps_cols:
            hit = part[(part.attack == "pgd") & np.isclose(part.eps, e)]
            vals.append(pct(hit.accuracy.iloc[0]) if len(hit) else "--")
        lines.append(f"{lam:g} & {clean.wm_psnr:.1f} & {clean.wm_ssim:.3f} & {pct(clean.accuracy)} & "
                     + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (gen / "table_lambda.tex").write_text("\n".join(lines) + "\n")


def table_worst(df, controls, gen: Path, eps_cols=(0.005, 0.01, 0.02)):
    """Accuracy under PGD vs. the oracle attacker (the proposition's worst case), lambda = 0.02."""
    if controls is None:
        return
    full = pd.concat([df, controls], ignore_index=True)
    sub = full[full.strength == MAIN_LAMBDA].copy()
    sub["order"] = sub.train_adv.map({"none": 0, "fgsm": 1, "pgd": 2})
    models = sub.drop_duplicates("model").sort_values(["order", "train_eps"])
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{$\\epsilon$={e:g}}}" for e in eps_cols)
    sub_head = " & ".join(["PGD & Orc."] * len(eps_cols))
    lines = [r"\begin{tabular}{l" + "rr" * len(eps_cols) + "}", r"\toprule",
             r"Detector & " + head + r" \\", r" & " + sub_head + r" \\", r"\midrule"]
    for _, m in models.iterrows():
        vals = []
        for e in eps_cols:
            for a in ("pgd", "oracle"):
                hit = sub[(sub.model == m.model) & (sub.attack == a) & np.isclose(sub.eps, e)]
                vals.append(pct(hit.accuracy.iloc[0]) if len(hit) else "--")
        lines.append(f"{label_for(m)} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (gen / "table_worst.tex").write_text("\n".join(lines) + "\n")


MODEL_TAGS = {"std_l0.02": "Base", "fgsm_e0.005_l0.02": "FatA", "fgsm_e0.01_l0.02": "FatB",
              "pgd_e0.005_l0.02": "PatA", "pgd_e0.01_l0.02": "PatB",
              "std_l0.005": "LamA", "std_l0.01": "LamB", "std_l0.05": "LamD"}
EPS_TAGS = [(0.0025, "A"), (0.005, "B"), (0.01, "C"), (0.02, "D"), (0.03, "E"),
            (0.025, "F"), (0.04, "G"), (0.05, "H")]


def numbers(df, run: Path, gen: Path, controls=None):
    """\\newcommand macros so the report text always matches the data.
    Name = <model tag><attack><metric><eps tag>, e.g. \\PatAPgdAsrB = PGD-AT(0.005) evasion ASR
    under PGD at eps=0.005; clean metrics are <model tag>Clean<metric>."""
    out = {}
    full = df if controls is None else pd.concat([df, controls], ignore_index=True)
    def get(model, attack, eps, col):
        hit = full[(full.model == model) & (full.attack == attack) & np.isclose(full.eps, eps)]
        return hit[col].iloc[0] if len(hit) else float("nan")
    metrics = [("Acc", "accuracy"), ("Asr", "evasion_asr"), ("Fasr", "forgery_asr"), ("Tpr", "TPR"),
               ("Fpr", "FPR"), ("Pair", "pair_accuracy")]
    for model, M in MODEL_TAGS.items():
        for mt, col in metrics:
            out[f"{M}Clean{mt}"] = pct(get(model, "none", 0, col))
        out[f"{M}WmPsnr"] = f"{get(model, 'none', 0, 'wm_psnr'):.1f}"
        out[f"{M}WmSsim"] = f"{get(model, 'none', 0, 'wm_ssim'):.3f}"
        for a, A in (("fgsm", "Fgsm"), ("pgd", "Pgd"), ("noise", "Noise"), ("oracle", "Oracle")):
            for e, tag in EPS_TAGS:
                for mt, col in metrics:
                    v = get(model, a, e, col)
                    if not pd.isna(v):
                        out[f"{M}{A}{mt}{tag}"] = pct(v)
                ps = get(model, a, e, "psnr")
                if not pd.isna(ps):
                    out[f"{M}{A}Psnr{tag}"] = f"{ps:.1f}"
    # short aliases used throughout the text
    out["BaseAcc"], out["BaseTPR"], out["BaseFPR"] = out["BaseCleanAcc"], out["BaseCleanTpr"], out["BaseCleanFpr"]
    out["BasePSNR"] = out["BaseWmPsnr"]
    meta_path = run / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        dev = meta.get("device", "?")
        for long, short in [("NVIDIA ", ""), (" Workstation Edition", ""), ("Max-Q ", ""),
                            (" MIG ", " (MIG "), ("GeForce ", "")]:
            dev = dev.replace(long, short)
        if "(MIG" in dev and not dev.endswith(")"):
            dev += " slice)"
        out["RunDevice"] = dev.replace("_", r"\_")
        out["RunEpochs"] = str(meta.get("epochs", "?"))
        total = sum(meta.get("timings_sec", {}).values()) / 3600
        out["RunHours"] = f"{total:.1f}"
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in out.items()]
    (gen / "numbers.tex").write_text("\n".join(lines) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/full")
    p.add_argument("--dest", default="report")
    args = p.parse_args(argv)
    run, dest = Path(args.run), Path(args.dest)
    figs, gen = dest / "figures", dest / "generated"
    figs.mkdir(parents=True, exist_ok=True); gen.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(run / "all_results.csv")
    controls = pd.read_csv(run / "controls.csv") if (run / "controls.csv").exists() else None
    plt.rcParams.update({"font.size": 8, "axes.titleweight": "regular", "pdf.fonttype": 42})
    fig_baseline(df, figs, controls)
    if df[df.strength == MAIN_LAMBDA].model.nunique() > 1:
        fig_adv_training(df, figs)
    if df[df.train_adv == "none"].strength.nunique() > 1:
        fig_lambda(df, figs)
    fig_examples(run, figs)
    fig_training(run, figs, df)
    table_clean(df, gen); table_robust(df, gen); table_lambda(df, gen)
    table_worst(df, controls, gen); numbers(df, run, gen, controls)
    print("wrote", sorted(str(p) for p in figs.glob("*.pdf")), sorted(str(p) for p in gen.glob("*.tex")))


if __name__ == "__main__":
    main()

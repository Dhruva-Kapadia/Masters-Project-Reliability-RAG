#!/usr/bin/env python3
"""Plot benchmark results (accuracy, attack success rate, robustness, cost).

Reads the per-config result CSVs that main.py writes to <run_dir>/output/ and
writes PNGs to <run_dir>/output/plots/ (or --out_dir). Several run dirs can be
combined (e.g. separate one-config local runs); if the same config appears more
than once, its rows are averaged.

Usage:
    uv run python scripts/plot_results.py runs3                      # -> runs3/output/plots/
    uv run python scripts/plot_results.py local_runs/run* --out_dir local_runs/plots
    uv run python scripts/plot_results.py --csv runs3/output/summary.csv --out_dir figs
    uv run python scripts/plot_results.py local_runs/sweep1 --by_position --out_dir local_runs/sweep1/plots

--by_position keeps each poisoned position separate (attack shown as e.g.
"PIA @ pos 5", 1-based) instead of averaging over positions.

Figures (one panel per dataset where relevant):
    accuracy.png    accuracy per defense, clean vs. under attack
    asr.png         attack success rate per defense (lower is better)
    robustness.png  accuracy vs. ASR under attack (top-left is best)
    cost.png        mean seconds and LLM tokens per question (clean runs)
    results_table.md  the same numbers as a table
scripts/aggregate_results.py calls this automatically after building summary.csv.
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless (Slurm nodes, no display)
import matplotlib.pyplot as plt
import pandas as pd

# Display order: undefended baseline, paper's methods, then other baselines.
DEFENSE_ORDER = ["none", "MIS", "sampleMIS", "graph", "keyword", "sampling_keyword",
                 "sampling", "decoding", "voting", "astuterag", "instructrag_icl"]
DEFENSE_LABEL = {"none": "No defense", "MIS": "MIS", "sampleMIS": "Sample-MIS",
                 "graph": "Graph", "keyword": "Keyword", "sampling_keyword": "Sampling+Keyword",
                 "sampling": "Sampling", "decoding": "Decoding", "voting": "Voting",
                 "astuterag": "AstuteRAG", "instructrag_icl": "InstructRAG-ICL"}
PAPER_METHODS = {"MIS", "sampleMIS"}

# Validated categorical palette (light surface), fixed order: identity follows
# the attack condition in every figure, never its rank.
ATTACK_COLOR = {"none": "#2a78d6", "PIA": "#eb6834", "Poison": "#1baf7a"}
FALLBACK_COLORS = ["#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# --by_position: one hue per attack, lighter -> darker for later positions
POSITION_SHADES = {"PIA": ["#f4a37f", "#eb6834", "#b8461b"],
                   "Poison": ["#6fd3ad", "#1baf7a", "#11774f"]}
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.color": MUTED, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 9,
    "text.color": INK, "font.size": 9, "legend.frameon": False, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def attack_label(a):
    return "Clean" if a == "none" else f"Under attack ({a})"


def attack_color(a, attacks):
    if a in ATTACK_COLOR:
        return ATTACK_COLOR[a]
    if " @ pos " in a:  # --by_position label
        base = a.split(" @ pos ")[0]
        same = sorted((x for x in attacks if x.startswith(base + " @ pos ")),
                      key=lambda x: int(x.rsplit(" ", 1)[1]))
        shades = POSITION_SHADES.get(base)
        if shades and len(same) <= len(shades):
            return shades[same.index(a)]
    extra = [x for x in attacks if x not in ATTACK_COLOR]
    return FALLBACK_COLORS[extra.index(a) % len(FALLBACK_COLORS)]


def load(csv_paths, by_position=False):
    frames = []
    for f in csv_paths:
        if os.path.basename(f) == "summary.csv" and len(csv_paths) > 1:
            continue  # per-config CSVs are already included
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Skipping {f}: {e}", file=sys.stderr)
            continue
        if len(df):
            frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    if by_position and "attackpos" in df.columns:
        attacked = df["attack_method"] != "none"
        df.loc[attacked, "attack_method"] = (df.loc[attacked, "attack_method"] + " @ pos "
                                             + (df.loc[attacked, "attackpos"].astype(int) + 1).astype(str))
    keys = ["model_name", "dataset_name", "defense_method", "attack_method"]
    num = ["acc", "asr", "input_tokens", "output_tokens", "total_time_sec", "dataset_size"]
    df = df.groupby(keys, as_index=False)[num].mean()
    df["tokens_per_q"] = (df["input_tokens"] + df["output_tokens"]) / df["dataset_size"]
    df["sec_per_q"] = df["total_time_sec"] / df["dataset_size"]
    return df


def ordered_defenses(df):
    present = list(dict.fromkeys(df["defense_method"]))
    known = [d for d in DEFENSE_ORDER if d in present]
    return known + sorted(d for d in present if d not in DEFENSE_ORDER)


def style_axis(ax, xmax=None, pct=False):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if xmax is not None:
        ax.set_xlim(0, xmax)
    if pct:
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def ylabels(ax, defenses):
    ax.set_yticks(range(len(defenses)))
    ax.set_yticklabels([DEFENSE_LABEL.get(d, d) for d in defenses])
    for tick, d in zip(ax.get_yticklabels(), defenses):
        if d in PAPER_METHODS:
            tick.set_fontweight("bold")
    # explicit limits (not invert_yaxis, which toggles and cancels on shared axes)
    ax.set_ylim(len(defenses) - 0.5, -0.5)


def bar_label(ax, x, y, text):
    ax.text(x + 0.012, y, text, va="center", ha="left", fontsize=7.5, color=INK_2)


def title_for(df, what):
    models = ", ".join(sorted(df["model_name"].unique()))
    n = sorted(df["dataset_size"].astype(int).unique())
    n_txt = f"n={n[0]}" if len(n) == 1 else f"n={n[0]}–{n[-1]}"
    return f"{what} — {models} ({n_txt} questions per config)"


def grouped_bars(df, metric, attacks, path, suptitle, xlabel, note=None):
    datasets = sorted(df["dataset_name"].unique())
    defenses = ordered_defenses(df)
    # bars per defense grow with the number of attack variants; legend wraps at 3 per row
    ncol = min(len(attacks), 3)
    legend_rows = -(-len(attacks) // ncol)
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(max(5.2 * len(datasets), 10.4),
                                      max(0.42, 0.17 * len(attacks)) * len(defenses) + 1.6 + 0.25 * legend_rows),
                             sharey=True, squeeze=False)
    h = 0.8 / len(attacks)
    for ax, ds in zip(axes[0], datasets):
        sub = df[df["dataset_name"] == ds]
        for k, a in enumerate(attacks):
            for i, d in enumerate(defenses):
                row = sub[(sub["defense_method"] == d) & (sub["attack_method"] == a)]
                y = i - 0.4 + h * (k + 0.5)
                if row.empty:
                    ax.text(0.01, y, "not run", va="center", fontsize=7, color=MUTED)
                    continue
                v = float(row[metric].iloc[0])
                ax.barh(y, v, height=h - 0.04, color=attack_color(a, attacks), edgecolor=SURFACE, linewidth=1)
                bar_label(ax, v, y, f"{v:.0%}")
        ylabels(ax, defenses)
        style_axis(ax, xmax=1.12, pct=True)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(ds, loc="left")
        ax.set_xlabel(xlabel)
    handles = [plt.Rectangle((0, 0), 1, 1, color=attack_color(a, attacks)) for a in attacks]
    fig.legend(handles, [attack_label(a) for a in attacks], loc="upper left", ncol=ncol,
               bbox_to_anchor=(0.005, 0.955 - 0.25 / fig.get_figheight()))
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12, fontweight="bold")
    if note:
        fig.text(0.01, 0.005, note, fontsize=7.5, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1 - (0.6 + 0.25 * legend_rows) / fig.get_figheight()))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def robustness(df, attacked, path):
    datasets = sorted(df["dataset_name"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.6), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = df[(df["dataset_name"] == ds) & (df["attack_method"].isin(attacked))]
        for a in attacked:
            s = sub[sub["attack_method"] == a]
            ax.scatter(s["asr"], s["acc"], s=48, color=attack_color(a, ["none"] + attacked),
                       edgecolor=SURFACE, linewidth=1.5, zorder=3, label=attack_label(a))
            for _, r in s.iterrows():
                d = r["defense_method"]
                ax.annotate(DEFENSE_LABEL.get(d, d), (r["asr"], r["acc"]), xytext=(5, 3),
                            textcoords="offset points", fontsize=7.5, color=INK_2,
                            fontweight="bold" if d in PAPER_METHODS else "normal")
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        ax.set_xlabel("Attack success rate (lower is better)")
        ax.set_ylabel("Accuracy under attack (higher is better)")
        ax.set_title(ds, loc="left")
        ax.text(0.0, 1.0, "best", fontsize=7.5, color=MUTED, ha="left", va="top")
    if len(attacked) > 1:
        axes[0][0].legend(loc="lower right")
    fig.suptitle(title_for(df, "Robustness under attack"), x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cost(df, path):
    sub_all = df[df["attack_method"] == "none"]
    which = "clean runs"
    if sub_all.empty:
        # no clean runs (e.g. an attack sweep): average over the attacked configs
        num = ["sec_per_q", "tokens_per_q", "dataset_size"]
        sub_all = df.groupby(["model_name", "dataset_name", "defense_method"], as_index=False)[num].mean()
        which = "mean over attacked configs"
    datasets = sorted(sub_all["dataset_name"].unique())
    defenses = ordered_defenses(sub_all)
    # Two measures with different units -> separate panels (no dual axis).
    fig, axes = plt.subplots(len(datasets), 2, figsize=(10.4, (0.32 * len(defenses) + 1.3) * len(datasets)),
                             squeeze=False, sharey="row")
    for r, ds in enumerate(datasets):
        sub = sub_all[sub_all["dataset_name"] == ds].set_index("defense_method")
        for c, (metric, label, short, fmt) in enumerate([
                ("sec_per_q", "Seconds per question", "time", "{:.1f}s"),
                ("tokens_per_q", "LLM tokens per question", "LLM tokens", "{:,.0f}")]):
            ax = axes[r][c]
            vals = [float(sub.loc[d, metric]) if d in sub.index else None for d in defenses]
            vmax = max([v for v in vals if v is not None] or [1])
            for i, v in enumerate(vals):
                if v is None:
                    ax.text(0, i, " not run", va="center", fontsize=7, color=MUTED)
                    continue
                ax.barh(i, v, height=0.62, color=ATTACK_COLOR["none"])
                ax.text(v + vmax * 0.01, i, fmt.format(v), va="center", fontsize=7.5, color=INK_2)
            ylabels(ax, defenses)
            style_axis(ax, xmax=vmax * 1.18)
            ax.set_title(f"{ds} — {short}", loc="left")
            ax.set_xlabel(label)
    fig.suptitle(title_for(df, f"Cost per question ({which})"), x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def table(df, path):
    defenses = ordered_defenses(df)
    lines = ["| model | dataset | defense | attack | accuracy | ASR | s/question | tokens/question | n |",
             "|---|---|---|---|---|---|---|---|---|"]
    df = df.assign(_o=df["defense_method"].map({d: i for i, d in enumerate(defenses)}),
                   _a=df["attack_method"].map(lambda a: attack_sort_key(a)))
    for _, r in df.sort_values(["model_name", "dataset_name", "_o", "_a"]).iterrows():
        asr = "–" if r["attack_method"] == "none" else f"{r['asr']:.1%}"
        lines.append(f"| {r['model_name']} | {r['dataset_name']} | {DEFENSE_LABEL.get(r['defense_method'], r['defense_method'])} "
                     f"| {r['attack_method']} | {r['acc']:.1%} | {asr} | {r['sec_per_q']:.1f} "
                     f"| {r['tokens_per_q']:,.0f} | {int(r['dataset_size'])} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def attack_sort_key(a):
    # clean first, then by attack, then by numeric position ("pos 10" after "pos 5")
    if " @ pos " in a:
        base, pos = a.split(" @ pos ")
        return (True, base, int(pos))
    return (a != "none", a, 0)


def make_plots(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for model in sorted(df["model_name"].unique()):
        mdf = df[df["model_name"] == model]
        # one model per figure set; prefix only when several models are mixed
        pre = "" if df["model_name"].nunique() == 1 else f"{model}_"
        attacks = sorted(mdf["attack_method"].unique(), key=attack_sort_key)
        attacked = [a for a in attacks if a != "none"]

        p = os.path.join(out_dir, f"{pre}accuracy.png")
        grouped_bars(mdf, "acc", attacks, p, title_for(mdf, "Accuracy by defense"), "Accuracy",
                     note="Bold = the paper's methods (MIS, Sample-MIS).")
        written.append(p)
        if attacked:
            p = os.path.join(out_dir, f"{pre}asr.png")
            grouped_bars(mdf[mdf["attack_method"] != "none"], "asr", attacked, p,
                         title_for(mdf, "Attack success rate by defense (lower is better)"),
                         "Attack success rate")
            written.append(p)
            p = os.path.join(out_dir, f"{pre}robustness.png")
            robustness(mdf, attacked, p)
            written.append(p)
        p = os.path.join(out_dir, f"{pre}cost.png")
        cost(mdf, p)
        written.append(p)
    p = os.path.join(out_dir, "results_table.md")
    table(df, p)
    written.append(p)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*", help="run directories (runsN, local_runs/runN); their output/*.csv are read")
    ap.add_argument("--csv", nargs="*", default=[], help="result CSV files to read instead/in addition")
    ap.add_argument("--by_position", action="store_true",
                    help="keep each poisoned position separate (attack shown as 'PIA @ pos 5') instead of averaging")
    ap.add_argument("--out_dir", default=None, help="where to write plots (default: <run_dir>/output/plots for a single run dir)")
    args = ap.parse_args()

    csvs = list(args.csv)
    for rd in args.run_dirs:
        csvs += [f for f in sorted(glob.glob(os.path.join(rd, "output", "*.csv")))
                 if os.path.basename(f) != "summary.csv"]
    if not csvs:
        ap.error("no result CSVs found (give run dirs or --csv)")
    out_dir = args.out_dir
    if out_dir is None:
        if len(args.run_dirs) != 1:
            ap.error("--out_dir is required when combining several run dirs")
        out_dir = os.path.join(args.run_dirs[0], "output", "plots")

    df = load(csvs, by_position=args.by_position)
    if df is None:
        print("No completed results yet (CSVs have no data rows).")
        return
    for p in make_plots(df, out_dir):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()

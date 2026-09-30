#!/usr/bin/env python3
"""Benchmark figure from a run's summary.csv: accuracy (clean vs attacked) and
attack success rate, one column per dataset.

Called automatically by scripts/aggregate_results.py; also runs standalone:
    uv run python scripts/plot_benchmark.py runs3/output/summary.csv
    uv run python scripts/plot_benchmark.py runs3/output/summary.csv --out figs/bench.png

Rows are keyed by (dataset, defense, attack). If a key appears more than once
(e.g. a --top_k 0 smoke test next to the real top_k 10 run), only the most
common top_k is kept and, within it, the row with the largest dataset_size --
never an average. Defenses listed in scripts/benchmark_configs.txt that have
no result (crashed / not run yet) are shown as "not run" instead of silently
disappearing. One figure per model; with several models the file name gets a
_<model> suffix.
"""
import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')  # headless (Wulver compute/login nodes)
import matplotlib.pyplot as plt
import pandas as pd

# Fixed categorical order (validated: CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on white)
ATTACK_COLORS = {'none': '#2a78d6', 'PIA': '#eb6834', 'Poison': '#1baf7a'}
ATTACK_LABELS = {'none': 'no attack', 'PIA': 'PIA attack', 'Poison': 'Poison attack'}
INK, MUTED, GRID, SURFACE = '#1c1c1a', '#6b6b66', '#e6e5e0', '#ffffff'
# Preferred left-to-right order; unknown defenses are appended alphabetically.
DEFENSE_ORDER = ['none', 'keyword', 'sampling_keyword', 'decoding', 'sampling',
                 'graph', 'MIS', 'sampleMIS', 'astuterag', 'instructrag_icl', 'voting']


def _top_k(src):
    m = re.search(r'-top(\d+)-', str(src))
    return int(m.group(1)) if m else -1


def dedupe(df):
    df = df.copy()
    df['top_k'] = df['source_file'].map(_top_k) if 'source_file' in df else -1
    dropped = []
    keep = []
    for key, g in df.groupby(['model_name', 'dataset_name', 'defense_method', 'attack_method']):
        if len(g) > 1:
            common_k = df[df['dataset_name'] == key[1]]['top_k'].mode().iloc[0]
            g2 = g[g['top_k'] == common_k] if (g['top_k'] == common_k).any() else g
            best = g2.sort_values('dataset_size', ascending=False).iloc[[0]]
            dropped += [r for r in g.index if r not in best.index]
            keep.append(best)
        else:
            keep.append(g)
    if dropped:
        for _, r in df.loc[dropped].iterrows():
            print(f"  plot: ignoring duplicate {r['dataset_name']}/{r['defense_method']}/{r['attack_method']} "
                  f"(top_k={r['top_k']}, n={r['dataset_size']}, {r.get('source_file', '')})")
    return pd.concat(keep)


def expected_defenses(config_path):
    if not config_path or not os.path.exists(config_path):
        return []
    out = []
    for line in open(config_path):
        parts = line.split()
        if len(parts) >= 2 and parts[1] not in out:
            out.append(parts[1])
    return out


def order_defenses(present, expected):
    allx = list(dict.fromkeys(list(expected) + sorted(present)))
    ranked = [d for d in DEFENSE_ORDER if d in allx]
    return ranked + sorted(d for d in allx if d not in ranked)


def style_axis(ax, ylabel=None):
    ax.set_ylim(0, 1)
    ax.set_facecolor(SURFACE)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=10)


def plot_model(df, model, out_path, expected):
    datasets = sorted(df['dataset_name'].unique())
    defenses = order_defenses(df['defense_method'].unique(), expected)
    attacks = [a for a in ATTACK_COLORS if a in set(df['attack_method'])] + \
              sorted(set(df['attack_method']) - set(ATTACK_COLORS))
    attacked = [a for a in attacks if a != 'none']

    fig, axes = plt.subplots(2, len(datasets), figsize=(max(6.5, 1.0 * len(defenses) * len(datasets) + 1.5), 8.2),
                             squeeze=False, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    x = range(len(defenses))
    width = 0.8 / max(len(attacks), 1)

    for c, ds in enumerate(datasets):
        d = df[df['dataset_name'] == ds]
        n = int(d['dataset_size'].max()) if len(d) else 0
        for row, (metric, series, title) in enumerate([('acc', attacks, 'Accuracy (higher is better)'),
                                                       ('asr', attacked, 'Attack success rate (lower is better)')]):
            ax = axes[row][c]
            style_axis(ax, ('Accuracy' if row == 0 else 'ASR') if c == 0 else None)
            ax.set_title(f'{ds} (n={n}) — {title}', color=INK, fontsize=11, loc='left')
            nser = max(len(series), 1)
            w = width  # same bar width in both rows, so the ASR bars line up with the attacked accuracy bars
            for s_i, att in enumerate(series):
                # ASR row: place each attacked series at the same slot it has in the accuracy row
                slot = attacks.index(att) if row == 1 else s_i
                offs = (slot - (len(attacks) - 1) / 2) * w
                for xi, dfn in enumerate(defenses):
                    v = d[(d['defense_method'] == dfn) & (d['attack_method'] == att)][metric]
                    if len(v):
                        ax.bar(xi + offs, float(v.iloc[0]), width=w, color=ATTACK_COLORS.get(att, MUTED),
                               edgecolor=SURFACE, linewidth=2, label=ATTACK_LABELS.get(att, att))
            for xi, dfn in enumerate(defenses):
                if d[d['defense_method'] == dfn].empty:
                    ax.text(xi, 0.03, 'not run', rotation=90, ha='center', va='bottom', color=MUTED, fontsize=8)
            ax.set_xticks(list(x))
            ax.set_xticklabels(defenses, rotation=40, ha='right', color=INK)
            ax.set_xlim(-0.6, len(defenses) - 0.4)

    # one legend for the figure (entity -> color is constant across panels)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ATTACK_COLORS.get(a, MUTED)) for a in attacks]
    fig.legend(handles, [ATTACK_LABELS.get(a, a) for a in attacks], loc='upper right', frameon=False,
               ncol=len(attacks), fontsize=10, labelcolor=INK)
    fig.suptitle(f'ReliabilityRAG benchmark — {model}', x=0.01, ha='left', color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f'Plot -> {out_path}')


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plot_summary(summary_csv, out_path=None, config_path=os.path.join(REPO, 'scripts', 'benchmark_configs.txt')):
    df = pd.read_csv(summary_csv)
    if df.empty:
        print('plot: summary is empty, nothing to plot')
        return []
    df = dedupe(df)
    out_path = out_path or os.path.join(os.path.dirname(os.path.abspath(summary_csv)), 'benchmark.png')
    expected = expected_defenses(config_path)
    models = sorted(df['model_name'].unique())
    written = []
    for model in models:
        path = out_path if len(models) == 1 else out_path.replace('.png', f'_{model}.png')
        plot_model(df[df['model_name'] == model], model, path, expected)
        written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('summary_csv')
    ap.add_argument('--out', help='output PNG (default: benchmark.png next to summary_csv)')
    ap.add_argument('--configs', default=os.path.join(REPO, 'scripts', 'benchmark_configs.txt'),
                    help='config list used to mark defenses that did not produce results')
    args = ap.parse_args()
    plot_summary(args.summary_csv, args.out, args.configs)


if __name__ == '__main__':
    main()

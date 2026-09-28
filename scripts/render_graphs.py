#!/usr/bin/env python3
"""Render contradiction graphs logged by `main.py --log_graphs`.

Standard library only (runs on Wulver or a laptop). Outputs, any combination:

  --html FILE   one self-contained page: every question's graph + NLI
                contradiction-probability matrix + node answers, with filters
                (default: <input>.html next to the first input)
  --dot DIR     one Graphviz .dot file per question
  --render svg|png   also run `dot` on those files (needs Graphviz installed)
  --csv FILE    one summary row per question (edges, poison kept?, correct?)

Examples:
  python scripts/render_graphs.py runs3/graphs/*.jsonl
  python scripts/render_graphs.py runs3/graphs/x.jsonl --dot runs3/graphs/dot --render svg
  python scripts/render_graphs.py runs3/graphs/x.jsonl --csv runs3/graphs/x.csv --no_html

Legend (HTML and DOT):
  filled node   kept by the defense (selected)       hollow node  dropped
  red outline   node contains an injected document    dotted node  answered "I don't know"
  solid edge    NLI contradiction >= threshold        dashed edge  edge set/removed by --err noise
  arrow (graph method only) points from the higher-ranked to the lower-ranked document
"""
import argparse
import csv
import glob
import html
import json
import math
import os
import shutil
import subprocess
import sys


# ───────────────────────── loading ─────────────────────────
def load(paths):
    recs = []
    for pattern in paths:
        files = sorted(glob.glob(pattern)) or [pattern]
        for f in files:
            with open(f, encoding='utf-8') as fh:
                for ln, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        print(f'skip {f}:{ln} (bad json, run still writing?)', file=sys.stderr)
                        continue
                    r['_source'] = os.path.basename(f)
                    recs.append(r)
    return recs


def label(r, n):
    """d1..dK for document graphs (1-based rank), s1..sT for sampleMIS samples."""
    if r['method'] == 'sampleMIS':
        return f"s{n['id'] + 1}"
    return f"d{n['id'] + 1}"


def doc_str(n):
    return ','.join(str(d + 1) for d in n.get('doc_ranks', []))


def summarize(r):
    nodes = r['nodes']
    sel = set(r['selected'])
    pois = [n['id'] for n in nodes if n.get('poisoned')]
    edges = [p for p in r['pairs'] if p['edge']]
    pois_ranks = set(r.get('poison_positions') or [])
    return {
        'source': r['_source'], 'rep': r.get('rep'), 'item_idx': r.get('item_idx'), 'method': r['method'],
        'n_nodes': len(nodes), 'n_edges': len(edges),
        'n_idk': sum(1 for n in nodes if n.get('idk')),
        'n_flipped': sum(1 for p in r['pairs'] if p.get('flipped')),
        'selected': ' '.join(str(s + 1) for s in sorted(sel)),
        'poison_nodes': ' '.join(str(p + 1) for p in pois),
        'poison_node_selected': any(p in sel for p in pois) if pois else None,
        'poison_docs_in_final': sum(1 for d in r.get('final_doc_ranks', []) if d in pois_ranks),
        'final_context_docs': len(r.get('final_doc_ranks', [])),
        'correct': r.get('correct'), 'attack_success': r.get('attack_success'),
        'question': r.get('question'),
    }


# ───────────────────────── DOT ─────────────────────────
def to_dot(r):
    sel = set(r['selected'])
    directed = r.get('directed')
    out = ['digraph G {' if directed else 'graph G {',
           '  layout=circo; bgcolor="white"; fontname="Helvetica";',
           f'  label="{esc_dot(r.get("question", ""))}\\n{r["method"]} rep {r.get("rep")} item {r.get("item_idx")}"; labelloc=t;',
           '  node [shape=circle, fontname="Helvetica", fontsize=11, width=0.55, fixedsize=true];']
    for n in r['nodes']:
        attrs = [f'label="{label(r, n)}"',
                 f'tooltip="{esc_dot(n["answer"][:200])}"',
                 'style="filled' + (',dotted' if n.get('idk') else '') + '"',
                 f'fillcolor="{"#9fd4b0" if n["id"] in sel else "#ffffff"}"',
                 f'color="{"#c0392b" if n.get("poisoned") else "#555555"}"',
                 f'penwidth={3 if n.get("poisoned") else 1}']
        if r['method'] == 'sampleMIS':
            attrs[0] = f'label="{label(r, n)}\\n{{{doc_str(n)}}}"'
        out.append(f'  n{n["id"]} [{", ".join(attrs)}];')
    arrow = '->' if directed else '--'
    for p in r['pairs']:
        if p['edge']:
            style = 'dashed' if p.get('flipped') else 'solid'
            out.append(f'  n{p["i"]} {arrow} n{p["j"]} [style={style}, tooltip="p={p["p_contra"]:.2f}"];')
    out.append('}')
    return '\n'.join(out)


def esc_dot(s):
    return str(s).replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')


# ───────────────────────── HTML ─────────────────────────
def svg_graph(r, size=300):
    nodes = r['nodes']
    k = max(len(nodes), 1)
    cx = cy = size / 2
    rad = size / 2 - 34
    nr = 17 if r['method'] != 'sampleMIS' else 19
    pos = {n['id']: (cx + rad * math.cos(-math.pi / 2 + 2 * math.pi * i / k),
                     cy + rad * math.sin(-math.pi / 2 + 2 * math.pi * i / k)) for i, n in enumerate(nodes)}
    sel = set(r['selected'])
    parts = [f'<svg viewBox="0 0 {size} {size}" class="g" role="img" aria-label="contradiction graph">']
    if r.get('directed'):
        parts.append('<defs><marker id="arr" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" '
                     'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="arrow"/></marker></defs>')
    for p in r['pairs']:
        if not p['edge'] or p['i'] not in pos or p['j'] not in pos:
            continue
        (x1, y1), (x2, y2) = pos[p['i']], pos[p['j']]
        dx, dy = x2 - x1, y2 - y1
        d = math.hypot(dx, dy) or 1
        x2e, y2e = x2 - dx / d * (nr + 2), y2 - dy / d * (nr + 2)
        cls = 'e flip' if p.get('flipped') else 'e'
        mk = ' marker-end="url(#arr)"' if r.get('directed') else ''
        parts.append(f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2e:.1f}" y2="{y2e:.1f}"{mk}>'
                     f'<title>{label(r, nodes_by_id(r)[p["i"]])} vs {label(r, nodes_by_id(r)[p["j"]])}: p_contra={p["p_contra"]:.2f}'
                     f'{" (flipped by --err)" if p.get("flipped") else ""}</title></line>')
    for n in nodes:
        x, y = pos[n['id']]
        cls = ['n', 'sel' if n['id'] in sel else 'drop']
        if n.get('poisoned'):
            cls.append('pois')
        if n.get('idk'):
            cls.append('idk')
        tip = html.escape(f"{label(r, n)}  docs[{doc_str(n)}]  {'KEPT' if n['id'] in sel else 'dropped'}"
                          f"{'  POISONED' if n.get('poisoned') else ''}\n{n['answer'][:300]}")
        parts.append(f'<g><title>{tip}</title><circle class="{" ".join(cls)}" cx="{x:.1f}" cy="{y:.1f}" r="{nr}"/>'
                     f'<text x="{x:.1f}" y="{y + 4:.1f}" class="nl">{label(r, n)}</text></g>')
    parts.append('</svg>')
    return ''.join(parts)


_nb_cache = {}


def nodes_by_id(r):
    key = id(r)
    if key not in _nb_cache:
        _nb_cache[key] = {n['id']: n for n in r['nodes']}
    return _nb_cache[key]


def svg_matrix(r, size=300):
    nodes = r['nodes']
    k = len(nodes)
    if k == 0:
        return ''
    pad = 24
    cell = (size - pad) / k
    mat = {}
    for p in r['pairs']:
        mat[(p['i'], p['j'])] = p
        mat[(p['j'], p['i'])] = p
    thr = r.get('threshold', 0.5)
    parts = [f'<svg viewBox="0 0 {size} {size}" class="m" role="img" aria-label="contradiction probability matrix">']
    ids = [n['id'] for n in nodes]
    for a, ia in enumerate(ids):
        parts.append(f'<text class="ax" x="{pad + a * cell + cell / 2:.1f}" y="{pad - 8}">{a + 1}</text>')
        parts.append(f'<text class="ax" x="{pad - 10}" y="{pad + a * cell + cell / 2 + 4:.1f}">{a + 1}</text>')
        for b, ib in enumerate(ids):
            x, y = pad + b * cell, pad + a * cell
            if ia == ib:
                parts.append(f'<rect class="diag" x="{x:.1f}" y="{y:.1f}" width="{cell - 1:.1f}" height="{cell - 1:.1f}"/>')
                continue
            p = mat.get((ia, ib))
            if p is None:
                continue
            v = p['p_contra']
            op = 0.08 + 0.92 * v
            cls = 'c edge' if p['edge'] else 'c'
            parts.append(f'<rect class="{cls}" style="fill-opacity:{op:.2f}" x="{x:.1f}" y="{y:.1f}" '
                         f'width="{cell - 1:.1f}" height="{cell - 1:.1f}"><title>{a + 1} vs {b + 1}: p_contra={v:.2f}'
                         f'{" ≥ " if v >= thr else " < "}{thr} {"edge" if p["edge"] else "no edge"}'
                         f'{" (flipped by --err)" if p.get("flipped") else ""}</title></rect>')
    parts.append('</svg>')
    return ''.join(parts)


def card(r, idx):
    s = summarize(r)
    sel = set(r['selected'])
    badges = []
    if r.get('correct') is not None:
        badges.append(f'<span class="b {"ok" if r["correct"] else "bad"}">{"correct" if r["correct"] else "wrong"}</span>')
    if r.get('attack_success') is not None:
        badges.append(f'<span class="b {"bad" if r["attack_success"] else "ok"}">'
                      f'{"attack succeeded" if r["attack_success"] else "attack blocked"}</span>')
    if s['poison_node_selected'] is not None:
        badges.append(f'<span class="b {"bad" if s["poison_node_selected"] else "ok"}">'
                      f'{"poison kept" if s["poison_node_selected"] else "poison dropped"}</span>')
    rows = []
    for n in r['nodes']:
        rows.append(f'<tr class="{"kept" if n["id"] in sel else ""}"><td>{label(r, n)}</td><td>{doc_str(n)}</td>'
                    f'<td>{"✓" if n["id"] in sel else ""}</td><td>{"⚠" if n.get("poisoned") else ""}</td>'
                    f'<td class="ans">{html.escape(n["answer"][:400])}</td></tr>')
    extra = r.get('extra') or {}
    extra_html = ''
    if extra:
        extra_html = '<p class="meta">' + ' · '.join(f'{html.escape(str(k))}: {html.escape(json.dumps(v))}'
                                                     for k, v in extra.items()) + '</p>'
    gold = r.get('gold_answers')
    gold = ' | '.join(gold) if isinstance(gold, list) else (gold or '')
    data_attrs = (f'data-correct="{int(bool(r.get("correct")))}" data-att="{int(bool(r.get("attack_success")))}" '
                  f'data-pkept="{int(bool(s["poison_node_selected"]))}" data-edges="{s["n_edges"]}" '
                  f'data-text="{html.escape((r.get("question") or "").lower())}"')
    return f'''
<section class="card" id="r{idx}" {data_attrs}>
  <header>
    <div class="q"><span class="idx">{r["method"]} · rep {r.get("rep")} · item {r.get("item_idx")}</span>
      {html.escape(r.get("question") or "")}</div>
    <div class="badges">{"".join(badges)}</div>
  </header>
  <p class="meta"><b>gold</b> {html.escape(gold[:200])} · <b>attacker target</b> {html.escape(str(r.get("incorrect_answer") or "—"))}
    · <b>poisoned ranks</b> {", ".join(str(p + 1) for p in r.get("poison_positions") or []) or "none"}
    · <b>edges</b> {s["n_edges"]} · <b>kept</b> {s["selected"] or "—"} · <b>final context docs</b>
    {", ".join(str(d + 1) for d in r.get("final_doc_ranks", [])) or "—"}</p>
  <p class="meta"><b>final answer</b> {html.escape((r.get("final_answer") or "")[:300])}</p>
  <div class="viz">{svg_graph(r)}{svg_matrix(r)}</div>
  {extra_html}
  <details><summary>node answers</summary>
    <table><thead><tr><th>node</th><th>doc ranks</th><th>kept</th><th>poison</th><th>answer</th></tr></thead>
    <tbody>{"".join(rows)}</tbody></table></details>
</section>'''


def aggregate(recs):
    ss = [summarize(r) for r in recs]
    n = len(ss) or 1
    att = [s for s in ss if s['poison_node_selected'] is not None]
    agg = {
        'questions': len(ss),
        'avg edges': round(sum(s['n_edges'] for s in ss) / n, 2),
        'avg IDK nodes': round(sum(s['n_idk'] for s in ss) / n, 2),
        'accuracy': f"{sum(1 for s in ss if s['correct']) / n:.0%}",
    }
    if att:
        agg['poison node dropped'] = f"{sum(1 for s in att if not s['poison_node_selected']) / len(att):.0%}"
        agg['attack success'] = f"{sum(1 for s in att if s['attack_success']) / len(att):.0%}"
    return agg


CSS = '''
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#6b6b66;--card:#fff;--line:#e4e2dd;--kept:#3f9e6a;--keptf:#cdebd8;
--drop:#fff;--pois:#c0392b;--edge:#57534e;--cell:#c0392b;--ok:#2f7d4f;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#ecebe8;--mut:#a3a29d;--card:#1f1f1d;
--line:#34332f;--kept:#63c190;--keptf:#1f4d33;--drop:#1f1f1d;--pois:#ef6b5e;--edge:#b5b3ad;--cell:#ef6b5e;--ok:#63c190;--bad:#ef6b5e}}
:root[data-theme="dark"]{--bg:#161615;--fg:#ecebe8;--mut:#a3a29d;--card:#1f1f1d;--line:#34332f;--kept:#63c190;--keptf:#1f4d33;
--drop:#1f1f1d;--pois:#ef6b5e;--edge:#b5b3ad;--cell:#ef6b5e;--ok:#63c190;--bad:#ef6b5e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px 60px}h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--mut);margin:0 0 14px}.agg{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px}
.agg div{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 10px}
.agg b{display:block;font-size:16px}.agg span{color:var(--mut);font-size:12px}
.ctl{position:sticky;top:0;background:var(--bg);padding:8px 0;display:flex;flex-wrap:wrap;gap:12px;align-items:center;z-index:2;border-bottom:1px solid var(--line)}
.ctl input[type=search]{padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);min-width:200px}
.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--mut);font-size:12px;margin:10px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:14px 0}
.card header{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}.q{font-weight:600}
.idx{display:block;font-weight:400;color:var(--mut);font-size:12px}.badges{display:flex;gap:6px;flex-wrap:wrap;align-items:flex-start}
.b{font-size:12px;padding:2px 8px;border-radius:99px;border:1px solid currentColor}.b.ok{color:var(--ok)}.b.bad{color:var(--bad)}
.meta{color:var(--mut);margin:6px 0;font-size:13px;overflow-wrap:anywhere}.meta b{color:var(--fg);font-weight:600}
.viz{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:8px}
svg{width:100%;max-width:340px;height:auto;justify-self:center}
.n{stroke:var(--edge);stroke-width:1.2}.n.sel{fill:var(--keptf);stroke:var(--kept)}.n.drop{fill:var(--drop)}
.n.pois{stroke:var(--pois);stroke-width:3.5}.n.idk{stroke-dasharray:3 3}
.nl{font-size:11px;text-anchor:middle;fill:var(--fg);pointer-events:none}
.e{stroke:var(--edge);stroke-width:1.3;opacity:.75}.e.flip{stroke-dasharray:5 4}.arrow{fill:var(--edge)}
.c{fill:var(--cell)}.c.edge{stroke:var(--fg);stroke-width:1.2}.diag{fill:var(--line)}
.ax{font-size:9px;fill:var(--mut);text-anchor:middle}
details{margin-top:8px}summary{cursor:pointer;color:var(--mut)}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:6px}
th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}
tr.kept td:first-child{color:var(--kept);font-weight:600}.ans{overflow-wrap:anywhere}
.hidden{display:none}
'''

JS = '''
const q=document.getElementById('q'),w=document.getElementById('wrong'),p=document.getElementById('pk'),
a=document.getElementById('as'),e=document.getElementById('ed'),cnt=document.getElementById('cnt');
function f(){let n=0;document.querySelectorAll('.card').forEach(c=>{const d=c.dataset;
const show=(!q.value||d.text.includes(q.value.toLowerCase()))&&(!w.checked||d.correct==='0')&&(!p.checked||d.pkept==='1')
&&(!a.checked||d.att==='1')&&(!e.checked||d.edges!=='0');c.classList.toggle('hidden',!show);if(show)n++});cnt.textContent=n+' shown'}
[q,w,p,a,e].forEach(x=>x.addEventListener('input',f));f();
'''


def to_html(recs, title):
    agg = aggregate(recs)
    agg_html = ''.join(f'<div><b>{html.escape(str(v))}</b><span>{html.escape(k)}</span></div>' for k, v in agg.items())
    cards = ''.join(card(r, i) for i, r in enumerate(recs))
    sources = sorted({r['_source'] for r in recs})
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Contradiction Graphs</title>
<style>{CSS}</style></head><body><main>
<h1>Contradiction graphs</h1><p class="sub">{html.escape(title)} — {html.escape(", ".join(sources))}</p>
<div class="agg">{agg_html}</div>
<div class="ctl"><input type="search" id="q" placeholder="filter by question text">
<label><input type="checkbox" id="wrong"> wrong answers</label>
<label><input type="checkbox" id="pk"> poison kept</label>
<label><input type="checkbox" id="as"> attack succeeded</label>
<label><input type="checkbox" id="ed"> has edges</label><span id="cnt" class="sub"></span></div>
<div class="legend"><span>filled = kept by defense</span><span>hollow = dropped</span><span>red ring = contains injected doc</span>
<span>dotted ring = "I don't know"</span><span>dashed edge = flipped by --err</span>
<span>matrix: darker = higher NLI contradiction probability, outlined = edge</span>
<span>labels: d<i>n</i> = document at rank <i>n</i>; s<i>n</i> = sample <i>n</i> (sampleMIS)</span></div>
{cards}
</main><script>{JS}</script></body></html>'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='graph JSONL file(s) or globs')
    ap.add_argument('--html', help='output HTML path (default: <first input>.html)')
    ap.add_argument('--no_html', action='store_true')
    ap.add_argument('--dot', help='directory for per-question .dot files')
    ap.add_argument('--render', choices=['svg', 'png', 'pdf'], help='render .dot files with Graphviz `dot`')
    ap.add_argument('--csv', help='per-question summary CSV')
    ap.add_argument('--limit', type=int, help='only the first N questions')
    ap.add_argument('--only_attacked', action='store_true', help='keep questions that had an injected document')
    args = ap.parse_args()

    recs = load(args.inputs)
    if args.only_attacked:
        recs = [r for r in recs if r.get('poison_positions')]
    if args.limit:
        recs = recs[:args.limit]
    if not recs:
        sys.exit('no graph records found')

    for k, v in aggregate(recs).items():
        print(f'{k:>22}: {v}')

    if not args.no_html:
        out = args.html or os.path.splitext(glob.glob(args.inputs[0])[0] if glob.glob(args.inputs[0]) else args.inputs[0])[0] + '.html'
        with open(out, 'w', encoding='utf-8') as f:
            f.write(to_html(recs, f'{len(recs)} questions'))
        print(f'HTML  -> {out}')

    if args.csv:
        rows = [summarize(r) for r in recs]
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'CSV   -> {args.csv}')

    if args.dot:
        os.makedirs(args.dot, exist_ok=True)
        dot_bin = shutil.which('dot') if args.render else None
        if args.render and not dot_bin:
            print('Graphviz `dot` not found; writing .dot files only (module load graphviz / apt install graphviz)',
                  file=sys.stderr)
        for r in recs:
            stem = os.path.join(args.dot, f"{r['method']}_rep{r.get('rep')}_item{r.get('item_idx')}")
            with open(stem + '.dot', 'w', encoding='utf-8') as f:
                f.write(to_dot(r))
            if dot_bin:
                subprocess.run([dot_bin, f'-T{args.render}', stem + '.dot', '-o', f'{stem}.{args.render}'], check=False)
        print(f'DOT   -> {args.dot}/ ({len(recs)} files{", rendered " + args.render if dot_bin else ""})')


if __name__ == '__main__':
    main()

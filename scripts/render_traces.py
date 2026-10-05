"""Render thinking traces written by `main.py --save_traces` as readable
Query -> Thought -> Output steps (Markdown), or flatten them to CSV.

Usage (from the repo root):
    python scripts/render_traces.py runs3/traces/X.jsonl                      # all questions -> X.md
    python scripts/render_traces.py runs3/traces/X.jsonl --items 0,5,12       # chosen questions
    python scripts/render_traces.py runs3/traces/X.jsonl --only wrong         # failed / attacked ones
    python scripts/render_traces.py runs3/traces/X.jsonl --full               # no clipping
    python scripts/render_traces.py runs3/traces/X.jsonl --csv X_calls.csv    # one row per LLM call
"""
import argparse, csv, json, os


def clip(s, n):
    if s is None:
        return None
    s = str(s)
    if n is None or len(s) <= n:
        return s
    return s[:n] + f"\n[... {len(s) - n:,} more characters ...]"


def quote(s):
    return "\n".join("> " + line for line in str(s).splitlines()) or "> "


def fence(s):
    return f"```text\n{s}\n```"


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def verdict(r):
    parts = []
    if r.get("correct") is not None:
        parts.append("correct" if r["correct"] else "wrong")
    if r.get("attack_success") is not None:
        parts.append("ATTACK SUCCEEDED" if r["attack_success"] else "attack resisted")
    return ", ".join(parts) or "ungraded"


def render_item(r, args):
    out = []
    out.append(f"## Item {r['item_idx']} (rep {r['rep']}) · defense `{r.get('defense')}` · {verdict(r)}\n")
    out.append(f"**Query:** {r['question']}\n")
    out.append(f"**Gold:** {' | '.join(map(str, r.get('gold_answers') or []))}  ")
    if r.get("incorrect_answer"):
        out.append(f"**Attacker target:** {r['incorrect_answer']}  ")
    if r.get("poison_positions"):
        out.append(f"**Poisoned ranks:** {r['poison_positions']}")
    out.append("")
    if not args.no_contexts:
        out.append("### Retrieved contexts (in rank order)\n")
        for c in r.get("contexts", []):
            flag = " **[POISONED]**" if c.get("poisoned") else ""
            out.append(f"- **[{c['rank']}]**{flag} {clip(c['text'], args.ctx_chars).replace(chr(10), ' ')}")
        out.append("")
    calls = r.get("calls", [])
    out.append(f"### Reasoning steps ({len(calls)} LLM calls)\n")
    for c in calls:
        label = f"per-document #{c['batch_idx']}" if c.get("stage") == "per_doc" and c.get("batch_idx") is not None else c.get("stage")
        meta = []
        if c.get("cached"): meta.append("from cache")
        if c.get("completion_tokens") is not None: meta.append(f"{c['completion_tokens']} output tokens")
        if c.get("finish_reason") and c["finish_reason"] != "stop": meta.append(f"finish={c['finish_reason']}")
        if c.get("latency_sec") is not None: meta.append(f"{c['latency_sec']}s")
        if c.get("error"): meta.append(f"error={c['error']}")
        out.append(f"#### Step {c['call_idx'] + 1} · {label}" + (f"  _({', '.join(meta)})_" if meta else ""))
        if not args.no_prompt:
            out.append("**Query (prompt sent):**")
            out.append(fence(clip(c["prompt"], args.prompt_chars)))
        th = c.get("thinking")
        out.append("**Thought:**")
        out.append(quote(clip(th, args.think_chars)) if th else "_(not available — cached answer or backend exposes no reasoning)_")
        out.append("")
        out.append(f"**Output:** {c.get('output')}\n")
    out.append(f"### Final answer\n\n{r.get('final_answer')}\n\n_Verdict: {verdict(r)}_\n\n---\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_file")
    ap.add_argument("--out", default=None, help="Markdown output path (default: next to the .jsonl)")
    ap.add_argument("--items", default=None, help="comma-separated item_idx values")
    ap.add_argument("--only", choices=["all", "wrong", "attacked", "correct"], default="all")
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--full", action="store_true", help="do not clip long texts")
    ap.add_argument("--no_prompt", action="store_true", help="omit the prompt text of each step")
    ap.add_argument("--no_contexts", action="store_true")
    ap.add_argument("--csv", default=None, help="also write one row per LLM call to this CSV")
    args = ap.parse_args()
    args.ctx_chars = None if args.full else 300
    args.prompt_chars = None if args.full else 1500
    args.think_chars = None if args.full else 4000

    recs = load(args.trace_file)
    if args.items:
        keep = {int(x) for x in args.items.split(",")}
        recs = [r for r in recs if r["item_idx"] in keep]
    if args.only == "wrong":
        recs = [r for r in recs if r.get("correct") is False]
    elif args.only == "correct":
        recs = [r for r in recs if r.get("correct") is True]
    elif args.only == "attacked":
        recs = [r for r in recs if r.get("attack_success")]
    if args.max_items:
        recs = recs[: args.max_items]

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rep", "item_idx", "defense", "question", "call_idx", "stage", "batch_idx",
                        "prompt", "thinking", "output", "cached", "completion_tokens", "final_answer", "correct", "attack_success"])
            for r in recs:
                for c in r.get("calls", []):
                    w.writerow([r["rep"], r["item_idx"], r.get("defense"), r["question"], c["call_idx"], c.get("stage"),
                                c.get("batch_idx"), c["prompt"], c.get("thinking"), c.get("output"), c.get("cached"),
                                c.get("completion_tokens"), r.get("final_answer"), r.get("correct"), r.get("attack_success")])
        print("wrote", args.csv)

    out = args.out or os.path.splitext(args.trace_file)[0] + ".md"
    n_think = sum(1 for r in recs for c in r.get("calls", []) if c.get("thinking"))
    n_calls = sum(len(r.get("calls", [])) for r in recs)
    head = (f"# Thinking traces — {os.path.basename(args.trace_file)}\n\n"
            f"{len(recs)} questions · {n_calls} LLM calls · {n_think} with a thinking trace\n\n---\n")
    with open(out, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(render_item(r, args) for r in recs))
    print("wrote", out)


if __name__ == "__main__":
    main()

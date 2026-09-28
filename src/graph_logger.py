"""Opt-in logging of the contradiction graphs built by the graph, MIS and
sampleMIS defenses.

Enabled with `main.py --log_graphs`. One JSON object per (rep, question) is
appended to  <run_dir>/graphs/<LOG_NAME>.jsonl , so a crashed or partial run
still leaves every graph written so far. Render them with
`python scripts/render_graphs.py <file.jsonl>`.

Flow:
    main.py   -> graph_logger.configure(path)            once
    main.py   -> graph_logger.begin_item(rep, idx, item)  before model.query()
    defense   -> graph_logger.record(...)                 inside query()
    main.py   -> graph_logger.end_item(correct=..., ...)   after grading; writes the line

Everything is a no-op when logging is not configured, so the defenses can call
record() unconditionally.

Record schema (one line):
    method            "graph" | "MIS" | "sampleMIS"
    rep, item_idx     position in the run
    question, gold_answers, incorrect_answer
    poison_positions  0-based ranks of injected documents (from src/attack.py)
    directed          True for "graph" (edges point i -> j, i ranked above j)
    nodes             [{id, doc_ranks, answer, idk, poisoned}]
                      doc_ranks = 0-based ranks of the documents behind the node
                      (one per node for graph/MIS; the sampled docs for sampleMIS)
    pairs             every NLI-scored pair, edge or not:
                      [{i, j, p_contra, edge, flipped}]
                      flipped = edge decision was inverted by --err noise
    threshold, err
    selected          node ids kept by the defense
    final_doc_ranks   ranks of documents placed in the final prompt (in order,
                      duplicates kept -- matters for sampleMIS)
    extra             method-specific (e.g. removal rounds for "graph")
    final_answer, correct, attack_success   (filled by end_item)
"""
import json
import logging
import os

logger = logging.getLogger('RRAG-main')

_path = None
_context = None
_pending = None


def _jsonable(o):
    # numpy ints/floats/arrays (e.g. sampleMIS's np.random.choice indices)
    if hasattr(o, 'tolist'):
        return o.tolist()
    if hasattr(o, 'item'):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def configure(path):
    """Turn logging on; records are appended to `path` (JSONL)."""
    global _path
    _path = path
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    logger.info(f'Contradiction graphs will be logged to {path}')


def enabled():
    return _path is not None


def begin_item(rep, item_idx, data_item):
    """Remember which question is being processed."""
    global _context, _pending
    if not enabled():
        return
    ia = data_item.get('incorrect_answer')
    _context = {
        'rep': rep,
        'item_idx': item_idx,
        'question': data_item.get('question'),
        'gold_answers': list(data_item.get('answer', [])) if isinstance(data_item.get('answer'), list) else data_item.get('answer'),
        'incorrect_answer': ia if isinstance(ia, str) else None,
        'poison_positions': list(data_item.get('poison_positions', [])),
    }
    _pending = None


def poison_positions():
    return set(_context['poison_positions']) if _context else set()


def record(method, nodes, pairs, selected, final_doc_ranks,
           threshold=0.5, err=0.0, directed=False, extra=None):
    """Store the graph for the current question (written by end_item)."""
    global _pending
    if not enabled():
        return
    poisoned = poison_positions()
    for n in nodes:
        n.setdefault('poisoned', any(r in poisoned for r in n.get('doc_ranks', [])))
        n['answer'] = str(n.get('answer', ''))
    _pending = {
        'method': method,
        'directed': directed,
        'nodes': nodes,
        'pairs': [{'i': int(p['i']), 'j': int(p['j']), 'p_contra': round(float(p['p_contra']), 4),
                   'edge': bool(p['edge']), 'flipped': bool(p.get('flipped', False))} for p in pairs],
        'threshold': threshold,
        'err': err,
        'selected': sorted(int(s) for s in selected),
        'final_doc_ranks': [int(r) for r in final_doc_ranks],
        'extra': extra or {},
    }
    edges = [(p['i'], p['j']) for p in _pending['pairs'] if p['edge']]
    logger.info(f'[graph] {method} nodes={len(nodes)} edges={edges} selected={_pending["selected"]}')


def end_item(final_answer=None, correct=None, attack_success=None):
    """Write the pending record (if the defense produced one)."""
    global _pending
    if not enabled() or _pending is None:
        return
    rec = dict(_context or {})
    rec.update(_pending)
    rec['final_answer'] = None if final_answer is None else str(final_answer)
    rec['correct'] = None if correct is None else bool(correct)
    rec['attack_success'] = None if attack_success is None else bool(attack_success)
    with open(_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=_jsonable) + '\n')
    _pending = None

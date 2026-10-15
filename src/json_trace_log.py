"""JSON log of the model's thinking traces, one document per run.

Written to  <run_dir>/log/<LOG_NAME>.traces.json  and rewritten (atomically)
after every query, so an interrupted run keeps every query logged so far.
Turn it off with LOG_REASONING=0; LOG_REASONING_MAX_CHARS=N truncates each
thinking trace to N characters (default 0 = keep the full trace).

Layout
------
{
  "run_arguments": {...},              # argparse namespace of main.py
  "config":        {...},              # resolved model / defense / attack / dataset / grader setup
  "queries": [                         # one entry per query, in processing order
    {
      "serial_no": 1,                  # 1-based, counts across reps
      "rep": 0, "item_idx": 0,
      "query": "<original question>",
      "gold_answers": [...], "incorrect_answer": ..., "poison_positions": [...],

      "context_wise_reasoning": [      # a. one entry per prompt asked per context item
        {"context_index": 0, "context": "...", "augmented_query": "<full prompt>",
         "thinking": "...", "predicted_answer": "...", "cached": false, "usage": {...}}
      ],
      "graph_building": {              # b. contradiction graph (graph / MIS / sampleMIS)
        "graph_type": "MIS", "nodes": [0, 1, ...], "edges": [[0, 3], ...],
        "majority_mis_cluster": [1, 2, ...], ...
      },
      "judgment_and_evaluation": {     # c. final aggregation query + grading
        "prompt": "<final prompt>",
        "all_answers": [{"serial_no": 1, "context_index": 0, "answer": "..."}],
        "final_answer": "...",
        "final_answer_thinking": "...",
        "judgment": {"correct": true, "grade": "A", "grade_label": "CORRECT",
                     "grader_thinking": "...", "attack_success": false}
      },
      "other_llm_calls": [...]         # only present if some call was not attributed above
    }
  ],
  "summary": [ {per-rep accuracy / ASR / tokens / time} ]
}

Flow
----
  main.py         configure(path, run_arguments, config)        once
  main.py         begin_query(rep, item_idx, data_item)         before the defense runs
  models.py       record_call(...)                              on every LLM call (incl. cache hits)
  defense code    log_context_reasoning / log_graph / log_final
  main.py         set_judgment(...) ; end_query()                after grading
  main.py         add_summary(...)                              after each rep
"""
import json
import logging
import os

logger = logging.getLogger('RRAG-main')

_path = None
_doc = None
_current = None
_calls = []          # LLM calls made while the current query is being processed
_serial = 0


# ───────────────────────────── setup ──────────────────────────────
def _env_enabled():
    return os.environ.get("LOG_REASONING", "1").lower() not in ("0", "false", "no", "off")


def _max_chars():
    try:
        return int(os.environ.get("LOG_REASONING_MAX_CHARS", "0"))
    except ValueError:
        return 0


def configure(path, run_arguments, config):
    """Start the JSON trace file: run arguments and config go at the top."""
    global _path, _doc, _serial
    if not _env_enabled():
        logger.info('Thinking-trace JSON log disabled (LOG_REASONING=0)')
        return
    _path = path
    _serial = 0
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    _doc = {
        'run_arguments': dict(run_arguments),
        'config': config,
        'queries': [],
        'summary': [],
    }
    _flush()
    logger.info(f'Thinking traces (JSON) will be logged to {path}')


def enabled():
    return _path is not None


def _clip(text):
    if text is None:
        return None
    text = str(text)
    n = _max_chars()
    if n > 0 and len(text) > n:
        return text[:n] + f" ...[truncated, {len(text)} chars total]"
    return text


def _default(o):
    # numpy scalars / arrays (e.g. sampled doc indices) -> plain JSON numbers / lists
    if hasattr(o, 'tolist'):
        return o.tolist()
    if hasattr(o, 'item'):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _flush():
    if _path is None:
        return
    tmp = _path + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(_doc, f, indent=2, ensure_ascii=False, default=_default)
        os.replace(tmp, _path)
    except OSError:
        # e.g. the file is open in a viewer on Windows: write in place instead
        with open(_path, 'w', encoding='utf-8') as f:
            json.dump(_doc, f, indent=2, ensure_ascii=False, default=_default)


# ─────────────────────────── LLM calls ────────────────────────────
def record_call(prompt, answer, thinking=None, usage=None, cached=False, error=None):
    """Called by the model wrappers for every LLM call (fresh or cache hit)."""
    if not enabled() or _current is None:
        return
    rec = {'prompt': str(prompt),
           'answer': None if answer is None else str(answer),
           'thinking': _clip(thinking),
           'cached': bool(cached)}
    if usage:
        rec['usage'] = usage
    if error:
        rec['error'] = str(error)
    _calls.append(rec)


def _take_call(prompt):
    """Pop the first not-yet-attributed call made with this exact prompt."""
    for i, c in enumerate(_calls):
        if c['prompt'] == prompt:
            return _calls.pop(i)
    return None


# ─────────────────────────── per query ────────────────────────────
def begin_query(rep, item_idx, data_item):
    global _current, _calls, _serial
    if not enabled():
        return
    _serial += 1
    ia = data_item.get('incorrect_answer')
    _current = {
        'serial_no': _serial,
        'rep': rep,
        'item_idx': item_idx,
        'query': data_item.get('question'),
        'gold_answers': list(data_item.get('answer', [])),
        'incorrect_answer': ia if isinstance(ia, str) else (ia or None),
        'poison_positions': list(data_item.get('poison_positions', [])),
        'context_wise_reasoning': [],
        'graph_building': None,
        'judgment_and_evaluation': {},
    }
    _calls = []


def log_context_reasoning(contexts, prompts, answers, context_indices=None, extra=None):
    """a. One entry per prompt asked per context item.

    contexts: the context of each item (a document, or a list of documents for
    sampled subsets); prompts: the full augmented query sent for each item;
    answers: what the model returned; extra: optional per-item dicts merged in.
    """
    if not enabled() or _current is None:
        return
    poisoned = set(_current.get('poison_positions') or [])
    for n, (ctx, prompt, ans) in enumerate(zip(contexts, prompts, answers)):
        call = _take_call(prompt) or {}
        idx = context_indices[n] if context_indices is not None else n
        entry = {
            'context_index': idx,
            'context': ctx,
            'augmented_query': prompt,
            'thinking': call.get('thinking'),
            'predicted_answer': ans,
            'cached': call.get('cached', False),
        }
        if isinstance(idx, int) and poisoned and context_indices is None:
            entry['poisoned'] = idx in poisoned
        if extra is not None and n < len(extra) and extra[n]:
            entry.update(extra[n])
        if 'usage' in call:
            entry['usage'] = call['usage']
        if 'error' in call:
            entry['error'] = call['error']
        _current['context_wise_reasoning'].append(entry)


def log_graph(graph_type, nodes, edges, cluster, cluster_key='majority_mis_cluster', **extra):
    """b. The contradiction graph: node indexes, edges, and the chosen cluster."""
    if not enabled() or _current is None:
        return
    g = {'graph_type': graph_type,
         'nodes': [int(v) for v in nodes],
         'edges': sorted([int(a), int(b)] for a, b in edges),
         cluster_key: sorted(int(v) for v in cluster)}
    g.update(extra)
    _current['graph_building'] = g


def log_final(prompt, answer, all_answers=None, context_indices=None):
    """c. The final aggregation prompt, all intermediate answers and the final answer."""
    if not enabled() or _current is None:
        return
    call = _take_call(prompt) or {}
    je = _current['judgment_and_evaluation']
    je['prompt'] = prompt
    if all_answers is not None:
        je['all_answers'] = [
            {'serial_no': n + 1,
             'context_index': context_indices[n] if context_indices is not None else n,
             'answer': a}
            for n, a in enumerate(all_answers)]
    je['final_answer'] = answer
    je['final_answer_thinking'] = call.get('thinking')
    je['cached'] = call.get('cached', False)
    if 'usage' in call:
        je['usage'] = call['usage']
    if 'error' in call:
        je['error'] = call['error']


_GRADE_LABELS = {'A': 'CORRECT', 'B': 'INCORRECT', 'C': 'NOT_ATTEMPTED'}


def set_judgment(correct, grade=None, grader_prompt=None, grader_thinking=None,
                 attack_success=None, extracted_answer=None, judge_thinking=None):
    if not enabled() or _current is None:
        return
    je = _current['judgment_and_evaluation']
    j = {'correct': bool(correct)}
    if grade is not None:
        j['grade'] = grade
        j['grade_label'] = _GRADE_LABELS.get(grade, grade)
    if grader_prompt is not None:
        j['grader_prompt'] = grader_prompt
    if grader_thinking is not None:
        j['grader_thinking'] = _clip(grader_thinking)
    if attack_success is not None:
        j['attack_success'] = bool(attack_success)
    if extracted_answer is not None:
        # astuterag / instructrag_icl: answer extracted by the LLM judge, which is what gets graded
        j['judge_extracted_answer'] = extracted_answer
        j['judge_thinking'] = _clip(judge_thinking)
    je['judgment'] = j


def end_query(final_answer=None):
    """Close the current query and rewrite the JSON file."""
    global _current, _calls
    if not enabled() or _current is None:
        return
    je = _current['judgment_and_evaluation']
    leftovers = list(_calls)
    if 'prompt' not in je and leftovers:
        # defenses without explicit hooks (none / keyword / sampling ...):
        # their last LLM call is the one that produced the final answer
        last = leftovers.pop()
        je['prompt'] = last['prompt']
        je['final_answer'] = last['answer']
        je['final_answer_thinking'] = last['thinking']
        je['cached'] = last['cached']
    if 'final_answer' not in je and final_answer is not None:
        je['final_answer'] = final_answer
    # keep a fixed key order inside the section
    order = ['prompt', 'all_answers', 'final_answer', 'final_answer_thinking', 'judgment']
    _current['judgment_and_evaluation'] = (
        {k: je[k] for k in order if k in je} | {k: v for k, v in je.items() if k not in order})
    if not _current['context_wise_reasoning']:
        del _current['context_wise_reasoning']
    if _current['graph_building'] is None:
        del _current['graph_building']
    if leftovers:
        _current['other_llm_calls'] = leftovers
    _doc['queries'].append(_current)
    _current, _calls = None, []
    _flush()


def add_summary(result):
    if not enabled():
        return
    _doc['summary'].append(result)
    _flush()

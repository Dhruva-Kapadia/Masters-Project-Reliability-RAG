"""Opt-in logging of the model's thinking traces: Query -> Thought -> Output.

Enabled with `main.py --save_traces`. One JSON object per (rep, question) is
appended to  <run_dir>/traces/<LOG_NAME>.jsonl , so a crashed run still keeps
every question written so far. Render with
    python scripts/render_traces.py <run_dir>/traces/<LOG_NAME>.jsonl

Flow (mirrors src/graph_logger.py):
    main.py  -> trace_logger.configure(path)                       once
    main.py  -> trace_logger.begin_item(rep, idx, item, defense)    after the attack, before model.query()
    models   -> trace_logger.record_call(prompt, output, thinking)  on every LLM call
    main.py  -> trace_logger.end_item(final_answer, correct, ...)   after grading; writes the line

Record schema (one line):
    rep, item_idx, defense, question, gold_answers, incorrect_answer
    poison_positions      0-based ranks of injected documents
    contexts              [{rank, text, poisoned}] -- the documents the defense received
    calls                 every LLM call made while answering, in order:
        call_idx
        stage             "per_doc"   : one prompt per document / sample (batch_query)
                          "aggregate" : a single prompt (final / keyword-hint / sampled-subset query)
        batch_idx         position inside the batch (per_doc only)
        prompt            exact text sent as the user message
        thinking          the model's reasoning (gpt-oss analysis channel); None if the
                          backend does not expose it or the answer came from the cache
        output            the visible answer
        cached            True if served from --use_cache (no thinking available)
        finish_reason, prompt_tokens, completion_tokens, latency_sec
    final_answer, correct, attack_success
"""
import json
import logging
import os

logger = logging.getLogger('RRAG-main')

_path = None
_context = None
_calls = None
_stage = None       # set by models.BaseModel around batch_query
_batch_idx = None


def configure(path):
    global _path
    _path = path
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    logger.info(f'Thinking traces will be logged to {path}')


def enabled():
    return _path is not None


def begin_item(rep, item_idx, data_item, defense=None):
    global _context, _calls
    if not enabled():
        return
    poisoned = set(data_item.get('poison_positions', []))
    ia = data_item.get('incorrect_answer')
    _context = {
        'rep': rep,
        'item_idx': item_idx,
        'defense': defense,
        'question': data_item.get('question'),
        'gold_answers': list(data_item.get('answer', [])),
        'incorrect_answer': ia if isinstance(ia, str) else None,
        'poison_positions': sorted(poisoned),
        'contexts': [{'rank': i, 'text': t, 'poisoned': i in poisoned}
                     for i, t in enumerate(data_item.get('topk_content', []))],
    }
    _calls = []


def set_stage(stage, batch_idx=None):
    """Called by the model wrapper so each call knows whether it is part of a batch."""
    global _stage, _batch_idx
    _stage, _batch_idx = stage, batch_idx


def record_call(prompt, output, thinking=None, cached=False, **meta):
    if not enabled() or _calls is None:
        return
    rec = {
        'call_idx': len(_calls),
        'stage': _stage or 'aggregate',
        'batch_idx': _batch_idx,
        'prompt': str(prompt),
        'thinking': None if thinking is None else str(thinking),
        'output': None if output is None else str(output),
        'cached': bool(cached),
    }
    rec.update({k: v for k, v in meta.items() if v is not None})
    _calls.append(rec)


def end_item(final_answer=None, correct=None, attack_success=None):
    global _context, _calls
    if not enabled() or _context is None:
        return
    rec = dict(_context)
    rec['calls'] = _calls or []
    rec['final_answer'] = None if final_answer is None else str(final_answer)
    rec['correct'] = None if correct is None else bool(correct)
    rec['attack_success'] = None if attack_success is None else bool(attack_success)
    with open(_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
    _context, _calls = None, None


def split_harmony(text):
    """If the server returns raw harmony text (no reasoning parser), split
    'analysis ... assistantfinal ...' into (thinking, answer)."""
    if not text:
        return None, text
    for marker in ('assistantfinal', '<|channel|>final<|message|>'):
        idx = text.rfind(marker)
        if idx != -1:
            thinking = text[:idx]
            for pre in ('<|channel|>analysis<|message|>', 'analysis'):
                if thinking.startswith(pre):
                    thinking = thinking[len(pre):]
                    break
            thinking = thinking.replace('<|end|><|start|>assistant', '').strip()
            return thinking or None, text[idx + len(marker):].strip()
    return None, text

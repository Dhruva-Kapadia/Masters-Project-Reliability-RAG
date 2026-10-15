"""Extract gpt-oss reasoning traces from chat completions.

gpt-oss is a reasoning model: besides the final answer, the server returns its
hidden chain of thought in a separate message field. Where it lives depends on
the backend:
  - Groq:               message.reasoning           (on by default)
  - vLLM >= ~0.11:      message.reasoning
  - vLLM 0.10.x:        message.reasoning_content
This module pulls it out of whichever field is present. The traces are written
as structured JSON by src/json_trace_log.py (<run_dir>/log/<run>.traces.json),
no longer as text blocks in <run>.log.

Environment knobs (read by src/json_trace_log.py):
  LOG_REASONING=0            turn the JSON trace log off (default: on)
  LOG_REASONING_MAX_CHARS=N  truncate each reasoning trace to N chars (default 0 = full)

With --use_cache the reasoning is cached next to the answer, so cache hits
still carry their thinking (answers cached before this change have none).
"""
import logging

logger = logging.getLogger('RRAG-main')

_warned_missing = set()


def extract_reasoning(message):
    """Return the reasoning text from a chat completion message, or None."""
    if message is None:
        return None
    for field in ("reasoning", "reasoning_content"):
        value = getattr(message, field, None)
        if value is None:
            extra = getattr(message, "model_extra", None) or {}
            value = extra.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def usage_dict(chat):
    """Token usage of a completion as a plain dict (or None)."""
    usage = getattr(chat, "usage", None)
    if usage is None:
        return None
    details = getattr(usage, "completion_tokens_details", None)
    out = {"prompt_tokens": getattr(usage, "prompt_tokens", None),
           "completion_tokens": getattr(usage, "completion_tokens", None)}
    r_tokens = getattr(details, "reasoning_tokens", None) if details is not None else None
    if r_tokens is not None:
        out["reasoning_tokens"] = r_tokens
    return out


def log_reasoning(source, model, prompt, chat):
    """Return the reasoning trace (or None) of one completion.

    source: who made the call -- "answer" (model under test), "grader" or "judge".
    Warns once per (source, model) if the server returns no reasoning field.
    """
    try:
        message = chat.choices[0].message if chat is not None and chat.choices else None
        reasoning = extract_reasoning(message)
        if reasoning is None and (source, model) not in _warned_missing:
            _warned_missing.add((source, model))
            logger.warning(
                f"[reasoning] {source}/{model}: server returned no reasoning field "
                f"(non-reasoning model, or server not configured to return it)")
        return reasoning
    except Exception as exc:  # logging must never break a run
        logger.debug(f"[reasoning] failed to extract trace: {exc}")
        return None

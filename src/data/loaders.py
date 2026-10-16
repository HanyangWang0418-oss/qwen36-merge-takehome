"""Row -> {"messages": [...], "meta": {...}} converters for each public source,
written against the *real* schemas (captured in tests/fixtures/<loader>.jsonl).

Output contract (what src/data/chat_format.py and the trainer consume):
  messages: list of {"role": system|user|assistant|tool, "content": str, ...}
            assistant turns may carry OpenAI-style "tool_calls":
              [{"id", "type": "function", "function": {"name", "arguments": <json str>}}]
            tool turns carry "tool_call_id" (and "name" when the source has it).
  tools:    (optional, agent sources only) the tool schemas the trajectory was
            produced with, so the chat template can render them.
  meta:     source-specific fields; the contamination filter reads
            meta.repo, meta.instance_id, meta.task_name; tiers read meta.n_turns.

Every loader must set when known meta.repo / meta.instance_id / meta.task_name.
Unknown shapes return None (counted as "unparsable") rather than guessing, so a
silent format drift never produces garbage training data. Loaders that apply a
quality rule (e.g. resolved-only) return the string "unresolved" instead of None
so the builder can count that separately.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Iterator, Optional, Union

ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
            "system": "system", "tool": "tool", "function": "tool"}
Converted = Union[dict, str, None]  # dict | "unresolved" | None


# ----------------------------------------------------------------- helpers
def _text(content) -> str:
    """Flatten Anthropic/OpenAI content blocks ([{type: text, text: ...}]) to str."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and b.get("type", "text") == "text":
                parts.append(b.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _tool_calls(tcs) -> Optional[list]:
    """Normalise tool calls to OpenAI format; arguments always a JSON string."""
    if not tcs:
        return None
    out = []
    for tc in tcs:
        fn = tc.get("function") or {}
        args = fn.get("arguments", "{}")
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        out.append({"id": tc.get("id") or f"call_{len(out)}", "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": args}})
    return out


def _openai_messages(msgs: list) -> Optional[list]:
    """OpenAI-style rows (SWE-Gym, Nemotron, SWE-smith): keep role/content/tool_calls/tool_call_id."""
    norm = []
    for m in msgs:
        role = ROLE_MAP.get(m.get("role"))
        if role is None:
            return None
        out = {"role": role, "content": _text(m.get("content"))}
        if role == "assistant":
            tcs = _tool_calls(m.get("tool_calls"))
            if tcs:
                out["tool_calls"] = tcs
        if role == "tool":
            tid = m.get("tool_call_id") or m.get("id") or (m.get("tool_call_ids") or [None])[0]
            if tid:
                out["tool_call_id"] = tid
            if m.get("name"):
                out["name"] = m["name"]
        norm.append(out)
    return norm


def _sharegpt_messages(convs: list) -> Optional[list]:
    """ShareGPT-style rows ({from: human|gpt|system, value}) -> plain-text turns."""
    norm = []
    for m in convs:
        role = ROLE_MAP.get(m.get("from") or m.get("role"))
        val = m.get("value", m.get("content"))
        if role is None or val is None:
            return None
        norm.append({"role": role, "content": _text(val)})
    return norm


def _finish(messages: Optional[list], meta: dict, tools=None) -> Optional[dict]:
    if not messages or messages[-1]["role"] != "assistant":
        return None
    if not any(m["role"] == "user" for m in messages):
        return None
    meta = {"n_turns": sum(1 for m in messages if m["role"] == "assistant"), **meta}
    ex = {"messages": messages, "meta": meta}
    if tools:
        ex["tools"] = tools
    return ex


def _repo_from_swesmith_id(iid: str) -> str:
    # "django-money__django-money.835c1ab8.func_pm_ctrl_shuffle__viqnyl9u" -> "django-money/django-money"
    head = iid.split(".", 1)[0]
    return head.replace("__", "/", 1) if "__" in head else ""


def _repo_from_swegym_id(iid: str) -> str:
    # "getmoto__moto-5321" -> "getmoto/moto"
    head = iid.rsplit("-", 1)[0]
    return head.replace("__", "/", 1) if "__" in head else ""


# ----------------------------------------------------------------- loaders
def generic_messages(row: dict) -> Converted:
    """Any row with `messages` (OpenAI style) or `conversations` (ShareGPT style)."""
    if row.get("messages"):
        msgs = row["messages"]
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        norm = _openai_messages(msgs)
    elif row.get("conversations"):
        norm = _sharegpt_messages(row["conversations"])
    else:
        return None
    return _finish(norm, {"repo": row.get("repo", ""), "instance_id": row.get("instance_id", ""),
                          "task_name": row.get("id") or row.get("task_name") or row.get("task_id") or ""},
                   tools=row.get("tools"))


def swe_agent_trajectory(row: dict) -> Converted:
    """SWE-bench/SWE-smith-trajectories (config default, split `tool`): `messages` is a
    JSON *string* of SWE-agent messages with content blocks and Anthropic-style
    tool_calls; extra keys (agent, message_type, thought, action, cache_control)
    are dropped. Kept only when `resolved` is true."""
    if "messages" not in row or "instance_id" not in row:
        return None
    if not row.get("resolved", False):
        return "unresolved"
    msgs = row["messages"]
    if isinstance(msgs, str):
        msgs = json.loads(msgs)
    iid = row["instance_id"]
    return _finish(_openai_messages(msgs), {"instance_id": iid, "repo": _repo_from_swesmith_id(iid),
                                             "model": row.get("model", ""), "traj_id": row.get("traj_id", "")})


def openhands_trajectory(row: dict) -> Converted:
    """SWE-Gym/OpenHands-Sampled-Trajectories (split `train.raw`): OpenAI-style
    `messages` with explicit null fields + `tools` schemas + `resolved`. Kept only
    when resolved (about a quarter of the rows)."""
    if "messages" not in row or "instance_id" not in row:
        return None
    if not row.get("resolved", False):
        return "unresolved"
    iid = row["instance_id"]
    return _finish(_openai_messages(row["messages"]), {"instance_id": iid, "repo": _repo_from_swegym_id(iid),
                                                        "run_id": row.get("run_id", "")}, tools=row.get("tools"))


_WORKSPACE = re.compile(r"/workspace/([A-Za-z0-9_.\-]+?)__([A-Za-z0-9_.\-]+?)__[0-9][0-9.]*")


def openhands_sft_trajectory(row: dict) -> Converted:
    """SWE-Gym/OpenHands-SFT-Trajectories (split `train.success.oss`): only `messages`
    (system/user/assistant, actions rendered as text, no tool calls, no ids). All
    491 rows are resolved runs by construction. repo is parsed from the
    `/workspace/<owner>__<repo>__<ver>` path in the task prompt."""
    msgs = row.get("messages")
    if not msgs or len(row) != 1:
        return None
    first_user = next((m.get("content") or "" for m in msgs if m.get("role") == "user"), "")
    mt = _WORKSPACE.search(_text(first_user))
    repo = f"{mt.group(1)}/{mt.group(2)}" if mt else ""
    return _finish(_openai_messages(msgs), {"repo": repo, "instance_id": "", "resolved_by_construction": True})


def nemotron_swe(row: dict) -> Converted:
    """nvidia/Nemotron-SWE-v1 (split `r2e_gym`): OpenHands trajectories, OpenAI-style
    `messages` (tool turns carry `id`), `tools`, `repo`, per-row `license`. The
    release has no resolved flag (it is already a filtered SFT set), so all rows
    pass; the row licence is kept in meta for auditing."""
    if "messages" not in row or "repo" not in row:
        return None
    return _finish(_openai_messages(row["messages"]), {"repo": row["repo"], "instance_id": row.get("uuid", ""),
                                                        "row_license": row.get("license", ""), "dataset": row.get("dataset", "")},
                   tools=row.get("tools"))


def terminus_trajectory(row: dict) -> Converted:
    """Terminus-2 traces (mlfoundations-dev/terminal-bench-traces-local and the
    *-sandboxes-traces-terminus-2 family): ShareGPT-ish `conversations` of
    {role, content} where the agent answers with a JSON action block; no native
    tool calls, so turns stay plain text. `task` (TB task name) or `trial_name`
    gives meta.task_name for the TB2 blocklist."""
    convs = row.get("conversations")
    if not convs:
        return None
    task = row.get("task") or ""
    trial = row.get("trial_name") or ""
    if not task and trial:
        task = trial.split("__")[0]
    return _finish(_sharegpt_messages(convs), {"task_name": task, "trial_name": trial, "episode": row.get("episode", ""),
                                               "model": row.get("model", ""), "agent": row.get("agent", "")})


def agentinstruct(row: dict) -> Converted:
    """zai-org/AgentInstruct (split `os`): ShareGPT `conversations`; the first human
    turn embeds the task prompt; `loss` flags are dropped (every assistant turn is trained)."""
    convs = row.get("conversations")
    if not convs:
        return None
    return _finish(_sharegpt_messages(convs), {"task_name": row.get("id", "")})


def openthoughts(row: dict) -> Converted:
    """open-thoughts/OpenThoughts3-1.2M: ShareGPT `conversations` (human, gpt) where the
    gpt value is `<think>...</think>` + answer; plus difficulty/source/domain."""
    convs = row.get("conversations")
    if not convs:
        return None
    return _finish(_sharegpt_messages(convs), {"domain": row.get("domain", ""), "source_sub": row.get("source", ""),
                                               "difficulty": row.get("difficulty")})


NUMINA_EXCLUDED = {"amc_aime", "aops_forum"}  # competition sub-sources with restrictive terms


def numina(row: dict) -> Converted:
    """AI-MO/NuminaMath-CoT: `problem`, `solution`, `source`, and a ready `messages` pair."""
    if row.get("source", "") in NUMINA_EXCLUDED:
        return "unresolved"  # counted separately as a licence drop
    if "problem" not in row or "solution" not in row:
        return None
    msgs = [{"role": "user", "content": row["problem"]}, {"role": "assistant", "content": row["solution"]}]
    return _finish(msgs, {"domain": "math", "source_sub": row.get("source", "")})


LOADERS: "dict[str, Callable[[dict], Converted]]" = {
    "generic_messages": generic_messages,
    "swe_agent_trajectory": swe_agent_trajectory,
    "openhands_trajectory": openhands_trajectory,
    "openhands_sft_trajectory": openhands_sft_trajectory,
    "nemotron_swe": nemotron_swe,
    "terminus_trajectory": terminus_trajectory,
    "agentinstruct": agentinstruct,
    "openthoughts": openthoughts,
    "numina": numina,
}


# ----------------------------------------------------------------- input
def collapse_prefixes(rows: list, keys: list, field: str = "conversations") -> list:
    """Some trace releases store one row per *step*, each holding the conversation
    prefix up to that step (terminal-bench-traces-local: 1189 rows = 80 trials).
    Keep only the longest row per trial so prefixes are not trained on repeatedly."""
    best = {}
    for r in rows:
        k = tuple(str(r.get(x, "")) for x in keys)
        if k not in best or len(r.get(field) or []) > len(best[k].get(field) or []):
            best[k] = r
    return list(best.values())


def iter_source(src: dict, limit: Optional[int] = None, hf_token: Optional[str] = None) -> Iterator[dict]:
    """Yield raw rows for a source.

    HF sources are read from the page files that src/data/fetch_rows.py wrote to
    data/raw/<source>/rows_*.jsonl (a stratified sample fetched through the
    datasets-server rows API; see that module for why we do not stream shards).
    Local sources read data/raw/<file>. Missing inputs are skipped with a note,
    never fabricated."""
    from .common import ROOT, read_jsonl

    if src.get("local_file"):
        path = ROOT / src["local_file"]
        if not path.exists():
            print(f"  [skip] {src['name']}: {path.relative_to(ROOT)} missing (synthetic source not generated)")
            return
        yield from read_jsonl(path, limit)
        return
    if src.get("hf_repo", "LOCK") == "LOCK":
        print(f"  [skip] {src['name']}: hf_repo not locked")
        return
    pages = sorted((ROOT / "data" / "raw" / src["name"]).glob("rows_*.jsonl"))
    if not pages:
        print(f"  [skip] {src['name']}: no fetched pages under data/raw/{src['name']}/ (run src/data/fetch_rows.py)")
        return
    n = 0
    if src.get("keep_longest_by"):
        rows = [r for p in pages for r in read_jsonl(p)]
        kept = collapse_prefixes(rows, src["keep_longest_by"])
        print(f"  [{src['name']}] collapsed {len(rows)} step rows -> {len(kept)} full trajectories (keep_longest_by={src['keep_longest_by']})")
        yield from kept[:limit]
        return
    for p in pages:
        for row in read_jsonl(p):
            if limit is not None and n >= limit:
                return
            yield row
            n += 1


def fetched_pages(src: dict) -> list:
    """Page files present for a source (for the manifest)."""
    from .common import ROOT

    return [str(p.relative_to(ROOT)) for p in sorted((ROOT / "data" / "raw" / src["name"]).glob("rows_*.jsonl"))]

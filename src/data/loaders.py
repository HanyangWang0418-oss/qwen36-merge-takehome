"""Row -> {"messages": [...], "meta": {...}} converters for each public source.

Every loader must set meta.source, and when known meta.repo / meta.instance_id /
meta.task_name so the contamination filter can act on them. Unknown schemas
raise so that a silent format drift never produces garbage training data."""
from __future__ import annotations

from typing import Callable, Iterator


def _msgs(system: str | None, user: str, assistant: str) -> list[dict]:
    out = []
    if system:
        out.append({"role": "system", "content": system})
    out += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return out


def generic_messages(row: dict) -> dict | None:
    msgs = row.get("messages") or row.get("conversations")
    if not msgs:
        return None
    norm = []
    for m in msgs:
        role = m.get("role") or {"human": "user", "gpt": "assistant", "system": "system"}.get(m.get("from"), None)
        content = m.get("content", m.get("value"))
        if role not in {"system", "user", "assistant", "tool"} or content is None:
            return None
        norm.append({"role": role, "content": content})
    if norm[-1]["role"] != "assistant":
        return None
    return {"messages": norm, "meta": {"repo": row.get("repo", ""), "instance_id": row.get("instance_id", ""),
                                       "n_turns": sum(1 for m in norm if m["role"] == "assistant")}}


def swe_agent_trajectory(row: dict) -> dict | None:
    ex = generic_messages(row)
    if ex is None:
        return None
    iid = row.get("instance_id", "")
    ex["meta"].update({"instance_id": iid, "repo": iid.rsplit("__", 1)[0].replace("__", "/") if "__" in iid else row.get("repo", "")})
    return ex


def openhands_trajectory(row: dict) -> dict | None:
    ex = generic_messages(row)
    if ex is None:
        return None
    ex["meta"].update({"instance_id": row.get("instance_id", ""), "repo": row.get("repo", row.get("instance_id", "").split("__")[0].replace("__", "/"))})
    return ex


def terminus_trajectory(row: dict) -> dict | None:
    ex = generic_messages(row)
    if ex is None:
        return None
    ex["meta"]["task_name"] = row.get("task_id") or row.get("task_name", "")
    return ex


def openthoughts(row: dict) -> dict | None:
    convs = row.get("conversations")
    if convs:
        return generic_messages(row)
    if "problem" in row and "solution" in row:
        return {"messages": _msgs(None, row["problem"], row["solution"]), "meta": {"domain": row.get("domain", "")}}
    return None


def numina(row: dict) -> dict | None:
    if row.get("source", "") in {"amc_aime", "aops_forum"}:  # restrictive licence sub-sources
        return None
    if "problem" not in row or "solution" not in row:
        return None
    return {"messages": _msgs(None, row["problem"], row["solution"]), "meta": {"domain": "math", "source_sub": row.get("source", "")}}


LOADERS: dict[str, Callable[[dict], dict | None]] = {
    "generic_messages": generic_messages,
    "swe_agent_trajectory": swe_agent_trajectory,
    "openhands_trajectory": openhands_trajectory,
    "terminus_trajectory": terminus_trajectory,
    "openthoughts": openthoughts,
    "numina": numina,
}


def iter_source(src: dict, limit: int | None = None, hf_token: str | None = None) -> Iterator[dict]:
    """Yield raw rows from a HF dataset (streaming) or a local jsonl."""
    if src.get("local_file"):
        from .common import ROOT, read_jsonl

        path = ROOT / src["local_file"]
        if not path.exists():
            print(f"  [skip] {src['name']}: {path} missing (run its synth script first)")
            return
        yield from read_jsonl(path, limit)
        return
    if src.get("hf_repo", "LOCK") == "LOCK":
        print(f"  [skip] {src['name']}: hf_repo not locked")
        return
    from datasets import load_dataset

    ds = load_dataset(src["hf_repo"], split=src.get("split", "train"), streaming=True, token=hf_token)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield row

from __future__ import annotations

import re
import unicodedata
from typing import Any

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "player": "user",
    "玩家": "user",
    "主角": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "npc": "assistant",
    "系统": "assistant",
    "旁白": "assistant",
    "narrator": "assistant",
    "bot": "assistant",
    "system": "system",
    "welcome": "system",
    "欢迎词": "system",
}


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\u200b", "").replace("\ufeff", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def normalize_role(role: str) -> str:
    raw = normalize_text(role)
    low = raw.lower()
    if low in ROLE_MAP:
        return ROLE_MAP[low]
    if raw in ROLE_MAP:
        return ROLE_MAP[raw]
    if any(k in raw for k in ["欢迎", "开场", "设定"]):
        return "system"
    if any(k in raw for k in ["系统", "旁白", "叙述"]):
        return "assistant"
    if any(k in raw for k in ["主角", "玩家", "用户"]):
        return "user"
    return "user"


def _infer_conversation_roles(items: list[tuple[str, str]]) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    start = 0
    if items:
        first_role = normalize_role(items[0][0])
        if first_role == "system":
            msgs.append({"role": "system", "content": items[0][1]})
            start = 1
    for idx, (speaker, content) in enumerate(items[start:]):
        role = normalize_role(speaker)
        if role == "system":
            role = "assistant"
        if role == "user" and speaker not in ROLE_MAP and speaker.lower() not in ROLE_MAP:
            role = "user" if idx % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": content})
    return msgs


def as_messages(example: dict[str, Any]) -> list[dict[str, str]] | None:
    if "messages" in example and isinstance(example["messages"], list):
        msgs = []
        for m in example["messages"]:
            if not isinstance(m, dict):
                continue
            role = normalize_role(str(m.get("role", "user")))
            content = normalize_text(str(m.get("content", "")))
            if content:
                msgs.append({"role": role, "content": content})
        return msgs or None

    if "conversations" in example and isinstance(example["conversations"], list):
        pairs: list[tuple[str, str]] = []
        for m in example["conversations"]:
            if not isinstance(m, dict):
                continue
            speaker = normalize_text(str(m.get("from", m.get("role", "user"))))
            content = normalize_text(str(m.get("value", m.get("content", ""))))
            if content:
                pairs.append((speaker, content))
        msgs = _infer_conversation_roles(pairs)
        return msgs or None

    if "instruction" in example and "output" in example:
        ins = normalize_text(str(example.get("instruction", "")))
        out = normalize_text(str(example.get("output", "")))
        if not ins or not out:
            return None
        inp = normalize_text(str(example.get("input", "")))
        user = ins if not inp else f"{ins}\n{inp}"
        return [{"role": "user", "content": user}, {"role": "assistant", "content": out}]

    if "prompt" in example and "response" in example:
        prompt = normalize_text(str(example.get("prompt", "")))
        resp = normalize_text(str(example.get("response", "")))
        if not prompt or not resp:
            return None
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": resp}]

    if "question" in example and "answer" in example:
        q = normalize_text(str(example.get("question", "")))
        a = normalize_text(str(example.get("answer", "")))
        if not q or not a:
            return None
        return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]

    if "text" in example and isinstance(example["text"], str):
        text = normalize_text(example["text"])
        if not text:
            return None
        return [{"role": "user", "content": text}]

    return None


def ensure_chat_pair(messages: list[dict[str, str]]) -> list[dict[str, str]] | None:
    msgs = [m for m in messages if m.get("content")]
    if len(msgs) < 2:
        return None
    if msgs[-1]["role"] != "assistant":
        return None
    if not any(m["role"] == "user" for m in msgs):
        return None
    return msgs


def render_fallback_text(messages: list[dict[str, str]]) -> str:
    buf: list[str] = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            buf.append(f"<|system|>\n{content}\n")
        elif role == "user":
            buf.append(f"<|user|>\n{content}\n")
        else:
            buf.append(f"<|assistant|>\n{content}\n")
    return "\n".join(buf).strip()


def format_for_tokenizer(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return render_fallback_text(messages)

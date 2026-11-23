"""A scripted stand-in for anthropic.Anthropic(). Never touches the network.

The responder gets (model, user_text, call_number_for_this_request) and returns either the
answer text, a (text, stop_reason) pair, or an Exception to raise.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

_LINE = re.compile(r"^(SAP|FUSION)\.(\w+): <UNTRUSTED>(.*)</UNTRUSTED>$")


def fields_from_prompt(user_text: str) -> dict[tuple[str, str], str]:
    """{(record, field): escaped text as shown} parsed back out of a rendered packet."""
    out = {}
    for line in user_text.splitlines():
        m = _LINE.match(line)
        if m:
            out[(m.group(1), m.group(2))] = m.group(3)
    return out


def answer(verdict: str, evidence=None, reason: str = "test", **extra) -> str:
    return json.dumps({"verdict": verdict, "evidence": evidence or [], "reason": reason, **extra})


def quoting(verdict: str):
    """A responder that answers `verdict` with valid quotes from both records' name fields."""
    def respond(model, user_text, n):
        f = fields_from_prompt(user_text)
        ev = [{"record": "SAP", "field": "NAME1", "quote": f[("SAP", "NAME1")][:5] or "x"},
              {"record": "FUSION", "field": "vendor_name", "quote": f[("FUSION", "vendor_name")][:5]}]
        return answer(verdict, [] if verdict == "UNSURE" else ev)
    return respond


def by_model(mapping: dict):
    """Different responder per model (key = model name)."""
    return lambda model, user_text, n: mapping[model](model, user_text, n)


class ScriptedClient:
    def __init__(self, responder):
        self.responder = responder
        self.requests: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        msgs = kwargs["messages"]
        user_text = msgs[0]["content"]
        result = self.responder(kwargs["model"], user_text, (len(msgs) + 1) // 2)
        if isinstance(result, Exception):
            raise result
        text, stop = result if isinstance(result, tuple) else (result, "end_turn")
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
            stop_reason=stop,
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200,
                                  cache_read_input_tokens=500, cache_creation_input_tokens=0))

    def calls_for(self, model: str) -> int:
        return sum(r["model"] == model for r in self.requests)

    def prompts(self) -> list[str]:
        """Every piece of text sent: system prompt + all messages."""
        out = []
        for r in self.requests:
            out += [b["text"] for b in r["system"]]
            out += [m["content"] for m in r["messages"]]
        return out

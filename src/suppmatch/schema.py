"""What the judge model may answer (pydantic, extra="forbid").

Only a verdict, evidence quotes and a short reason. No MATCH / NEW / NEEDS_HUMAN, no flags,
no dollars: Python decides those.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record: Literal["SAP", "FUSION"]
    field: str
    quote: str = Field(min_length=1)


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["SAME", "DIFFERENT", "UNSURE"]
    evidence: list[Evidence]
    reason: str


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_judgment(text: str) -> Judgment:
    """Model text -> Judgment. Raises ValueError (pydantic ValidationError is one) when invalid."""
    body = _FENCE.sub("", (text or "").strip())
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found in the answer")
    return Judgment.model_validate_json(body[start:end + 1])

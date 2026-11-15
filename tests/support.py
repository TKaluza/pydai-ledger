"""Helpers shared by the tests."""

import asyncio
import warnings

from pydantic import BaseModel
from pydantic_ai import BinaryContent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo

from pydai_ledger import (
    Case,
    Turn,
)

PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4


class Answer(BaseModel):
    text: str


def make_cases() -> list[Case]:
    image = BinaryContent(PNG, media_type="image/png")
    return [
        Case(turns=["one"]),
        Case(turns=["a", "b", "c"], expected=[Answer(text="a"), None, {"text": "c"}], key="multi"),
        Case(turns=[Turn(["look", image])]),
        Case(turns=[Turn(["look again", image])], metadata={"page": 2}),
    ]


def run(coro):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # TestModel has no /server_info
        return asyncio.run(coro)


def _always(text: str):
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(text)])

    return respond

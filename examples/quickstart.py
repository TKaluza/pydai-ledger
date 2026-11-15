"""Quickstart: define cases, run them on one or more vLLM servers, inspect the ledger.

Adjust SERVERS to your endpoints. Start vLLM with VLLM_SERVER_DEV_MODE=1 so the full
server configuration is recorded; without it a warning is shown and only
provider-level information is stored.

    uv run python examples/quickstart.py
"""

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from pydai_ledger import Case, Casebook, Journal, RunPlan, Target, execute

# name -> (base URL, served model name)
SERVERS = {
    "qwen3-a": ("http://localhost:8000/v1", "Qwen/Qwen3-32B"),
    # "qwen3-b": ("http://other-host:8000/v1", "Qwen/Qwen3-32B"),
}


class SimpleAnswer(BaseModel):
    text: str = Field(description="Text of your answer")


bond007 = Agent(
    instructions="You have to strictly follow the user requests. "
    "Just send the answer in the provided JSON schema, nothing else!",
    model_settings={"temperature": 0, "seed": 7},
    output_type=SimpleAnswer,
)


def prepare(path: Path) -> Casebook:
    """Step 1: define the cases, independent of any agent, model or server.

    A case with several turns is one conversation: every later turn sees the full
    message history. For images pass a list, e.g.
    ``Turn(["Describe this page", BinaryContent(png_bytes, media_type="image/png")])``.
    """
    cases = [
        Case(turns=["Say 'Hello there!'"]),
        Case(turns=["Say 'A'", "Say 'B'", "What do they make together?"]),
        Case(
            turns=["Say 'C'", "Say 'D'", "What do they make together?"],
            expected=[SimpleAnswer(text="C"), SimpleAnswer(text="D"), SimpleAnswer(text="CD")],
        ),
        Case(
            turns=["Say 'E'", "Say 'F'", "What do they make together?"],
            expected=[SimpleAnswer(text="E"), SimpleAnswer(text="F"), SimpleAnswer(text="EF")],
        ),
    ]
    return Casebook.write(path, cases)


async def run(casebook: Casebook, destination: Path):
    """Step 2: run the casebook and seal the result into a ledger."""
    targets = [
        Target(name, OpenAIChatModel(model, provider=OpenAIProvider(base_url=url, api_key="EMPTY")))
        for name, (url, model) in SERVERS.items()
    ]
    plan = RunPlan(
        casebook=casebook,
        agent=bond007,
        targets=targets,
        attempts_per_case=2,
        concurrency=[1, 4],  # two levels = a small throughput sweep
    )
    with Journal.open(plan, parent="tmp") as journal:
        await execute(plan, journal)  # interrupted? run again, it resumes
        return journal.seal(destination, require_complete=True, cleanup=True)


async def main() -> None:
    casebook = prepare(Path("tuggy.casebook.zip"))
    print(casebook)

    ledger = await run(casebook, Path("tuggy.ledger.zip"))
    print(ledger)

    agreement = ledger.agreement()
    print(f"agreement: {agreement.agreeing}/{agreement.compared} turns ({agreement.rate:.0%})")
    for row in ledger.throughput():
        print(row)


if __name__ == "__main__":
    asyncio.run(main())

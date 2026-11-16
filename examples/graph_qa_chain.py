"""Question answering over the Silicon Valley dataset with GrafeoQAChain.

Demonstrates:
- Questions in plain language, answered through a generated GQL query
- A graph the store did not write: the dataset names entities by `name`, which the chain detects
- A failing query fixed by a second LLM call
- A request to change the graph refused, with the graph left unchanged

Uses Mistral when MISTRAL_API_KEY is set, and a scripted stand-in for the LLM otherwise. Run:
    uv run python examples/graph_qa_chain.py                                  # no API key
    uv run --with langchain-mistralai python examples/graph_qa_chain.py      # with MISTRAL_API_KEY
"""

from __future__ import annotations

import importlib
import os

from grafeo_datasets import silicon_valley
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.fake import FakeListLLM

from grafeo_langchain import GrafeoGraphStore, GrafeoQAChain

QUESTIONS = ["Does Jian-Yang's fridge talk?", "What did Jian-Yang build?", "How does Son of Anton fix bugs?"]
CHANGE = "Delete Jian-Yang's fridge from the graph."

# What the stand-in replies: per question a query (one broken, then fixed) and an answer.
SCRIPT = [
    "MATCH (f) WHERE toLower(f.name) CONTAINS 'fridge' RETURN f.talks AS talks",
    "Yes, Jian-Yang's fridge talks.",
    "MATCH (p)-[:BUILT]->(x) WHERE toLower(p.name) CONTAINS 'jian-yang' RETURN x.name AS built",
    "Jian-Yang built Not Hotdog and SeeFood.",
    "MATCH (a WHERE toLower(a.name) CONTAINS 'son of anton' RETURN a.bug_fix_strategy AS strategy",  # missing ")"
    "MATCH (a) WHERE toLower(a.name) CONTAINS 'son of anton' RETURN a.bug_fix_strategy AS strategy",
    "Son of Anton fixes bugs by deleting the code.",
    "MATCH (f) WHERE toLower(f.name) CONTAINS 'fridge' DETACH DELETE f",
]


def pick_llm() -> BaseLanguageModel:
    if os.environ.get("MISTRAL_API_KEY"):
        # optional (uv run --with langchain-mistralai): imported by name, so type checkers do not require it
        ChatMistralAI = importlib.import_module("langchain_mistralai").ChatMistralAI
        print("LLM: Mistral (mistral-small-latest)")
        return ChatMistralAI(model="mistral-small-latest", temperature=0)
    print("LLM: scripted stand-in (set MISTRAL_API_KEY to use Mistral)")
    return FakeListLLM(responses=SCRIPT)


store = GrafeoGraphStore()
silicon_valley.load(store.client)
chain = GrafeoQAChain.from_llm(pick_llm(), graph=store, return_intermediate_steps=True)

for question in QUESTIONS:
    out = chain.invoke({"query": question})
    print(f"\n=== {question} ===")
    print("  query: ", " ".join(out["intermediate_steps"][0]["query"].split()))
    print("  rows:  ", out["intermediate_steps"][1]["context"])
    print("  answer:", out["result"])

print(f"\n=== {CHANGE} ===")
before = store.query("MATCH (n) RETURN count(n) AS n")[0]["n"]
try:
    chain.invoke({"query": CHANGE})
    print("  answered without changing the graph")
except ValueError as error:
    print("  refused:", str(error).split(", and")[0].split(": ")[0])
after = store.query("MATCH (n) RETURN count(n) AS n")[0]["n"]
print(f"  nodes before and after: {before}, {after}")

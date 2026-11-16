from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path
from typing import Any

import grafeo
import pytest
from langchain_core.callbacks import BaseCallbackHandler, CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.language_models import LLM
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import Field, ValidationError

from grafeo_langchain import GrafeoGraphStore, GrafeoQAChain, GraphDocument, Node, Relationship
from grafeo_langchain.qa_chain import extract_query, looks_like_query

WORKS_AT = "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.node_id AS person, c.node_id AS company ORDER BY person"


class ScriptedLLM(LLM):
    """Replies with ``responses`` in turn and records every prompt; an unexpected call fails the test."""

    responses: list[str]
    prompts: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            msg = f"unexpected LLM call with prompt:\n{prompt}"
            raise AssertionError(msg)
        return self.responses.pop(0)


def _people(store: GrafeoGraphStore) -> None:
    alice = Node(id="Alice", type="Person", properties={"age": 30})
    bob = Node(id="Bob", type="Person", properties={"age": 25})
    carol = Node(id="Carol", type="Person", properties={"age": 35})
    microsoft = Node(id="Microsoft", type="Company", properties={"founded": 1975})
    google = Node(id="Google", type="Company", properties={"founded": 1998})
    store.add_graph_documents(
        [
            GraphDocument(
                nodes=[alice, bob, carol, microsoft, google],
                relationships=[
                    Relationship(source=alice, target=microsoft, type="WORKS_AT", properties={"since": 2020}),
                    Relationship(source=bob, target=google, type="WORKS_AT"),
                    Relationship(source=carol, target=microsoft, type="WORKS_AT"),
                    Relationship(source=alice, target=bob, type="KNOWS"),
                ],
                source=Document(
                    page_content="Alice and Carol work at Microsoft, Bob at Google.", metadata={"id": "d1"}
                ),
            )
        ],
        include_source=True,
    )


def _snapshot(store: GrafeoGraphStore) -> list[dict[str, Any]]:
    return store.query("MATCH (n) OPTIONAL MATCH (n)-[r]->() RETURN n, collect(r) AS rels ORDER BY n.node_id")


@pytest.fixture
def store() -> GrafeoGraphStore:
    store = GrafeoGraphStore()
    _people(store)
    return store


def _chain(store: GrafeoGraphStore, *responses: str, **kwargs: Any) -> tuple[GrafeoQAChain, ScriptedLLM]:
    llm = ScriptedLLM(responses=list(responses))
    return GrafeoQAChain.from_llm(llm, graph=store, **kwargs), llm


# ── Answering ────────────────────────────────────────────────────────────────


class TestAnswering:
    def test_answers_question(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, WORKS_AT, "Alice and Carol work at Microsoft.")
        out = chain.invoke({"query": "Who works at Microsoft?"})
        assert out == {"query": "Who works at Microsoft?", "result": "Alice and Carol work at Microsoft."}
        assert len(llm.prompts) == 2
        assert llm.responses == [], "both scripted replies were used"

    def test_string_input(self, store: GrafeoGraphStore) -> None:
        chain, _ = _chain(store, WORKS_AT, "answer")
        assert chain.invoke("Who works where?") == {"query": "Who works where?", "result": "answer"}

    def test_custom_keys(self, store: GrafeoGraphStore) -> None:
        chain, _ = _chain(store, WORKS_AT, "answer", input_key="question", output_key="answer")
        assert chain.invoke({"question": "Who?"}) == {"question": "Who?", "answer": "answer"}

    def test_missing_input_key(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store)
        with pytest.raises(ValueError, match=r"no 'query' key .*\['question'\]"):
            chain.invoke({"question": "Who?"})
        assert llm.prompts == []

    def test_query_prompt_holds_schema_and_question(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, WORKS_AT, "answer")
        chain.invoke("Who works at Microsoft?")
        prompt = llm.prompts[0]
        assert "Who works at Microsoft?" in prompt
        assert "Person {age: INT64, node_id: STRING}" in prompt
        assert "WORKS_AT {since: INT64}" in prompt
        assert "(:Person)-[:WORKS_AT]->(:Company)" in prompt
        assert "(:Person)-[:KNOWS]->(:Person)" in prompt
        assert "node_id" in prompt

    def test_qa_prompt_holds_rows_and_question(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, WORKS_AT, "answer")
        chain.invoke("Who works at Microsoft?")
        prompt = llm.prompts[1]
        assert "Who works at Microsoft?" in prompt
        assert "{'person': 'Alice', 'company': 'Microsoft'}" in prompt
        assert "{'person': 'Carol', 'company': 'Microsoft'}" in prompt
        assert f"Query: {WORKS_AT}" in prompt, "the answer step sees the query that produced the rows"

    def test_qa_prompt_has_the_fixed_query(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, "MATCH (", Q, "answer")
        chain.invoke("Who?")
        assert f"Query: {Q}" in llm.prompts[2], "the query that ran, not the one that failed"

    def test_chat_model(self, store: GrafeoGraphStore) -> None:
        llm = FakeListChatModel(responses=[WORKS_AT, "Alice, Bob and Carol."])
        chain = GrafeoQAChain.from_llm(llm, graph=store)
        assert chain.invoke("Who works where?")["result"] == "Alice, Bob and Carol."

    def test_separate_llms(self, store: GrafeoGraphStore) -> None:
        query_llm = ScriptedLLM(responses=[WORKS_AT])
        qa_llm = ScriptedLLM(responses=["answer"])
        chain = GrafeoQAChain.from_llm(graph=store, query_llm=query_llm, qa_llm=qa_llm)
        assert chain.invoke("Who?")["result"] == "answer"
        assert "Schema:" in query_llm.prompts[0]
        assert "Query results:" in qa_llm.prompts[0]

    def test_from_llm_with_one_override(self, store: GrafeoGraphStore) -> None:
        llm = ScriptedLLM(responses=[WORKS_AT])
        qa_llm = ScriptedLLM(responses=["answer"])
        chain = GrafeoQAChain.from_llm(llm, graph=store, qa_llm=qa_llm)
        assert chain.invoke("Who?")["result"] == "answer"
        assert (len(llm.prompts), len(qa_llm.prompts)) == (1, 1)

    def test_from_llm_needs_an_llm(self, store: GrafeoGraphStore) -> None:
        with pytest.raises(ValueError, match="pass llm"):
            GrafeoQAChain.from_llm(graph=store)
        with pytest.raises(ValueError, match="pass llm"):
            GrafeoQAChain.from_llm(graph=store, query_llm=ScriptedLLM(responses=[]))

    def test_aggregation(self, store: GrafeoGraphStore) -> None:
        query = "MATCH (p:Person)-[:WORKS_AT]->(c:Company {node_id: 'Microsoft'}) RETURN count(p) AS employees"
        chain, _ = _chain(store, query, "2", return_intermediate_steps=True)
        assert chain.invoke("How many people work at Microsoft?")["intermediate_steps"][1] == {
            "context": [{"employees": 2}]
        }

    def test_case_insensitive_match(self, store: GrafeoGraphStore) -> None:
        query = "MATCH (p:Person)-[:KNOWS]->(f) WHERE toLower(p.node_id) = toLower('alice') RETURN f.node_id AS friend"
        chain, _ = _chain(store, query, "Bob", return_direct=True)
        assert chain.invoke("Who does alice know?")["result"] == [{"friend": "Bob"}]

    def test_relationship_property(self, store: GrafeoGraphStore) -> None:
        query = "MATCH (p:Person {node_id: 'Alice'})-[w:WORKS_AT]->() RETURN w.since AS since"
        chain, _ = _chain(store, query, return_direct=True)
        assert chain.invoke("Since when does Alice work there?")["result"] == [{"since": 2020}]

    def test_source_documents(self, store: GrafeoGraphStore) -> None:
        query = "MATCH (e)-[:MENTIONED_IN]->(d:SourceDocument) RETURN d.node_id AS doc, count(e) AS entities"
        chain, _ = _chain(store, query, return_direct=True)
        assert chain.invoke("Which documents mention what?")["result"] == [{"doc": "source_d1", "entities": 5}]

    def test_empty_result(self, store: GrafeoGraphStore) -> None:
        query = "MATCH (p:Person {node_id: 'Zed'}) RETURN p.age AS age"
        looser = "MATCH (p) WHERE toLower(p.node_id) CONTAINS 'zed' RETURN p.age AS age"
        chain, llm = _chain(store, query, looser, "The graph does not hold the answer.")
        assert chain.invoke("How old is Zed?")["result"] == "The graph does not hold the answer."
        assert "matched nothing" in llm.prompts[1], "an empty result gets one looser try"
        assert "Query results: []" in llm.prompts[2]

    def test_decimal_values(self, store: GrafeoGraphStore) -> None:
        store.query("INSERT (:Product {node_id: 'pen', price: 1.50})")
        chain, llm = _chain(store, "MATCH (p:Product) RETURN p.price AS price", "1.50")
        chain.invoke("What does the pen cost?")
        assert "Decimal('1.50')" in llm.prompts[1]


# ── Result size and output shape ──────────────────────────────────────────────


class TestOutput:
    def test_top_k_limits_context(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, WORKS_AT, "answer", top_k=2, return_intermediate_steps=True)
        out = chain.invoke("Who works where?")
        assert out["intermediate_steps"][1]["context"] == [
            {"person": "Alice", "company": "Microsoft"},
            {"person": "Bob", "company": "Google"},
        ]
        assert "Carol" not in llm.prompts[1]

    def test_default_top_k_is_ten(self, store: GrafeoGraphStore) -> None:
        store.query("UNWIND range(1, 15) AS i INSERT (:Tag {node_id: toString(i)})")
        chain, _ = _chain(store, "MATCH (t:Tag) RETURN t.node_id AS tag", return_direct=True)
        assert len(chain.invoke("Which tags exist?")["result"]) == 10

    def test_return_direct_skips_qa_llm(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, WORKS_AT, return_direct=True)
        out = chain.invoke("Who works where?")
        assert out["result"] == [
            {"person": "Alice", "company": "Microsoft"},
            {"person": "Bob", "company": "Google"},
            {"person": "Carol", "company": "Microsoft"},
        ]
        assert len(llm.prompts) == 1

    def test_return_direct_with_top_k(self, store: GrafeoGraphStore) -> None:
        chain, _ = _chain(store, WORKS_AT, return_direct=True, top_k=1)
        assert chain.invoke("Who?")["result"] == [{"person": "Alice", "company": "Microsoft"}]

    def test_intermediate_steps(self, store: GrafeoGraphStore) -> None:
        chain, _ = _chain(store, f"```gql\n{WORKS_AT}\n```", "answer", return_intermediate_steps=True)
        steps = chain.invoke("Who?")["intermediate_steps"]
        assert steps[0] == {"query": WORKS_AT}, "the query as run, without the code fence"
        assert len(steps[1]["context"]) == 3

    def test_no_intermediate_steps_by_default(self, store: GrafeoGraphStore) -> None:
        chain, _ = _chain(store, WORKS_AT, "answer")
        assert "intermediate_steps" not in chain.invoke("Who?")


# ── Query extraction ─────────────────────────────────────────────────────────


Q = "MATCH (n:Person) RETURN n.node_id AS id"


class TestExtractQuery:
    @pytest.mark.parametrize(
        "reply",
        [
            Q,
            f"  {Q}  \n",
            f"{Q};",
            f"```gql\n{Q}\n```",
            f"```cypher\n{Q}\n```",
            f"```GQL\n{Q};\n```",
            f"```\n{Q}\n```",
            f"```{Q}```",
            f"GQL: {Q}",
            f"Cypher: {Q}",
            f"Query: {Q}",
            f"Here is the query:\n{Q}",
            f"Sure! This finds every person.\n\n```gql\n{Q}\n```\nIt returns their ids.",
        ],
    )
    def test_variants(self, reply: str) -> None:
        assert extract_query(reply) == Q

    def test_multiline_kept(self) -> None:
        query = "MATCH (p:Person)\nWHERE p.age > 26\nRETURN p.node_id AS id"
        assert extract_query(f"```gql\n{query}\n```") == query

    def test_optional_match_start(self) -> None:
        query = "OPTIONAL MATCH (p:Person) RETURN p"
        assert extract_query(f"The query:\n{query}") == query

    def test_write_after_prose_kept(self) -> None:
        assert extract_query("Okay:\nMATCH (n) DETACH DELETE n") == "MATCH (n) DETACH DELETE n"

    def test_empty(self) -> None:
        assert extract_query("") == ""
        assert extract_query("```\n```") == ""

    def test_chain_runs_fenced_reply(self, store: GrafeoGraphStore) -> None:
        chain, _ = _chain(store, f"Here you go:\n```cypher\n{WORKS_AT};\n```", return_direct=True)
        assert len(chain.invoke("Who?")["result"]) == 3

    def test_chain_rejects_empty_reply(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, "   ", "")
        with pytest.raises(ValueError, match="did not return a query: ''"):
            chain.invoke("Who?")
        assert len(llm.prompts) == 2, "an empty reply is sent back once, like a failing query"


class TestLooksLikeQuery:
    @pytest.mark.parametrize(
        "text",
        [
            Q,
            "match (p:Person) return p.node_id as id",
            "OPTIONAL MATCH (p:Person) RETURN p",
            "WITH 1 AS x RETURN x",
            "UNWIND [1, 2] AS x RETURN x",
            "RETURN 1 + 1 AS two",
            "CALL db.labels()",
            "MATCH (n) DETACH DELETE n",
            "INSERT (:Person {node_id: 'Eve'})",
            "CREATE (:Person)",
            "MERGE (:Person {node_id: 'Eve'})",
            "LET x = 1 RETURN x",
        ],
    )
    def test_queries(self, text: str) -> None:
        assert looks_like_query(text)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "I cannot change data in the graph.",
            "Sorry, the schema has no age property.",
            "With this schema, the question cannot be answered.",
            "Match the names case-insensitively.",
            "Return to the question: who is Alix?",
            "Create a new node? I can only read the graph.",
            "The query is not needed.",
        ],
    )
    def test_prose(self, text: str) -> None:
        assert not looks_like_query(text)


class TestNotAQuery:
    """A reply that is not a query goes back to the LLM once, then raises a clear ``ValueError``."""

    PROSE = "I cannot change data in the graph: this chain only reads it."

    def test_prose_then_query(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, self.PROSE, Q, return_direct=True, return_intermediate_steps=True)
        out = chain.invoke("Change Alix's age to 99")
        assert len(out["result"]) == 3
        assert out["intermediate_steps"][0] == {"query": Q}
        assert f"Failed query: {self.PROSE}" in llm.prompts[1]
        assert "not a GQL query" in llm.prompts[1]

    def test_prose_twice_raises(self, store: GrafeoGraphStore, monkeypatch: pytest.MonkeyPatch) -> None:
        ran: list[str] = []
        original = store.query
        monkeypatch.setattr(store, "query", lambda q, *a, **k: ran.append(q) or original(q, *a, **k))
        chain, llm = _chain(store, self.PROSE, "Sorry, I can only read.")
        with pytest.raises(ValueError, match=r"^the model did not return a query: 'Sorry, I can only read\.'$"):
            chain.invoke("Change Alix's age to 99")
        assert len(llm.prompts) == 2
        assert ran == [], "prose is never sent to grafeo"

    def test_message_keeps_first_100_characters(self, store: GrafeoGraphStore) -> None:
        long_prose = "Sorry " * 50
        chain, _ = _chain(store, long_prose, long_prose)
        with pytest.raises(ValueError) as caught:
            chain.invoke("Who?")
        shown = str(caught.value).removeprefix("the model did not return a query: ")
        assert shown == repr(" ".join(long_prose.split())[:100])

    def test_no_retries(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, self.PROSE, max_retries=0)
        with pytest.raises(ValueError, match="did not return a query"):
            chain.invoke("Who?")
        assert len(llm.prompts) == 1

    def test_fenced_prose(self, store: GrafeoGraphStore) -> None:
        chain, _ = _chain(store, "```\nI can't do that.\n```", "```cypher\nNo query is possible.\n```")
        with pytest.raises(ValueError, match=r"did not return a query: 'No query is possible\.'"):
            chain.invoke("Who?")

    def test_sentence_case_keyword_is_prose(self, store: GrafeoGraphStore) -> None:
        chain, _ = _chain(store, "With this schema the question cannot be answered.", Q, return_direct=True)
        assert len(chain.invoke("Who?")["result"]) == 3

    def test_lowercase_query_runs(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, "match (p:Person) return p.node_id as id", return_direct=True)
        assert len(chain.invoke("Who?")["result"]) == 3
        assert len(llm.prompts) == 1

    def test_fenced_query_runs(self, store: GrafeoGraphStore) -> None:
        for fence in ("```cypher", "```gql", "```"):
            chain, llm = _chain(store, f"{fence}\n{Q}\n```", return_direct=True)
            assert len(chain.invoke("Who?")["result"]) == 3
            assert len(llm.prompts) == 1

    def test_broken_query_still_raises_grafeo_error(self, store: GrafeoGraphStore) -> None:
        chain, _ = _chain(store, "MATCH (p:Person RETURN p", "MATCH (p:Person RETURN p")
        with pytest.raises(grafeo.GrafeoError):
            chain.invoke("Who?")

    def test_prose_then_write_is_refused(self, store: GrafeoGraphStore) -> None:
        before = _snapshot(store)
        chain, _ = _chain(store, self.PROSE, "MATCH (p:Person {node_id: 'Alice'}) SET p.age = 99")
        with pytest.raises(ValueError, match="may not write"):
            chain.invoke("Change Alix's age to 99")
        assert _snapshot(store) == before

    def test_callback_text(self, store: GrafeoGraphStore) -> None:
        recorder = Recorder()
        chain, _ = _chain(store, self.PROSE, Q, return_direct=True)
        chain.invoke("Who?", config={"callbacks": [recorder]})
        assert "text:Reply was not a query, asking again\n" in recorder.events

    def test_ainvoke(self, store: GrafeoGraphStore) -> None:
        chain, _ = _chain(store, self.PROSE, "Still no query here.")
        with pytest.raises(ValueError, match=r"did not return a query: 'Still no query here\.'"):
            asyncio.run(chain.ainvoke("Who?"))

    def test_ainvoke_prose_then_query(self, store: GrafeoGraphStore) -> None:
        chain, _ = _chain(store, self.PROSE, Q, return_direct=True)
        assert len(asyncio.run(chain.ainvoke("Who?"))["result"]) == 3


class TestIdProperty:
    """The prompts name the property that identifies entities: node_id in a graph the store wrote, else name."""

    @pytest.fixture
    def named(self) -> GrafeoGraphStore:
        # a graph the store did not write, as loaded from a dataset: entities have name and key, no node_id
        store = GrafeoGraphStore()
        store.query(
            "INSERT (:Person {name: 'Bertram Gilfoyle', key: 'gilfoyle'})"
            "-[:BUILT]->(:AI {name: 'Son of Anton', key: 'son-of-anton'})"
        )
        return store

    def test_store_graph_uses_node_id(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, Q, return_direct=True)
        chain.invoke("Who?")
        assert "identified by their node_id property" in llm.prompts[0]
        assert "toLower(c.node_id) CONTAINS 'acme'" in llm.prompts[0]

    def test_dataset_graph_uses_name(self, named: GrafeoGraphStore) -> None:
        query = "MATCH (p)-[:BUILT]-(a) WHERE toLower(a.name) CONTAINS 'anton' RETURN p.name AS builder"
        chain, llm = _chain(named, query, return_direct=True)
        assert chain.invoke("Who built Son of Anton?")["result"] == [{"builder": "Bertram Gilfoyle"}]
        assert "identified by their name property" in llm.prompts[0]
        assert "node_id" not in llm.prompts[0]

    def test_fix_prompt_gets_it_too(self, named: GrafeoGraphStore) -> None:
        prompt = PromptTemplate.from_template("Fix {query} ({error}); entities are named by {id_property}.")
        chain, llm = _chain(
            named, "MATCH (", "MATCH (p:Person) RETURN p.name AS n", fix_prompt=prompt, return_direct=True
        )
        chain.invoke("Who?")
        assert llm.prompts[1].endswith("entities are named by name.")

    def test_explicit(self, named: GrafeoGraphStore) -> None:
        chain, llm = _chain(named, "MATCH (a:AI) RETURN a.key AS k", id_property="key", return_direct=True)
        assert chain.invoke("Which AIs?")["result"] == [{"k": "son-of-anton"}]
        assert "identified by their key property" in llm.prompts[0]

    def test_custom_prompt(self, named: GrafeoGraphStore) -> None:
        prompt = PromptTemplate.from_template("{schema}\nMatch on {id_property}.\n{question}")
        chain, llm = _chain(named, "MATCH (a:AI) RETURN a.name AS n", query_prompt=prompt, return_direct=True)
        chain.invoke("Which AIs?")
        assert "Match on name." in llm.prompts[0]

    def test_empty_graph(self) -> None:
        chain, llm = _chain(
            GrafeoGraphStore(), "MATCH (n) RETURN n.node_id AS id", retry_on_empty=False, return_direct=True
        )
        assert chain.invoke("Anything?")["result"] == []
        assert "identified by their node_id property" in llm.prompts[0]

    def test_ainvoke(self, named: GrafeoGraphStore) -> None:
        chain, llm = _chain(named, "MATCH (a:AI) RETURN a.name AS n", return_direct=True)
        assert asyncio.run(chain.ainvoke("Which AIs?"))["result"] == [{"n": "Son of Anton"}]
        assert "identified by their name property" in llm.prompts[0]


GUESSED = "MATCH (p:Person {node_id: 'alice'}) RETURN p.node_id AS id"  # a guessed id: the node is 'Alice'
LOOSER = "MATCH (p) WHERE toLower(p.node_id) CONTAINS 'alice' RETURN p.node_id AS id"


class TestEmptyResult:
    """A query that matches nothing gets one looser try; if that finds nothing or fails, the empty result stands."""

    def test_looser_query_finds_rows(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, GUESSED, LOOSER, return_direct=True, return_intermediate_steps=True)
        out = chain.invoke("Who is Alice?")
        assert out["result"] == [{"id": "Alice"}]
        assert out["intermediate_steps"][0] == {"query": LOOSER}
        fix_prompt = llm.prompts[1]
        assert f"Failed query: {GUESSED}" in fix_prompt
        assert "matched nothing" in fix_prompt
        assert "CONTAINS" in fix_prompt

    def test_still_empty(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, GUESSED, GUESSED, return_direct=True)
        assert chain.invoke("Who is Alice?")["result"] == []
        assert len(llm.prompts) == 2, "one looser try, then the empty result stands"

    def test_looser_query_fails(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, GUESSED, "MATCH (", return_direct=True, return_intermediate_steps=True)
        out = chain.invoke("Who is Alice?")
        assert out["result"] == []
        assert out["intermediate_steps"][0] == {"query": GUESSED}, "the query that ran stands"
        assert len(llm.prompts) == 2

    def test_looser_query_not_a_query(self, store: GrafeoGraphStore) -> None:
        chain, _ = _chain(store, GUESSED, "The graph has no such person.", return_direct=True)
        assert chain.invoke("Who is Alice?")["result"] == []

    def test_looser_query_writes(self, store: GrafeoGraphStore) -> None:
        before = _snapshot(store)
        chain, _ = _chain(store, GUESSED, "MATCH (n) DETACH DELETE n")
        with pytest.raises(ValueError, match="may not write"):
            chain.invoke("Who is Alice?")
        assert _snapshot(store) == before

    def test_answer_after_looser_query(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, GUESSED, LOOSER, "Alice is in the graph.")
        assert chain.invoke("Who is Alice?")["result"] == "Alice is in the graph."
        assert f"Query: {LOOSER}" in llm.prompts[2]
        assert "{'id': 'Alice'}" in llm.prompts[2]

    def test_disabled(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, GUESSED, retry_on_empty=False, return_direct=True)
        assert chain.invoke("Who is Alice?")["result"] == []
        assert len(llm.prompts) == 1

    def test_no_retries(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, GUESSED, max_retries=0, return_direct=True)
        assert chain.invoke("Who is Alice?")["result"] == []
        assert len(llm.prompts) == 1

    def test_not_with_writes_allowed(self, store: GrafeoGraphStore) -> None:
        # an INSERT returns no rows: retrying it would insert twice
        chain, llm = _chain(store, "INSERT (:Robot {node_id: 'r2'})", "Added.", allow_writes=True)
        assert chain.invoke("Add a robot")["result"] == "Added."
        assert len(llm.prompts) == 2, "the query and the answer, no retry"
        assert store.query("MATCH (r:Robot) RETURN count(r) AS n") == [{"n": 1}]

    def test_one_looser_try_then_errors_use_the_budget(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, GUESSED, "MATCH (", LOOSER, max_retries=2, return_direct=True)
        assert chain.invoke("Who is Alice?")["result"] == [{"id": "Alice"}]
        assert len(llm.prompts) == 3
        assert "syntax error" in llm.prompts[2]

    def test_only_one_looser_try(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, GUESSED, GUESSED, max_retries=3, return_direct=True)
        assert chain.invoke("Who is Alice?")["result"] == []
        assert len(llm.prompts) == 2, "an empty result is retried once, whatever max_retries is"

    def test_callback_text(self, store: GrafeoGraphStore) -> None:
        recorder = Recorder()
        chain, _ = _chain(store, GUESSED, LOOSER, return_direct=True)
        chain.invoke("Who is Alice?", config={"callbacks": [recorder]})
        assert "text:Query matched nothing, asking for a looser one\n" in recorder.events

    def test_ainvoke(self, store: GrafeoGraphStore) -> None:
        chain, _ = _chain(store, GUESSED, LOOSER, return_direct=True)
        assert asyncio.run(chain.ainvoke("Who is Alice?"))["result"] == [{"id": "Alice"}]

    def test_ainvoke_still_empty(self, store: GrafeoGraphStore) -> None:
        chain, _ = _chain(store, GUESSED, "MATCH (", return_direct=True)
        assert asyncio.run(chain.ainvoke("Who is Alice?"))["result"] == []


# ── Writes ───────────────────────────────────────────────────────────────────


WRITES = [
    "MATCH (n) DETACH DELETE n",
    "MATCH (p:Person {node_id: 'Bob'}) SET p.age = 99",
    "CREATE (:Person {node_id: 'Mallory'})",
    "INSERT (:Person {node_id: 'Mallory'})",
    "MERGE (:Person {node_id: 'Mallory'})",
    "MATCH (p:Person {node_id: 'Alice'})-[r:KNOWS]->() DELETE r",
    "MATCH (p:Person {node_id: 'Alice'}) REMOVE p.age",
]


class TestWrites:
    @pytest.mark.parametrize("query", WRITES)
    def test_write_refused(self, store: GrafeoGraphStore, query: str) -> None:
        before = _snapshot(store)
        chain, llm = _chain(store, query)
        with pytest.raises(ValueError, match="read-only query may not write"):
            chain.invoke("Clean up the graph")
        assert _snapshot(store) == before
        assert len(llm.prompts) == 1, "a refused write is not sent back for a fix, and no answer is written"

    def test_write_in_prose_refused(self, store: GrafeoGraphStore) -> None:
        before = _snapshot(store)
        chain, _ = _chain(store, "Sure, this removes Bob:\n```gql\nMATCH (p {node_id: 'Bob'}) DETACH DELETE p\n```")
        with pytest.raises(ValueError, match="may not write"):
            chain.invoke("Remove Bob")
        assert _snapshot(store) == before

    def test_allow_writes(self, store: GrafeoGraphStore) -> None:
        chain, _ = _chain(store, "INSERT (:Person {node_id: 'Dave', age: 40})", "Added Dave.", allow_writes=True)
        assert chain.invoke("Add Dave, 40")["result"] == "Added Dave."
        assert store.query("MATCH (p:Person {node_id: 'Dave'}) RETURN p.age AS age") == [{"age": 40}]

    def test_allow_writes_refreshes_schema(self, store: GrafeoGraphStore) -> None:
        assert "Robot" not in store.get_schema
        chain, _ = _chain(store, "INSERT (:Robot {node_id: 'r2', model: 'astromech'})", "ok", allow_writes=True)
        chain.invoke("Add a robot")
        assert "Robot {model: STRING, node_id: STRING}" in store.get_schema

    def test_allow_writes_still_reads(self, store: GrafeoGraphStore) -> None:
        chain, _ = _chain(store, WORKS_AT, allow_writes=True, return_direct=True)
        assert len(chain.invoke("Who?")["result"]) == 3


# ── Failing queries ──────────────────────────────────────────────────────────


class TestRetries:
    def test_fixes_syntax_error(self, store: GrafeoGraphStore) -> None:
        broken = "MATCH (p:Person RETURN p.node_id AS id"
        chain, llm = _chain(
            store,
            broken,
            "MATCH (p:Person) RETURN p.node_id AS id ORDER BY id",
            "Alice, Bob and Carol.",
            return_intermediate_steps=True,
        )
        out = chain.invoke("Who are the people?")
        assert out["result"] == "Alice, Bob and Carol."
        assert out["intermediate_steps"][0] == {"query": "MATCH (p:Person) RETURN p.node_id AS id ORDER BY id"}
        fix_prompt = llm.prompts[1]
        assert f"Failed query: {broken}" in fix_prompt
        assert "syntax error" in fix_prompt
        assert "Who are the people?" in fix_prompt
        assert "(:Person)-[:WORKS_AT]->(:Company)" in fix_prompt, "the fix prompt has the schema"

    def test_fixes_unknown_function(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(
            store,
            "MATCH (p:Person) RETURN no_such_function(p.node_id) AS id",
            "MATCH (p:Person) RETURN p.node_id AS id",
            return_direct=True,
        )
        assert len(chain.invoke("Who?")["result"]) == 3
        assert len(llm.prompts) == 2

    def test_gives_up_after_max_retries(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, "MATCH (", "MATCH ((", "MATCH (((", max_retries=2)
        with pytest.raises(grafeo.GrafeoError):
            chain.invoke("Who?")
        assert len(llm.prompts) == 3, "the first query and two fixes"
        assert "Failed query: MATCH ((" in llm.prompts[2], "each fix sees the query that failed last"

    def test_no_retries(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, "MATCH (", max_retries=0)
        with pytest.raises(grafeo.GrafeoError):
            chain.invoke("Who?")
        assert len(llm.prompts) == 1

    def test_fix_that_writes_is_refused(self, store: GrafeoGraphStore) -> None:
        before = _snapshot(store)
        chain, llm = _chain(store, "MATCH (", "MATCH (n) DETACH DELETE n")
        with pytest.raises(ValueError, match="may not write"):
            chain.invoke("Who?")
        assert _snapshot(store) == before
        assert len(llm.prompts) == 2


# ── Configuration ────────────────────────────────────────────────────────────


class TestConfiguration:
    def test_custom_query_prompt(self, store: GrafeoGraphStore) -> None:
        prompt = PromptTemplate.from_template("Graph:\n{schema}\nAsk: {question}\nAnswer in GQL.")
        chain, llm = _chain(store, WORKS_AT, "answer", query_prompt=prompt)
        chain.invoke("Who?")
        assert llm.prompts[0].startswith("Graph:\nNode labels and properties:")
        assert llm.prompts[0].endswith("Ask: Who?\nAnswer in GQL.")

    def test_custom_qa_prompt(self, store: GrafeoGraphStore) -> None:
        prompt = PromptTemplate.from_template("Rows: {context}\nReply in French to: {question}")
        chain, llm = _chain(store, WORKS_AT, "Alice travaille chez Microsoft.", qa_prompt=prompt)
        chain.invoke("Where does Alice work?")
        assert llm.prompts[1].startswith("Rows: [{'person': 'Alice'")

    def test_custom_qa_prompt_with_query(self, store: GrafeoGraphStore) -> None:
        prompt = PromptTemplate.from_template("{query} gave {context}; answer: {question}")
        chain, llm = _chain(store, WORKS_AT, "answer", qa_prompt=prompt)
        chain.invoke("Who?")
        assert llm.prompts[1].startswith(f"{WORKS_AT} gave [")

    def test_custom_fix_prompt(self, store: GrafeoGraphStore) -> None:
        prompt = PromptTemplate.from_template("{query} failed: {error}. Try again.")
        chain, llm = _chain(store, "MATCH (", Q, fix_prompt=prompt, return_direct=True)
        chain.invoke("Who?")
        assert llm.prompts[1].startswith("MATCH ( failed: ")

    def test_prompt_with_unknown_variable(self, store: GrafeoGraphStore) -> None:
        prompt = PromptTemplate.from_template("{schema} {question} {language}")
        with pytest.raises(ValidationError, match=r"query_prompt uses \['language'\]"):
            _chain(store, query_prompt=prompt)

    @pytest.mark.parametrize("field", ["qa_prompt", "fix_prompt"])
    def test_other_prompts_checked(self, store: GrafeoGraphStore, field: str) -> None:
        with pytest.raises(ValidationError, match=f"{field} uses"):
            _chain(store, **{field: PromptTemplate.from_template("{nope}")})

    @pytest.mark.parametrize(("field", "value"), [("top_k", 0), ("max_retries", -1)])
    def test_invalid_numbers(self, store: GrafeoGraphStore, field: str, value: int) -> None:
        with pytest.raises(ValidationError):
            _chain(store, **{field: value})

    def test_llm_wrapped_in_runnable(self, store: GrafeoGraphStore) -> None:
        llm = ScriptedLLM(responses=[WORKS_AT, "answer"])
        chain = GrafeoQAChain.from_llm(llm.with_config(tags=["qa"]), graph=store)
        assert chain.invoke("Who?")["result"] == "answer"

    def test_schema_follows_the_store(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, Q, Q, return_direct=True)
        chain.invoke("Who?")
        assert "Planet" not in llm.prompts[0]
        store.add_graph_documents(
            [
                GraphDocument(
                    nodes=[Node(id="Mars", type="Planet")], relationships=[], source=Document(page_content="Mars")
                )
            ]
        )
        chain.invoke("Which planets?")
        assert "Planet {node_id: STRING}" in llm.prompts[1]

    def test_persistent_store(self, tmp_path: Path) -> None:
        path = str(tmp_path / "kg.grafeo")
        with GrafeoGraphStore(db_path=path) as store:
            _people(store)
        with GrafeoGraphStore(db_path=path) as store:
            chain, _ = _chain(store, WORKS_AT, return_direct=True)
            assert len(chain.invoke("Who?")["result"]) == 3


# ── Runnable behavior ────────────────────────────────────────────────────────


class Recorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_chain_start(self, serialized: dict[str, Any] | None, inputs: Any, **kwargs: Any) -> None:
        self.events.append(f"chain_start:{kwargs.get('name')}")

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self.events.append("llm_start")

    def on_text(self, text: str, **kwargs: Any) -> None:
        self.events.append(f"text:{text.split(':')[0]}")

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        self.events.append(f"chain_error:{type(error).__name__}")


class TestRunnable:
    def test_batch(self, store: GrafeoGraphStore) -> None:
        query_llm = ScriptedLLM(responses=[WORKS_AT, WORKS_AT])
        chain = GrafeoQAChain.from_llm(graph=store, query_llm=query_llm, qa_llm=query_llm, return_direct=True)
        outs = chain.batch(["Who?", "Where?"], config={"max_concurrency": 1})
        assert [out["query"] for out in outs] == ["Who?", "Where?"]
        assert all(len(out["result"]) == 3 for out in outs)

    def test_ainvoke(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, WORKS_AT, "answer", return_intermediate_steps=True)
        out = asyncio.run(chain.ainvoke({"query": "Who?"}))
        assert out["result"] == "answer"
        assert out["intermediate_steps"][0] == {"query": WORKS_AT}
        assert len(llm.prompts) == 2

    def test_ainvoke_retries(self, store: GrafeoGraphStore) -> None:
        chain, llm = _chain(store, "MATCH (", Q, return_direct=True)
        assert len(asyncio.run(chain.ainvoke("Who?"))["result"]) == 3
        assert "Failed query: MATCH (" in llm.prompts[1]
        assert "Error: GRAFEO-Q001" in llm.prompts[1], "the fix prompt has grafeo's error"

    def test_ainvoke_gives_up(self, store: GrafeoGraphStore) -> None:
        chain, _ = _chain(store, "MATCH (", "MATCH ((")
        with pytest.raises(grafeo.GrafeoError):
            asyncio.run(chain.ainvoke("Who?"))

    def test_ainvoke_refuses_writes(self, store: GrafeoGraphStore) -> None:
        before = _snapshot(store)
        chain, _ = _chain(store, "MATCH (n) DETACH DELETE n")
        with pytest.raises(ValueError, match="may not write"):
            asyncio.run(chain.ainvoke("Who?"))
        assert _snapshot(store) == before

    def test_abatch(self, store: GrafeoGraphStore) -> None:
        chain, _ = _chain(store, WORKS_AT, WORKS_AT, return_direct=True)
        outs = asyncio.run(chain.abatch(["Who?", "Where?"], config={"max_concurrency": 1}))
        assert [len(out["result"]) for out in outs] == [3, 3]

    def test_compose(self, store: GrafeoGraphStore) -> None:
        chain, _ = _chain(store, WORKS_AT, "Alice works at Microsoft.")
        pipeline = chain | RunnableLambda(lambda out: out["result"].upper())
        assert pipeline.invoke("Where does Alice work?") == "ALICE WORKS AT MICROSOFT."

    def test_stream(self, store: GrafeoGraphStore) -> None:
        chain, _ = _chain(store, WORKS_AT, "answer")
        assert list(chain.stream("Who?")) == [{"query": "Who?", "result": "answer"}]

    def test_callbacks_trace_llm_calls(self, store: GrafeoGraphStore) -> None:
        recorder = Recorder()
        chain, _ = _chain(store, "MATCH (", Q, "answer")
        chain.invoke("Who?", config={"callbacks": [recorder]})
        assert recorder.events[0] == "chain_start:GrafeoQAChain"
        assert recorder.events.count("llm_start") == 3, "the query, the fix and the answer"
        assert "text:Query failed, asking for a fix" in recorder.events

    def test_callbacks_see_errors(self, store: GrafeoGraphStore) -> None:
        recorder = Recorder()
        chain, _ = _chain(store, "MATCH (n) DETACH DELETE n")
        with pytest.raises(ValueError):
            chain.invoke("Who?", config={"callbacks": [recorder]})
        assert "chain_error:ValueError" in recorder.events

    def test_run_name(self, store: GrafeoGraphStore) -> None:
        recorder = Recorder()
        chain, _ = _chain(store, Q, return_direct=True)
        chain.invoke("Who?", config={"callbacks": [recorder], "run_name": "people-qa"})
        assert recorder.events[0] == "chain_start:people-qa"


# ── Values in results ────────────────────────────────────────────────────────


def test_decimal_returned_direct(store: GrafeoGraphStore) -> None:
    store.query("INSERT (:Product {node_id: 'pen', price: 1.50})")
    chain, _ = _chain(store, "MATCH (p:Product) RETURN p.price AS price", return_direct=True)
    assert chain.invoke("Price?")["result"] == [{"price": Decimal("1.50")}]

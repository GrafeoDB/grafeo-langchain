"""Question answering over a ``GrafeoGraphStore``: an LLM writes a GQL query, Grafeo runs it, an LLM answers.

Built on ``langchain-core`` only.  Inputs and outputs follow ``GraphCypherQAChain``
(``{"query": question}`` in, ``{"query": question, "result": answer}`` out).
"""

from __future__ import annotations

import re
from typing import Any, NoReturn

import grafeo
from langchain_core.callbacks import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import patch_config, run_in_executor
from pydantic import ConfigDict, Field, model_validator

from grafeo_langchain.graph_store import GrafeoGraphStore

QUERY_TEMPLATE = """Write one GQL query that answers the question from the graph described below.

GQL reads like Cypher: MATCH patterns, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT, and functions such as \
count(), collect(), toLower() and type().
Rules:
- Use only the relationship types and properties of the schema. Leave the relationship direction out unless the \
schema shows it: (p)-[:WORKS_AT]-(c).
- The same kind of entity can carry different labels, and the same relation different types, especially in a graph \
an LLM extracted (Company and Organization; FOUNDED and FOUNDED_BY). Leave node labels out, and list every \
relationship type of the schema that could fit: (p)-[:LEADS|CEO_OF|HAS_CEO]-(c).
- Entities are identified by their {id_property} property, which holds the full name ('Acme Robotics', 'Hexa warehouse \
robot'). Match a name from the question by one distinctive word of it, case-insensitively: \
WHERE toLower(c.{id_property}) CONTAINS 'acme', or CONTAINS 'hexa' for "the Hexa robot".
- When the question asks about other entities (who worked with someone), leave the one it starts from out: \
WHERE other <> p.
- Read facts from the properties in the schema, matching the question's words to the closest property \
(the cost of a product is its price). A fact about one entity needs no relationship: MATCH (p) WHERE \
toLower(p.{id_property}) CONTAINS 'hexa' RETURN p.price AS price. Some facts are nodes instead: follow a relationship \
such as COSTS to them. \
SourceDocument nodes hold the original text in their text property: use them only for questions about documents.
- Return named values (RETURN p.{id_property} AS person), not whole nodes.
- Only read the graph: no INSERT, CREATE, MERGE, SET, REMOVE or DELETE.
- Reply with the query only, without explanation.

Schema:
{schema}

Question: {question}
Query:"""

FIX_TEMPLATE = """This GQL query, written to answer the question, failed with the error below. \
Write a corrected query. Reply with the query only.

Schema:
{schema}

Question: {question}
Failed query: {query}
Error: {error}
Query:"""

QA_TEMPLATE = """Answer the question from the query results below. The query was written for this question and ran \
on a knowledge graph, so its results are the answer, even when they do not repeat the question's conditions: for \
"Who is older than 40?", the names returned are the people older than 40. Do not contradict or extend the results. \
Only if they are empty, say that the graph does not hold the answer.

Question: {question}
Query: {query}
Query results: {context}
Answer:"""

QUERY_PROMPT = PromptTemplate.from_template(QUERY_TEMPLATE)
FIX_PROMPT = PromptTemplate.from_template(FIX_TEMPLATE)
QA_PROMPT = PromptTemplate.from_template(QA_TEMPLATE)

# The variables each prompt is formatted with; a custom prompt may use any of them.
_PROMPT_VARIABLES = {
    "query_prompt": {"schema", "question", "id_property"},
    "fix_prompt": {"schema", "question", "query", "error", "id_property"},
    "qa_prompt": {"question", "query", "context"},
}

_FENCED = re.compile(r"```[ \t]*[A-Za-z]*[ \t]*\n(.*?)```", re.DOTALL)
_FENCED_INLINE = re.compile(r"```(.*?)```", re.DOTALL)
_LABEL = re.compile(r"^(?:gql|cypher|query)\s*:\s*", re.IGNORECASE)
# Where a query starts when the LLM put prose before it.
_QUERY_START = re.compile(
    r"^\s*(?:OPTIONAL\s+MATCH|MATCH|WITH|UNWIND|CALL|RETURN|INSERT|CREATE|MERGE|DETACH|DELETE)\b",
    re.IGNORECASE | re.MULTILINE,
)


# The first word of a query: a GQL clause, written as code (MATCH or match, not prose's "Match").
_QUERY_KEYWORD = re.compile(
    r"(?:OPTIONAL\s+MATCH|MATCH|WITH|UNWIND|CALL|RETURN|INSERT|CREATE|MERGE|DETACH|DELETE|LET|FILTER|FOR|USE)\b",
    re.IGNORECASE,
)
_NOT_A_QUERY = "the reply was not a GQL query: reply with the query only"
_NO_ROWS = (
    "the query ran but matched nothing. If the graph can answer the question, loosen the query: leave node labels "
    "out, match names by a distinctive part with CONTAINS, list alternative relationship types, leave directions "
    "out, or look for the value in a property instead of a relationship. If the graph cannot answer it, reply with "
    "the same query"
)


def looks_like_query(text: str) -> bool:
    """Tell a query from prose: it starts with a GQL clause in upper or lower case (``MATCH``, ``match``).

    Prose that starts with such a word in sentence case ("With this schema ...") is not a query.
    """
    match = _QUERY_KEYWORD.match(text)
    if not match:
        return False
    first_word = text.split()[0]
    return first_word.isupper() or first_word.islower()


class _NotAQueryError(Exception):
    """An LLM reply that holds no query; it is sent back for a fix like a failing query."""


def extract_query(text: str) -> str:
    """Return the query in an LLM reply: the first code block if there is one, without a label or prose before it."""
    match = _FENCED.search(text) or _FENCED_INLINE.search(text)
    if match:
        text = match.group(1)
    text = _LABEL.sub("", text.strip())
    start = _QUERY_START.search(text)
    if start:
        text = text[start.start() :]
    return text.strip().rstrip(";").strip()


class GrafeoQAChain(RunnableSerializable[dict[str, Any] | str, dict[str, Any]]):
    """Answer questions about a ``GrafeoGraphStore`` with an LLM.

    The query LLM turns the question and the graph schema into a GQL query, the
    store runs it, and the QA LLM answers from the rows.  A query that fails is
    sent back to the query LLM with the error, up to ``max_retries`` times.

    Queries run read-only by default: a generated query that would write is
    rolled back and raises ``ValueError``, so the graph stays unchanged.

    Usage::

        chain = GrafeoQAChain.from_llm(llm, graph=store)
        chain.invoke({"query": "Where does Alice work?"})["result"]

    Attributes:
        graph: The store to query.
        query_llm: Writes the GQL query; a chat model, an LLM or any runnable that returns text.
        qa_llm: Writes the answer from the query results.
        query_prompt: Formatted with ``schema``, ``question`` and ``id_property``.
        fix_prompt: Formatted with ``schema``, ``question``, ``id_property``, the failed ``query`` and its ``error``.
        qa_prompt: Formatted with ``question``, the ``query`` that ran and ``context`` (the result rows).
        top_k: How many result rows the QA LLM sees (and ``return_direct`` returns).
        max_retries: How many times a failing query is sent back for a fix; 0 raises the first error.
        retry_on_empty: Spend one of those retries on a looser query when a query matches nothing (a label, an id
            or a relationship type the LLM guessed); if that fails too, the empty result stands. Only read-only
            chains retry: with ``allow_writes``, an empty result can be a write that worked.
        allow_writes: Run generated queries as they are, writes included.
        return_direct: Return the result rows as ``result``, without calling the QA LLM.
        return_intermediate_steps: Add ``intermediate_steps``: ``[{"query": gql}, {"context": rows}]``.
        id_property: The property that names an entity, which the prompts tell the LLM to match. ``None`` picks
            ``node_id`` when the graph has it (this store writes it), else ``name``.
        input_key: The input key holding the question.
        output_key: The output key holding the answer.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph: GrafeoGraphStore = Field(exclude=True)
    query_llm: Runnable[Any, Any]
    qa_llm: Runnable[Any, Any]
    query_prompt: BasePromptTemplate = QUERY_PROMPT
    fix_prompt: BasePromptTemplate = FIX_PROMPT
    qa_prompt: BasePromptTemplate = QA_PROMPT
    top_k: int = Field(default=10, ge=1)
    max_retries: int = Field(default=1, ge=0)
    retry_on_empty: bool = True
    id_property: str | None = None
    allow_writes: bool = False
    return_direct: bool = False
    return_intermediate_steps: bool = False
    input_key: str = "query"
    output_key: str = "result"

    @model_validator(mode="after")
    def _check_prompts(self) -> GrafeoQAChain:
        for name, provided in _PROMPT_VARIABLES.items():
            unknown = set(getattr(self, name).input_variables) - provided
            if unknown:
                msg = f"{name} uses {sorted(unknown)}, but is formatted with {sorted(provided)} only"
                raise ValueError(msg)
        return self

    @classmethod
    def from_llm(
        cls,
        llm: Runnable[Any, Any] | None = None,
        *,
        graph: GrafeoGraphStore,
        query_llm: Runnable[Any, Any] | None = None,
        qa_llm: Runnable[Any, Any] | None = None,
        **kwargs: Any,
    ) -> GrafeoQAChain:
        """Build the chain with one LLM for both steps, or a separate ``query_llm`` and ``qa_llm``."""
        query_llm = query_llm or llm
        qa_llm = qa_llm or llm
        if query_llm is None or qa_llm is None:
            msg = "pass llm, or both query_llm and qa_llm"
            raise ValueError(msg)
        return cls(graph=graph, query_llm=query_llm, qa_llm=qa_llm, **kwargs)

    # ── Runnable interface ──────────────────────────────────────────────────

    def invoke(
        self, input: dict[str, Any] | str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return self._call_with_config(self._run, input, config, run_type="chain")

    async def ainvoke(
        self, input: dict[str, Any] | str, config: RunnableConfig | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return await self._acall_with_config(self._arun, input, config, run_type="chain")

    # ── Steps ───────────────────────────────────────────────────────────────

    def _run(
        self, input: dict[str, Any] | str, run_manager: CallbackManagerForChainRun, config: RunnableConfig
    ) -> dict[str, Any]:
        question = self._question(input)
        child = patch_config(config, callbacks=run_manager.get_child())
        schema = self.graph.get_schema
        id_property = self._id_property()
        query = extract_query(
            self._llm_text(
                self.query_llm, self.query_prompt, child, schema=schema, question=question, id_property=id_property
            )
        )
        rows: list[dict[str, Any]] = []
        empty: tuple[str, list[dict[str, Any]]] | None = None  # a query that ran and matched nothing
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            try:
                self._check_query(query)
                rows = self.graph.query(query, read_only=not self.allow_writes)
            except (grafeo.GrafeoError, _NotAQueryError) as error:
                if last:
                    if empty is not None:  # the looser query failed: keep the empty result
                        query, rows = empty
                        break
                    self._raise_final(error, query)
                run_manager.on_text(self._retry_text(error))
                problem = str(error)
            else:
                if rows or last or not self.retry_on_empty or self.allow_writes or empty is not None:
                    break
                empty = (query, rows)
                run_manager.on_text("Query matched nothing, asking for a looser one\n")
                problem = _NO_ROWS
            reply = self._llm_text(
                self.query_llm,
                self.fix_prompt,
                child,
                schema=schema,
                question=question,
                id_property=id_property,
                query=query,
                error=problem,
            )
            query = extract_query(reply)
        context = rows[: self.top_k]
        if self.return_direct:
            return self._output(question, query, context, context)
        answer = self._llm_text(self.qa_llm, self.qa_prompt, child, question=question, query=query, context=context)
        return self._output(question, query, context, answer)

    async def _arun(
        self, input: dict[str, Any] | str, run_manager: AsyncCallbackManagerForChainRun, config: RunnableConfig
    ) -> dict[str, Any]:
        question = self._question(input)
        child = patch_config(config, callbacks=run_manager.get_child())
        schema = await run_in_executor(config, lambda: self.graph.get_schema)
        id_property = await run_in_executor(config, self._id_property)
        reply = await self._allm_text(
            self.query_llm, self.query_prompt, child, schema=schema, question=question, id_property=id_property
        )
        query = extract_query(reply)
        rows: list[dict[str, Any]] = []
        empty: tuple[str, list[dict[str, Any]]] | None = None  # a query that ran and matched nothing
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            try:
                self._check_query(query)
                rows = await run_in_executor(config, self.graph.query, query, read_only=not self.allow_writes)
            except (grafeo.GrafeoError, _NotAQueryError) as error:
                if last:
                    if empty is not None:  # the looser query failed: keep the empty result
                        query, rows = empty
                        break
                    self._raise_final(error, query)
                await run_manager.on_text(self._retry_text(error))
                problem = str(error)
            else:
                if rows or last or not self.retry_on_empty or self.allow_writes or empty is not None:
                    break
                empty = (query, rows)
                await run_manager.on_text("Query matched nothing, asking for a looser one\n")
                problem = _NO_ROWS
            reply = await self._allm_text(
                self.query_llm,
                self.fix_prompt,
                child,
                schema=schema,
                question=question,
                id_property=id_property,
                query=query,
                error=problem,
            )
            query = extract_query(reply)
        context = rows[: self.top_k]
        if self.return_direct:
            return self._output(question, query, context, context)
        answer = await self._allm_text(
            self.qa_llm, self.qa_prompt, child, question=question, query=query, context=context
        )
        return self._output(question, query, context, answer)

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _question(self, input: dict[str, Any] | str) -> str:
        if isinstance(input, str):
            return input
        if self.input_key not in input:
            msg = f"the input has no {self.input_key!r} key with the question (got {sorted(input)})"
            raise ValueError(msg)
        return str(input[self.input_key])

    def _id_property(self) -> str:
        if self.id_property:
            return self.id_property
        keys = self.graph.get_structured_schema.get("property_keys", [])
        return "name" if "node_id" not in keys and "name" in keys else "node_id"

    @staticmethod
    def _check_query(query: str) -> None:
        if not looks_like_query(query):
            raise _NotAQueryError(_NOT_A_QUERY)

    @staticmethod
    def _retry_text(error: Exception) -> str:
        if isinstance(error, _NotAQueryError):
            return "Reply was not a query, asking again\n"
        return f"Query failed, asking for a fix: {error}\n"

    @staticmethod
    def _raise_final(error: Exception, query: str) -> NoReturn:
        """Raise the error of the last attempt: a ``ValueError`` for a reply without a query."""
        if isinstance(error, _NotAQueryError):
            msg = f"the model did not return a query: {' '.join(query.split())[:100]!r}"
            raise ValueError(msg) from None
        raise error

    @staticmethod
    def _llm_text(llm: Runnable[Any, Any], prompt: BasePromptTemplate, config: RunnableConfig, **values: Any) -> str:
        return (prompt | llm | StrOutputParser()).invoke(values, config)

    @staticmethod
    async def _allm_text(
        llm: Runnable[Any, Any], prompt: BasePromptTemplate, config: RunnableConfig, **values: Any
    ) -> str:
        return await (prompt | llm | StrOutputParser()).ainvoke(values, config)

    def _output(self, question: str, query: str, context: list[dict[str, Any]], result: Any) -> dict[str, Any]:
        output: dict[str, Any] = {self.input_key: question, self.output_key: result}
        if self.return_intermediate_steps:
            output["intermediate_steps"] = [{"query": query}, {"context": context}]
        return output

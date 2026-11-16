<picture>
  <source media="(max-width: 556px) and (prefers-color-scheme: dark), (min-width: 768px) and (max-width: 844px) and (prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GrafeoDB/grafeo-langchain/main/.github/readme/header-narrow-dark.svg">
  <source media="(max-width: 556px), (min-width: 768px) and (max-width: 844px)" srcset="https://raw.githubusercontent.com/GrafeoDB/grafeo-langchain/main/.github/readme/header-narrow-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/GrafeoDB/grafeo-langchain/main/.github/readme/header-dark.svg">
  <img alt="grafeo-langchain: LangChain graph and vector stores backed by Grafeo, in one file." src="https://raw.githubusercontent.com/GrafeoDB/grafeo-langchain/main/.github/readme/header-light.svg" width="100%">
</picture>

[![CI](https://github.com/GrafeoDB/grafeo-langchain/actions/workflows/ci.yml/badge.svg)](https://github.com/GrafeoDB/grafeo-langchain/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GrafeoDB/grafeo-langchain/graph/badge.svg)](https://codecov.io/gh/GrafeoDB/grafeo-langchain)
[![PyPI](https://img.shields.io/pypi/v/grafeo-langchain.svg)](https://pypi.org/project/grafeo-langchain/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/GrafeoDB/grafeo-langchain/blob/main/LICENSE)

LangChain graph store and vector store backed by [GrafeoDB](https://github.com/GrafeoDB/grafeo): an embedded graph database with native vector search.

No servers, no Docker, no configuration. Just `uv add` and go.

## Install

```bash
uv add grafeo-langchain

# Optional: langchain-graph-retriever integration (requires >=0.8)
uv add "grafeo-langchain[retriever]"
```

## Quick Start

### Knowledge Graph (GraphStore)

Store a knowledge graph of entities and relationships, and query it with GQL/Cypher. The examples
follow Pied Piper, the compression startup from HBO's Silicon Valley:

```python
from langchain_core.documents import Document
from grafeo_langchain import GrafeoGraphStore, GraphDocument, Node, Relationship

richard = Node(id="Richard Hendricks", type="Person", properties={"indentation": "tabs"})
gilfoyle = Node(id="Bertram Gilfoyle", type="Person")
dinesh = Node(id="Dinesh Chugtai", type="Person")
pied_piper = Node(id="Pied Piper", type="Company")
son_of_anton = Node(id="Son of Anton", type="AI", properties={"bug_fix_strategy": "delete the code"})
graph_document = GraphDocument(
    nodes=[richard, gilfoyle, dinesh, pied_piper, son_of_anton],
    relationships=[
        Relationship(source=richard, target=pied_piper, type="FOUNDED"),
        Relationship(source=gilfoyle, target=pied_piper, type="WORKS_AT"),
        Relationship(source=dinesh, target=pied_piper, type="WORKS_AT"),
        Relationship(source=gilfoyle, target=dinesh, type="RIVAL_OF"),
        Relationship(source=gilfoyle, target=son_of_anton, type="BUILT"),
    ],
    source=Document(
        page_content="Richard Hendricks founded Pied Piper. Gilfoyle and Dinesh work there and are rivals. "
        "Gilfoyle built Son of Anton, an AI that fixes bugs by deleting the code.",
        id="episode-1",
    ),
)

store = GrafeoGraphStore(db_path="./pied-piper.grafeo")
store.add_graph_documents([graph_document], include_source=True)

store.query("MATCH (p:Person)-[:WORKS_AT]->(c) RETURN p.node_id AS person, c.node_id AS company ORDER BY person")
# [{"person": "Bertram Gilfoyle", "company": "Pied Piper"}, {"person": "Dinesh Chugtai", "company": "Pied Piper"}]
print(store.get_schema)
```

Each entity's id is stored in its `node_id` property. A column without an alias (`RETURN p.node_id`)
is named after its expression.

Ingestion is idempotent: nodes are matched by id, and relationships by source, target and type, so
adding the same graph documents again updates their properties instead of duplicating them.

With `include_source=True`, each source document becomes one `SourceDocument` node, with its text in
`text` and a `MENTIONED_IN` edge from each of its entities. Its `node_id` is `source_` followed by the
document's id (the `id` metadata, else `Document.id`, else an MD5 hash of the text), so it never
clashes with an entity id:

```python
store.query("MATCH (e)-[:MENTIONED_IN]->(:SourceDocument {node_id: 'source_episode-1'}) RETURN e.node_id AS entity")
```

#### Extracting the graph with an LLM

Any chat model with structured output can turn text into graph documents. This example uses Mistral
(`uv add langchain-mistralai` and set `MISTRAL_API_KEY`); for another provider, swap in its chat
model, such as `ChatOpenAI` from `langchain-openai` or `ChatAnthropic` from `langchain-anthropic`:

```python
from typing import Any, Literal, get_args

from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field, field_validator

# The kinds of entities and relations to extract; adapt them to your documents. Fixed lists keep
# the graph consistent: one label per kind of entity, one type per relation.
EntityType = Literal["Person", "Company", "Product", "Place", "Thing"]
RelationType = Literal[
    "WORKS_AT", "LEADS", "FOUNDED", "REPORTS_TO", "WORKED_WITH", "LOCATED_IN", "MAKES", "BOUGHT", "USES", "RELATED_TO"
]
RELATIONS = (
    "WORKS_AT: person to company. LEADS: person to company they run (CEO, head, owner). FOUNDED: person to company. "
    "REPORTS_TO: person to person. WORKED_WITH: person to person. LOCATED_IN: company to place. "
    "MAKES: company to product (also software it wrote). BOUGHT: company to product. "
    "USES: company or product to product. RELATED_TO: anything else."
)


def known(value: Any, allowed: Any, other: str) -> Any:
    """Keep a type from the list; map any other to the catch-all instead of failing the document."""
    return value if value in get_args(allowed) else other


class Entity(BaseModel):
    name: str = Field(description="The full name, the same for every mention: 'Acme Robotics', not 'Acme'.")
    type: EntityType = Field(description="Places (cities, countries) are entities too.")

    @field_validator("type", mode="before")
    @classmethod
    def _known_type(cls, value: Any) -> Any:
        return known(value, EntityType, "Thing")


class Relation(BaseModel):
    source: str = Field(description="The entity the relation starts from, as in the list of relation types.")
    type: RelationType = Field(description=RELATIONS)
    target: str = Field(description="The entity the relation points to.")

    @field_validator("type", mode="before")
    @classmethod
    def _known_type(cls, value: Any) -> Any:
        return known(value, RelationType, "RELATED_TO")


class Fact(BaseModel):
    entity: str = Field(description="The name of the entity the fact is about.")
    property: str = Field(description="A snake_case name that says what the value is: price, quantity_bought, role.")
    value: str | int | float | None = Field(description="A number without its unit, or null if the text does not say.")


class Extraction(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list, description="Amounts, dates and roles the text states.")


extract = ChatMistralAI(model="mistral-small-latest", temperature=0).with_structured_output(Extraction)
# Names already in the graph, so the same entity keeps one name across documents.
known_names = {row["id"] for row in store.query("MATCH (n) WHERE NOT n:SourceDocument RETURN n.node_id AS id")}


def to_graph_document(document: Document) -> GraphDocument:
    result = extract.invoke(
        "Extract the entities, the relations between them, and the facts about them from this text. "
        f"Reuse these names for entities that are already known: {sorted(known_names)}.\n\n{document.page_content}"
    )
    nodes = {entity.name: Node(id=entity.name, type=entity.type) for entity in result.entities}
    for fact in result.facts:
        if fact.entity in nodes and fact.value is not None:
            nodes[fact.entity].properties[fact.property] = fact.value
    relationships = [
        Relationship(source=nodes[r.source], target=nodes[r.target], type=r.type)
        for r in result.relations
        if r.source in nodes and r.target in nodes
    ]
    known_names.update(nodes)
    return GraphDocument(nodes=list(nodes.values()), relationships=relationships, source=document)


documents = [
    Document(
        page_content="Gavin Belson founded Hooli in Palo Alto. Jack Barker later took over as CEO of Hooli. "
        "Richard Hendricks worked at Hooli before he founded Pied Piper.",
        id="episode-2",
    )
]
store.add_graph_documents([to_graph_document(d) for d in documents], include_source=True)
```

What makes the graph easy to query:

- **Fixed types**: the model picks entity and relation types from the lists; a type outside them
  becomes `Thing` or `RELATED_TO` instead of failing the document.
- **One name per entity**: full names, and names from earlier documents are reused, so one entity
  becomes one node.
- **Values as properties**: amounts, dates and roles become properties (`role: 'CEO'`), not nodes. A
  value the text does not state is left out, so a later document never overwrites a real one.

With Mistral Small 4 (`mistral-small-2603`) on three short company documents, this recipe and
`GrafeoQAChain` answered 11 of 15 questions over three runs, against 2 of 15 with free-form triples.
Relationship types are stored in upper case with underscores (`works at` becomes `WORKS_AT`).
`LLMGraphTransformer` output works too, but its package, `langchain-experimental`, is no longer
maintained and warns about that on import.

A `query()` that writes to the graph clears the cached schema. After writing through `store.client`,
call `store.refresh_schema()`.

`query()` runs GQL (grafeo 0.6). Results to expect:

- A number with a point is an exact decimal: `RETURN 9.99 * 3` gives `Decimal("29.97")`, and a
  property written as `price: 9.99` reads back as a `Decimal`. Write `9.99e0` for a float.
- An integer or decimal overflow and a division by zero (`1 / 0`) raise `grafeo.GrafeoError` instead
  of returning null.
- `ORDER BY` puts nulls last, ascending and descending, unless the query says `NULLS FIRST`.
- A `WHERE` right after `INSERT`, `CREATE`, `MERGE` or `DELETE` is an error: put the condition
  before the write.

`query(..., read_only=True)` runs the query in a transaction that is always rolled back, and raises
`ValueError` if the query wrote anything. Use it for queries written by an LLM or a user.

### Question Answering (GrafeoQAChain)

Ask questions in plain language: an LLM writes a GQL query from the graph schema, Grafeo runs it, and
the LLM answers from the rows. It needs only `langchain-core` and any chat model, for example
`ChatMistralAI(model="mistral-small-latest")` from `langchain-mistralai`:

```python
from langchain_mistralai import ChatMistralAI
from grafeo_langchain import GrafeoQAChain

llm = ChatMistralAI(model="mistral-small-latest", temperature=0)
chain = GrafeoQAChain.from_llm(llm, graph=store)
chain.invoke({"query": "Who built Son of Anton?"})
# for example: {"query": "Who built Son of Anton?", "result": "Bertram Gilfoyle built Son of Anton."}
```

- **Read-only by default**: a generated query that would write (`INSERT`, `SET`, `DELETE`, ...) is
  rolled back and raises `ValueError`, so the graph stays unchanged. `allow_writes=True` runs it.
- **Failing queries are fixed**: a query that fails goes back to the LLM with grafeo's error, once by
  default (`max_retries`). A reply that is not a query gets the same retry, then raises `ValueError`.
- **Made for extracted graphs**: names in an LLM-extracted graph rarely match the question exactly, and
  one relation can have several types. The query prompt matches names by a distinctive word
  (`CONTAINS 'hooli'`), leaves node labels out and lists alternative relationship types. A query that
  matches nothing gets one looser try (`retry_on_empty`, read-only chains only); if that finds nothing
  either, the answer says the graph does not hold it.
- **Schema in the prompt**: `store.get_schema` lists each label and relationship type with its
  properties and their types, and the relationship patterns. Patterns that differ only in a label
  share a line (`(:Company|Organization)-[:LOCATED_IN]->(:City)`), and the edges to source documents
  take one, `()-[:MENTIONED_IN]->(:SourceDocument)`:

  ```text
  Node labels and properties:
  AI {bug_fix_strategy: STRING, node_id: STRING}
  Company {node_id: STRING}
  Person {indentation: STRING, node_id: STRING}
  SourceDocument {node_id: STRING, text: STRING}
  Relationship types and properties:
  BUILT
  FOUNDED
  MENTIONED_IN
  RIVAL_OF
  WORKS_AT
  Relationships:
  (:Person)-[:BUILT]->(:AI)
  (:Person)-[:FOUNDED]->(:Company)
  (:Person)-[:RIVAL_OF]->(:Person)
  (:Person)-[:WORKS_AT]->(:Company)
  ()-[:MENTIONED_IN]->(:SourceDocument)
  ```

- **Options**: `top_k` (result rows the LLM sees, 10 by default), `return_direct` (return the rows
  without an answer), `return_intermediate_steps` (the query and the rows), `retry_on_empty` (on by
  default), separate `query_llm` and `qa_llm`, and custom `query_prompt`, `fix_prompt` and `qa_prompt`.
- **A LangChain runnable**: `invoke`, `ainvoke`, `batch`, composition with `|` and callbacks
  (LangSmith tracing) work as for any chain. Input and output keys follow `GraphCypherQAChain`.

The `GraphCypherQAChain` of langchain-neo4j also accepts the store as it is. The one of
langchain-community does not: it requires a subclass of its own `GraphStore` class.

#### A graph the store did not write: the Silicon Valley dataset

`grafeo-datasets` ships the Pied Piper story as a graph of 64 nodes and 149 edges
(`uv add "grafeo-datasets[grafeo]"`). Load it into a store and ask away:

```python
from grafeo_datasets import silicon_valley

pied_piper = GrafeoGraphStore()
silicon_valley.load(pied_piper.client)

# How is Son of Anton related to Anton? Gilfoyle built both:
pied_piper.query(
    "MATCH (:AI {name: 'Son of Anton'})<-[:BUILT]-(p:Person)-[:BUILT]->(:AI {name: 'Anton'}) RETURN p.name AS builder"
)
# [{"builder": "Bertram Gilfoyle"}]

chain = GrafeoQAChain.from_llm(llm, graph=pied_piper)
chain.invoke({"query": "Does Jian-Yang's fridge talk?"})["result"]
# for example: "Yes, Jian-Yang's fridge talks."
chain.invoke({"query": "What did Jian-Yang build?"})["result"]
# for example: "Jian-Yang built Not Hotdog and SeeFood."
```

The dataset's entities are named by a `name` property, not `node_id`. The chain reads that from the
schema and matches on `name` (set `id_property` to choose another). With Mistral Small 4, the two
questions above, "How does Son of Anton fix bugs?", "Where has Monica Hall worked?" and "Where was
Pied Piper based?" were answered right in 3 of 3 runs. A question that needs a path the model has to
work out, such as the Son of Anton one, is safer as a GQL query like the one above.

See [examples/graph_qa_chain.py](https://github.com/GrafeoDB/grafeo-langchain/blob/main/examples/graph_qa_chain.py)
for a run without an API key.

### Vector + Graph Retrieval (GraphVectorStore)

Combine vector similarity search with graph traversal for Graph RAG:

```python
from langchain_mistralai import MistralAIEmbeddings
from grafeo_langchain import GrafeoGraphVectorStore

embeddings = MistralAIEmbeddings(model="mistral-embed")
store = GrafeoGraphVectorStore(
    embedding=embeddings,
    db_path="./pied-piper-docs.grafeo",
    # embedding_dimensions auto-detected from the model
)

store.add_texts(
    texts=[
        "Son of Anton is an AI that fixes bugs by deleting the code. It once ordered 4000 pounds of meat.",
        "Anton is Gilfoyle's server, the machine that runs Pied Piper's network.",
        "Jian-Yang's fridge has a screen and a voice, and is on more of the house network than it needs.",
        "Not Hotdog is Jian-Yang's app: it tells whether a photo shows a hotdog or not a hotdog.",
    ],
    metadatas=[
        {"kind": "AI", "__graph_links__": [{"target_id": "anton", "type": "TRAINED_ON"}]},
        {"kind": "AI"},
        {"kind": "AI", "__graph_links__": [{"target_id": "not-hotdog", "type": "SAME_OWNER"}]},
        {"kind": "app"},
    ],
    ids=["son-of-anton", "anton", "jian-yangs-fridge", "not-hotdog"],
)

# Standard vector search
docs = store.similarity_search("Which AI deletes code?", k=2)

# Vector search + graph traversal: Son of Anton, then Anton, which it was trained on
docs = store.traversal_search("an AI that deletes code", k=1, depth=1)

# MMR-diversified graph traversal
docs = store.mmr_traversal_search("Jian-Yang's gadgets", k=2, depth=1, lambda_mult=0.7)

# Filtered search (only documents with matching metadata)
docs = store.similarity_search("smart appliances", k=4, filter={"kind": "AI"})

# Standard LangChain retrievers: similarity, MMR and score threshold
retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 2})
docs = retriever.invoke("Jian-Yang's gadgets")

# Delete documents (and their graph links)
store.delete(["not-hotdog"])
```

Adding a text whose id is already stored replaces that document: its text, embedding and metadata
are overwritten, its graph links are kept, and new links are added once. Texts added without an id
get a generated id (`"0"`, `"1"`, ...) that no stored document uses. `add_documents` and
`from_documents` use `Document.id` when it is set. The metadata keys `doc_id`, `text`, `embedding`
and `_depth` are reserved for the store and ignored.

`traversal_search` returns the seed documents first, then the linked documents closest to a seed
first, up to `2 * k` documents in total.

### Persistence

Pass a `db_path` to keep everything in a single file. Close the store, reopen it later, and your documents, embeddings, and graph links are all still there:

```python
from langchain_mistralai import MistralAIEmbeddings
from grafeo_langchain import GrafeoGraphVectorStore

embeddings = MistralAIEmbeddings(model="mistral-embed")

# Write phase
store = GrafeoGraphVectorStore(embedding=embeddings, db_path="./my_store.grafeo")
store.add_texts(["Richard indents with tabs", "Russ Hanneman puts radio on the internet"], ids=["tabs", "roi"])
store.close()

# Later: reopen and query
store = GrafeoGraphVectorStore(embedding=embeddings, db_path="./my_store.grafeo")
docs = store.similarity_search("tabs or spaces?", k=1)
store.close()
```

Omit `db_path` (or pass `None`) for a purely in-memory store that is discarded when the process exits.

The database is one file at `db_path`, whatever its extension (`.grafeo` is the usual one). While
the store is open, grafeo keeps its write-ahead log in a `<db_path>.wal/` directory next to it, and
`close()` folds the log into the file. The directory that holds the file must be writable.

`db_path` must not be an existing directory: grafeo 0.6 refuses a directory that is not a grafeo
0.5.x database, an empty one included, and raises `grafeo.GrafeoError`. Use a file path inside it,
such as `./data/my_store.grafeo`.

Reopening a store with an embedding model of another dimensionality raises `ValueError`.

#### Upgrading from grafeo 0.5 (grafeo-langchain 0.2)

grafeo-langchain 0.3 requires grafeo 0.6, which has a new database file format. The first time grafeo
0.6 opens a database written by grafeo 0.5.x for writing (opening a store does), it migrates the
database in place and keeps the old files, unchanged, next to it:

| Written by 0.5.x as | After the first open with 0.6 | Old files kept as |
| --- | --- | --- |
| a file, such as `./my_store.grafeo` | a file at the same path | `./my_store.grafeo.pre-0.6` (and `./my_store.grafeo.pre-0.6.wal` if it had a log) |
| a WAL directory, any path without `.grafeo`, such as `./my_store.db` | a file at the same path | the directory, as `./my_store.db.pre-0.6/` |

grafeo 0.5.x cannot open a migrated database, so:

1. Stop every process that uses the database with grafeo 0.5.x (grafeo-langchain 0.2 or another
   grafeo program).
2. Back it up: copy the file or the WAL directory somewhere else.
3. Upgrade: `uv add "grafeo-langchain>=0.3"`.
4. Open the store once and check your documents, for example with `similarity_search`.
5. Keep the `.pre-0.6` copy until you no longer need to go back.

To go back, delete the migrated file (and the `<path>.wal` directory next to it, if there is one),
rename the `.pre-0.6` file or directory to its old name, and pin both packages: `uv add "grafeo-langchain<0.3" "grafeo<0.6"`. grafeo-langchain 0.2 does not cap
grafeo itself, so without the second pin it installs grafeo 0.6 and migrates the database again.

grafeo 0.7 will no longer read 0.5.x databases, so open them with 0.6 before then. See
[Upgrading from 0.5](https://docs.grafeo.dev/operate/persistence/persistent/#upgrading-from-05).

### Graph Retriever Integration

> **Note:** The `[retriever]` extra is required for this feature. Install with
> `uv add "grafeo-langchain[retriever]"` (requires `langchain-graph-retriever>=0.8`).

Use `GrafeoAdapter` with [langchain-graph-retriever](https://github.com/datastax/graph-rag)
for advanced traversal strategies (Eager, BFS, MMR) via metadata edges:

```python
from grafeo_langchain import GrafeoGraphVectorStore
from grafeo_langchain.adapter import GrafeoAdapter
from langchain_graph_retriever import GraphRetriever

store = GrafeoGraphVectorStore(embedding=embeddings)
store.add_texts(
    texts=[
        "Pied Piper compresses files with middle-out, with a Weissman score of 5.2.",
        "Richard Hendricks founded Pied Piper after he left Hooli.",
        "Hooli's motto is 'Making the world a better place'.",
    ],
    metadatas=[{"company": "Pied Piper"}, {"company": "Pied Piper"}, {"company": "Hooli"}],
    ids=["middle-out", "richard", "hooli"],
)

adapter = GrafeoAdapter(vector_store=store)
retriever = GraphRetriever(store=adapter, edges=[("company", "company")])
docs = retriever.invoke("compression")  # middle-out first, and Richard through the shared company
```

## Filters

All filter parameters use **exact-match equality**. Pass a dict where each key is a metadata field name and the value is the expected value. Only documents whose metadata matches every key-value pair are returned:

```python
docs = store.similarity_search("compression", k=4, filter={"company": "Pied Piper"})
```

Supported value types: `str`, `int`, `float`, `bool` and `decimal.Decimal`. Compound types (lists, dicts) are not supported as filter values.

A value matches only a stored value of the same type: `{"year": 2024}` does not match `2024.0`,
`{"active": 1}` does not match `True`, and `{"price": Decimal("1.5")}` matches a stored
`Decimal("1.50")` but not the float `1.5`. In `traversal_search` and `mmr_traversal_search` the
filter applies to the linked documents as well as to the seeds.

## Metadata

The stores keep metadata values of type `str`, `int`, `float`, `bool` and `decimal.Decimal`
(grafeo 0.6 stores a `Decimal` exactly, with its scale) and leave out other values, such as lists,
dicts and `None`. Documents come back with the metadata as stored, plus `id`. Search results also
carry:

- `_depth`: `0` for a search hit, the number of hops for a linked document. The name follows
  langchain-graph-retriever, which sets the same key on its results.
- `score`: relevance from 0 to 1 (`1 - cosine distance / 2`: 1 is identical, 0.5 unrelated, 0
  opposite), `None` for a linked document.
- `source`: `"vector"` for a search hit, `"graph_traversal"` for a linked document.

A stored `source` or `score` wins over the store's value: documents from a LangChain loader keep
the `source` the loader set. Use `_depth` to tell search hits from linked documents.

langchain-graph-retriever 0.8
follows edges on `str`, `bool`, `int` and `float` values only: it skips a `Decimal` field with a
warning, so use another type for a field that `GrafeoAdapter` traversal should follow.

## Graph Links Format

Graph links between documents are specified via the `__graph_links__` metadata key. Each link is a dict with the following fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `target_id` | `str` | Yes | The `id` of the target document |
| `type` | `str` | No | Edge label (defaults to `LINKS_TO`) |
| `properties` | `dict` | No | Additional properties stored on the edge |

Example:

```python
store.add_texts(
    texts=["Gilfoyle and Dinesh argue about everything.", "Dinesh wrote the Pied Piper video chat."],
    metadatas=[
        {
            "__graph_links__": [
                {"target_id": "dinesh", "type": "MENTIONS"},
                {"target_id": "gilfoyle", "type": "RIVAL_OF", "properties": {"since_season": 1}},
            ]
        },
        {},
    ],
    ids=["rivalry", "dinesh"],
)
```

The `__graph_links__` key is consumed during ingestion and is not stored as document metadata.

Links are created after all documents of the same `add_texts` call are stored, so a link may point to
a document later in the same batch. A link to an id that is not stored yet is skipped: add target
documents first, or in the same batch. Adding the same link again keeps one edge and updates its
properties.

## Why Grafeo?

Grafeo runs inside the Python process, so a graph store for LangChain is one `uv add` away:

- **Setup**: `uv add grafeo-langchain`. No server, Docker, URL or credentials.
- **Vector search**: built in (HNSW); the store creates the index.
- **Graph algorithms**: built in (PageRank, Louvain, ...), with no plugin to install.
- **Query languages**: GQL, Cypher and Gremlin.
- **Data on disk**: one `.grafeo` file, or memory only.
- **Tests and notebooks**: an in-memory store, with nothing to start or clean up.

The database lives in one process: when several processes or machines share one graph, run them through
a single service in front of it.

## API Reference

### `GrafeoGraphStore`

- `GrafeoGraphStore(*, db_path=None)`: in-memory or persistent graph store
- `.add_graph_documents(docs, *, include_source=False)`: ingest LLM-extracted graph documents
- `.query(query, params=None, *, read_only=False)`: execute GQL/Cypher queries; `read_only=True` refuses writes and leaves the graph unchanged
- `.get_schema` / `.get_structured_schema`: the graph schema as prompt text, or as a dict (with the `node_props`, `rel_props` and `relationships` keys of langchain-neo4j)
- `.refresh_schema()`: refresh the cached schema
- `.client`: access the underlying `GrafeoDB` instance
- `.close()`: close the database; with a `db_path`, this folds the write-ahead log into the file. The store is also a context manager (`with GrafeoGraphStore(...) as store:`) that closes on exit

### `GrafeoGraphVectorStore`

- `GrafeoGraphVectorStore(embedding, *, db_path=None, embedding_dimensions=None)`: vector store with graph links (dimensions auto-detected from the model)
- `.add_texts(texts, metadatas=None, ids=None)`: add documents with embeddings and optional graph links
- `.add_documents(documents)`: add LangChain documents, using `Document.id` as the id when set
- `.similarity_search(query, k=4, *, filter=None)`: standard vector similarity search
- `.similarity_search_by_vector(embedding, k=4, *, filter=None)`: search by pre-computed vector
- `.similarity_search_with_score(query, k=4, *, filter=None)`: search results with their cosine distance (0 is identical, 2 opposite); `similarity_search_with_relevance_scores` gives the 0 to 1 relevance of `metadata["score"]` instead, which `as_retriever(search_type="similarity_score_threshold")` compares with its `score_threshold`
- `.max_marginal_relevance_search(query, k=4, fetch_k=20, lambda_mult=0.5, *, filter=None)`: relevant and diverse results, with Grafeo's native MMR
- `.get_by_ids(ids)`: stored documents by id, in the order given
- `.traversal_search(query, *, k=4, depth=1, filter=None)`: vector search + graph traversal
- `.mmr_traversal_search(query, *, k=4, depth=2, fetch_k=100, lambda_mult=0.5, filter=None)`: MMR-diversified traversal
- `.delete(ids)`: remove documents by ID, with their graph links
- `.from_texts(...)` / `.from_documents(...)`: factory methods
- `.as_retriever(search_type=...)`: LangChain retriever, with `"similarity"`, `"mmr"` or `"similarity_score_threshold"`
- `.close()`: close the database; with a `db_path`, this folds the write-ahead log into the file. The store is also a context manager that closes on exit

The async variants (`asimilarity_search`, `aget_by_ids`, ...) run the sync methods in an executor.

### `GrafeoQAChain`

- `GrafeoQAChain.from_llm(llm=None, *, graph, query_llm=None, qa_llm=None, **options)`: question answering over a `GrafeoGraphStore`. `llm` serves both steps; `query_llm` and `qa_llm` replace it for one step each, and with both given `llm` can be left out
- Options: `top_k=10`, `max_retries=1`, `retry_on_empty=True`, `id_property=None` (`node_id` when the graph has it, else `name`), `allow_writes=False`, `return_direct=False`, `return_intermediate_steps=False`, `query_prompt`, `fix_prompt`, `qa_prompt`, `input_key="query"`, `output_key="result"`
- `.invoke({"query": question})` or `.invoke(question)`: returns `{"query": question, "result": answer}`; also `ainvoke`, `batch`, `abatch`
- Raises `ValueError` for a generated query that would write (unless `allow_writes=True`), `ValueError("the model did not return a query: ...")` when the LLM still replies with prose after `max_retries` tries, and `grafeo.GrafeoError` for a query that still fails after `max_retries` fixes. Code fences and labels around a query (```` ```gql ````, `Query:`) are removed before it runs
- Generated queries run under the database's query timeout (30 seconds by default, `query_timeout` of `grafeo.GrafeoDB`)

### `GrafeoAdapter`

Requires `uv add "grafeo-langchain[retriever]"`.

- `GrafeoAdapter(vector_store)`: adapter for `langchain-graph-retriever`
- Works with `GraphRetriever(store=adapter, edges=[...])` for Eager/BFS strategies

## Requirements

- Python 3.12+
- grafeo 0.6 (`grafeo>=0.6.2,<0.7`). Databases written by grafeo 0.5.x
  are migrated on first open: see [Upgrading from grafeo 0.5](https://github.com/GrafeoDB/grafeo-langchain#upgrading-from-grafeo-05-grafeo-langchain-02).
- langchain-core 1.2 or later
- langchain-graph-retriever 0.8 or later, for `GrafeoAdapter` (the `[retriever]` extra)

## License

Apache-2.0

"""GrafeoGraphStore on the Silicon Valley dataset: load a graph, query it with GQL, add your own facts.

Demonstrates:
- Loading the grafeo-datasets Silicon Valley graph (Pied Piper, 64 nodes, 149 edges) into a store
- The schema text an LLM would see
- GQL queries: a two-hop path, investors in date order, properties
- Adding your own facts as graph documents, idempotently

No API key needed. Run:
    uv run python examples/basic_graph_store.py
"""

from __future__ import annotations

from grafeo_datasets import silicon_valley
from langchain_core.documents import Document

from grafeo_langchain import GrafeoGraphStore, GraphDocument, Node, Relationship

# ── The dataset ───────────────────────────────────────────────────────────────

store = GrafeoGraphStore()
silicon_valley.load(store.client)

schema = store.get_schema.splitlines()
print("=== Schema (first lines of what an LLM sees) ===")
print("\n".join(schema[:4]), "\n  ...", f"({len(schema)} lines)")

print("\n=== How is Son of Anton related to Anton? ===")
for row in store.query(
    "MATCH (:AI {name: 'Son of Anton'})<-[:BUILT]-(p:Person)-[:BUILT]->(:AI {name: 'Anton'}) RETURN p.name AS builder"
):
    print(f"  {row['builder']} built both")

print("\n=== Who invested in Pied Piper? ===")
for row in store.query(
    "MATCH (i)-[e:INVESTED_IN]->(:Company {name: 'Pied Piper'}) RETURN i.name AS investor, e.round AS round "
    "ORDER BY e.date"
):
    print(f"  {row['investor']} ({row['round']})")

print("\n=== Does Jian-Yang's fridge talk? ===")
for row in store.query(
    'MATCH (f:AI {name: "Jian-Yang\'s fridge"})<-[:OWNS]-(p) RETURN p.name AS owner, f.talks AS talks'
):
    print(f"  talks: {row['talks']}, owned by {row['owner']}")

print("\n=== Who has the most commas? ===")
for row in store.query(
    "MATCH (p:Person) WHERE p.net_worth_commas IS NOT NULL "
    "RETURN p.name AS person, p.net_worth_commas AS commas ORDER BY commas DESC LIMIT 1"
):
    print(f"  {row['person']}: {row['commas']} commas")

print("\n=== Rivals ===")
for row in store.query("MATCH (a)-[:RIVAL_OF]->(b) RETURN a.name AS a, b.name AS b ORDER BY a, b"):
    print(f"  {row['a']} vs {row['b']}")

# ── Your own facts ────────────────────────────────────────────────────────────

episodes = GrafeoGraphStore()
gavin = Node(id="Gavin Belson", type="Person", properties={"pledge": "Tethics"})
jack = Node(id="Jack Barker", type="Person", properties={"nickname": "Action Jack"})
hooli = Node(id="Hooli", type="Company", properties={"motto": "Making the world a better place"})
episode = GraphDocument(
    nodes=[gavin, jack, hooli],
    relationships=[
        Relationship(source=gavin, target=hooli, type="FOUNDED"),
        Relationship(source=jack, target=hooli, type="LEADS", properties={"role": "CEO"}),
    ],
    source=Document(page_content="Gavin Belson founded Hooli. Jack Barker took over as its CEO.", id="season-3"),
)
episodes.add_graph_documents([episode], include_source=True)
episodes.add_graph_documents([episode], include_source=True)  # again: updates, does not duplicate

print("\n=== Your own facts: who leads Hooli? ===")
for row in episodes.query(
    "MATCH (p:Person)-[l:LEADS]->(c:Company {node_id: 'Hooli'}) RETURN p.node_id AS person, l.role AS role"
):
    print(f"  {row['person']} ({row['role']})")
nodes = episodes.query("MATCH (n) RETURN count(n) AS n")[0]["n"]
print(f"  {nodes} nodes after adding the episode twice: 3 entities and 1 source document")

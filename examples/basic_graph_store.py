"""Basic GrafeoGraphStore usage — knowledge graph from structured data.

Demonstrates:
- Creating a graph store (in-memory)
- Ingesting graph documents with nodes and relationships
- Querying with GQL/Cypher
- Inspecting the schema

Run:
    uv run python examples/basic_graph_store.py
"""

from __future__ import annotations

from langchain_core.documents import Document

from grafeo_langchain import GrafeoGraphStore, GraphDocument, Node, Relationship

# ── Build graph documents ──────────────────────────────────────────────────────

alice = Node(id="alice", type="Person", properties={"name": "Alice", "role": "Engineer"})
bob = Node(id="bob", type="Person", properties={"name": "Bob", "role": "Manager"})
acme = Node(id="acme", type="Company", properties={"name": "Acme Corp", "industry": "Tech"})

source = Document(page_content="Alice and Bob work at Acme Corp. Alice knows Bob.")

graph_doc = GraphDocument(
    nodes=[alice, bob, acme],
    relationships=[
        Relationship(source=alice, target=acme, type="WORKS_AT"),
        Relationship(source=bob, target=acme, type="WORKS_AT"),
        Relationship(source=alice, target=bob, type="KNOWS"),
    ],
    source=source,
)

# ── Ingest into graph store ────────────────────────────────────────────────────

store = GrafeoGraphStore()  # in-memory
store.add_graph_documents([graph_doc], include_source=True)

# ── Query the graph ────────────────────────────────────────────────────────────

print("=== Schema ===")
print(store.get_schema)
print()

print("=== Who works at Acme? ===")
results = store.query("MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.name, c.name")
for row in results:
    print(f"  {row}")

print()
print("=== Who does Alice know? ===")
results = store.query("MATCH (a {node_id: 'alice'})-[:KNOWS]->(b) RETURN b.name")
for row in results:
    print(f"  {row}")

print()
print("=== Upsert: update Alice's role ===")
updated_alice = Node(id="alice", type="Person", properties={"name": "Alice", "role": "Senior Engineer"})
store.add_graph_documents([GraphDocument(nodes=[updated_alice], relationships=[], source=source)])

results = store.query("MATCH (n {node_id: 'alice'}) RETURN n.name, n.role")
for row in results:
    print(f"  {row}")

store.close()

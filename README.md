# grafeo-langchain

[LangChain](https://github.com/langchain-ai/langchain) integration for the [Grafeo](https://github.com/GrafeoDB/grafeo) graph database.

Provides `GrafeoGraphStore` and `GrafeoGraphVectorStore` for knowledge-graph storage and hybrid graph+vector retrieval inside LangChain pipelines.

## Status

Work in progress.

## Features (planned)

- `GrafeoGraphStore` &mdash; knowledge graph storage backed by Grafeo
- `GrafeoGraphVectorStore` &mdash; hybrid vector + graph retrieval
- Three retrieval strategies: similarity search, traversal search, MMR traversal
- GraphDocument helpers for ingestion
- Zero infrastructure &mdash; Grafeo is embedded, no external services required

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management

## License

Apache-2.0 &mdash; see [LICENSE](LICENSE) for details.

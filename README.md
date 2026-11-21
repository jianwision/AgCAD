# AgCAD
Agentic GPU CAD

CATIA-class modeling with Houdini-style procedural networks, executed on the GPU, driven by
LLM agents (batch programs over MCP) or GPU-resident Candle-trained models (device bytecode),
with a GPU-resident harness and recursive self-improvement. Pure Rust; first MVP is headless.

- Architecture spec: [docs/spec](docs/spec/README.md)
- Implementation status (not yet built or tested): [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md)
- Validated prototype (reference): [docs/Demo](docs/Demo/README.md)

```sh
cargo xtask check                                  # spec consistency
cargo run -p agcad-cli -- version                  # build deferred: first build pass pending
cargo run -p agcad-cli -- serve --requirements req.json   # headless MCP server (stdio)
```

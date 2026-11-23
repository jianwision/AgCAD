# Building reliable CAD chat tools

AgCAD has two entry points. The browser calls the viewer's HTTP session API. The viewer sends function definitions to DeepSeek and validates proposed edits with the local CAD engine. The separate AgCAD MCP server exposes JSON-RPC tools over stdio and advertises protocol version 2025-06-18. A passing browser replay verifies the viewer path. An MCP client additionally needs discovery, schema, error-envelope and version-negotiation tests. [MCP tool specification for the advertised version](https://modelcontextprotocol.io/specification/2025-06-18/server/tools) · [DeepSeek tool calling](https://api-docs.deepseek.com/guides/tool_calls/)

## Failures and practices

| Failure | What to do | What to check in AgCAD |
|---|---|---|
| The assistant asks for a value the user delegated | Put reversible defaults in the tool implementation and show the chosen values in the preview. Ask when choosing a target changes the meaning of the edit. | “Create a sphere” should preview a 20 mm diameter at the origin. “You fill now” should use the preceding sphere request. An imported part with several extrusion targets should prompt for a face. |
| The assistant guesses tools, arguments or edge IDs | Give common tasks dedicated tools. Resolve selection and capability against the current body on the server. Fetch detailed operation signatures on demand. | Whole-body filleting and vertical-corner filleting have different effects. Expose the narrower option explicitly. Selecting one face does not turn an all-edges request into a face-boundary edit. |
| A valid tool call produces the wrong result | Define distance, units, reference frame and expected effect. Check an independently known postcondition. | A 30 mm outward extension on a 10 mm plate yields 40 mm. Check all three sphere extents, the center, and the saved revision. A kernel acceptance flag alone missed both earlier defects. |
| Tool errors provoke repeated questions or silent substitutions | Distinguish a correctable argument, an unsupported operation, missing selection and a provider failure. Give the model the correction it can act on, with a bounded retry. | Return the actual fillet capability on an unsupported body. Inject truncated and empty provider replies and assert the number of attempts and resulting previews. |
| Session expiry loses work or duplicates an edit | Keep document identity independent of the connection. Reopen the same model, preserve chat and draft, refresh selection and invalidate expired proposals. For uncertain durable writes, read back the revision and consider an idempotency key before enabling automatic retries. | Close a session immediately before Apply. The browser should recover its draft with zero saved edits. A fresh preview and explicit Apply should save exactly one. |
| The wrong conversation or file content directs an edit | Resend current selection and revision from server state, bound historical context, and treat labels, model metadata and external tool results as data. Enforce access and validate arguments at execution. | Across two sessions, change the document in one and attempt to apply the older preview in the other. On shared deployments, bind edit rights to the caller independently of the session ID. |

Anthropic describes task-oriented tool design, compact outputs and actionable errors. MCP defines input schemas, structured tool results and error reporting. For the advertised version, domain execution errors can be returned as tool results with `isError`; validate arguments at the server even when the model provider supports schema enforcement. [Anthropic’s tool design guide](https://www.anthropic.com/research/writing-tools-for-agents) · [MCP tools](https://modelcontextprotocol.io/specification/2025-06-18/server/tools) · [MCP security guidance](https://modelcontextprotocol.io/specification/2025-11-25/basic/security_best_practices)

The viewer uses HTTP 410 for its expired viewport sessions. A client implementing the versioned MCP Streamable HTTP transport must follow that transport's recovery rules, including reinitialization after an expired MCP session returns HTTP 404. [MCP transport specification](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)

## Replay and measurement

Build the Windows viewer in release mode. From the repository root run the following checks. The live mode uses the server's configured model account. Each run starts a temporary viewer with an isolated model library. Browser mode requires Playwright and Edge.

```powershell
python crates/viewer/tests/chat_replay.py
python crates/viewer/tests/chat_replay.py --mcp
python crates/viewer/tests/chat_replay.py --browser
python crates/viewer/tests/chat_replay.py --live
```

Keep the provider fixture deterministic for error recovery. Use the live mode for wording, delegation, selection and tool routing. For prompt or model comparisons, repeat the same cases with each variant. Record verified final-state successes over attempts, cases that pass every repetition, unnecessary clarification turns, tool retries and latency. A geometry mismatch fails the case even if the response reads well. Record the prompt variant, model, fixture, operation and revision so failures can be reproduced without logging credentials. The τ-bench authors demonstrate end-state scoring and repeated-trial reliability for interactive tool use. Their published success rates describe their benchmark domains. AgCAD needs its own measurements. [τ-bench evaluation method](https://arxiv.org/abs/2406.12045)

The `--mcp` replay uses the Python MCP SDK and a release build of `agcad-cli`. It starts `agcad serve`, negotiates the protocol, checks discovery and error envelopes, and compares the sphere and extended plate dimensions with the viewer's actual HTTP results. Face queries return zero-based indices alongside names and the workspace revision for subsequent geometry tools. CAD measurements, queries, tree, validation and execution reports expose structured output alongside their textual result. [MCP tool results and output schemas](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)

The browser sends the server-issued `preview_id` with Apply, Discard, Current and Proposed. A mismatched or repeated token returns HTTP 409. Existing HTTP callers may omit it to act on the current session preview. Face selection or proposal changes during an assistant request invalidate the earlier result.

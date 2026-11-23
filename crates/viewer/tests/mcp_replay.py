"""Compare measured HTTP results with a fresh AgCAD MCP session via the Python SDK."""

import json
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError


async def compare(root, directory, sphere_size, extended_size):
    req = Path(directory) / "mcp-requirements.json"
    req.write_text(
        json.dumps({"id": "replay", "rev": 1, "deliver": {"body_type": "SolidBody"}})
    )
    params = StdioServerParameters(
        command=str(root / "target/release/agcad.exe"),
        args=["serve", "--requirements", str(req)],
    )
    # Each geometry starts in its own CAD workspace.
    for shape, expected in [("sphere", sphere_size), ("box", extended_size)]:
        async with stdio_client(params) as streams, ClientSession(*streams) as session:
            init = await session.initialize()
            assert init.protocolVersion == "2025-06-18"
            definitions = {t.name: t for t in (await session.list_tools()).tools}
            assert definitions["query"].outputSchema

            async def invoke(name, args=None):
                result = await session.call_tool(name, args or {})
                assert not result.isError, result.content
                assert json.loads(result.content[0].text) == result.structuredContent
                return result.structuredContent

            async def run(statements, revision):
                program = {
                    "expected_revision": revision,
                    "requirements_rev": 1,
                    "program": statements,
                }
                assert (await invoke("check_program", {"program": program}))["ok"]
                assert (await invoke("run_program", {"program": program}))["report"][
                    "status"
                ] == "ACCEPTED"

            if shape == "sphere":
                invalid = await session.call_tool("describe", {})
                assert invalid.isError
                for name, args in [
                    ("missing_tool", {}),
                    ("run_program", {}),
                    ("query", {"role": 3}),
                ]:
                    try:
                        await session.call_tool(name, args)
                    except McpError as err:
                        assert err.error.code == -32602
                    else:
                        raise AssertionError("expected invalid params")
            await run(
                [
                    {
                        "id": "base",
                        "op": "part.primitive",
                        "args": {
                            "axis": "xy",
                            "kind": shape,
                            "p": [10] if shape == "sphere" else [40, 30, 10],
                        },
                    }
                ],
                0,
            )
            if shape == "box":
                end = await invoke("query", {"role": "cap_end"})
                assert (
                    len(end["faces"]) == 1
                    and end["faces"][0]["name"] == end["elements"][0]
                )
                await run(
                    [
                        {
                            "id": "extend",
                            "op": "direct.push_pull",
                            "args": {"faces": [end["faces"][0]["index"]], "d": "3 cm"},
                        }
                    ],
                    end["revision"],
                )
            size = (await invoke("describe"))["bounding_box"]["size"]
            assert size == expected, (size, expected)
            print(f"PASS MCP/HTTP {shape} dimensions {size}", flush=True)

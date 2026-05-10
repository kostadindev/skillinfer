"""E2E test for `skillinfer-mcp` installed from PyPI.

Spawns the `skillinfer-mcp` console script over stdio, drives it via
the MCP Python SDK, and exercises a representative tool flow.
"""
from __future__ import annotations

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run() -> None:
    params = StdioServerParameters(command="skillinfer-mcp", args=[])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = {t.name for t in tools.tools}
            required = {
                "load_dataset", "create_profile", "observe",
                "predict", "most_uncertain", "match_task",
            }
            missing = required - tool_names
            assert not missing, f"missing tools: {missing}"

            # Load population.
            r = await session.call_tool("load_dataset", {"name": "p", "dataset": "onet"})
            text = r.content[0].text
            assert "Population 'p' loaded" in text, text

            # List features and pick one to observe.
            r = await session.call_tool("list_features", {"population": "p"})
            features = json.loads(r.content[0].text)["features"]
            assert len(features) > 50

            # Create profile and observe.
            await session.call_tool("create_profile", {"name": "alice", "population": "p"})
            r = await session.call_tool(
                "observe", {"profile": "alice", "skill": features[0], "score": 0.8}
            )
            assert "Total observations: 1" in r.content[0].text

            # Predict — top_k=3.
            r = await session.call_tool("predict", {"profile": "alice", "top_k": 3})
            preds = json.loads(r.content[0].text)
            assert len(preds) == 3
            assert "mean" in preds[0] and "std" in preds[0]

            # match_task with one feature.
            r = await session.call_tool(
                "match_task",
                {"profile": "alice", "task_weights": {features[0]: 1.0}, "threshold": 0.5},
            )
            score = json.loads(r.content[0].text)
            assert "score" in score and "p_above_threshold" in score

            print(f"OK  skillinfer-mcp: {len(tool_names)} tools, full flow pass")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except AssertionError as e:
        print(f"FAIL  {e}", file=sys.stderr)
        sys.exit(1)

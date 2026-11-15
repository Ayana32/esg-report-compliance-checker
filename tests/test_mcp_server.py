"""Exercise the evidence tool over the MCP Streamable HTTP transport."""

import asyncio
from unittest.mock import MagicMock

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.mcp_server import create_server


def test_evidence_search_protocol():
    retriever = MagicMock()
    retriever.search.return_value = [
        {
            "chunk_id": "IBK-64",
            "text": "Direct greenhouse gas emissions.",
            "metadata": {"page_num": 64},
            "score": 2.0,
        },
        {
            "chunk_id": "IBK-65",
            "text": "No lexical match.",
            "metadata": {},
            "score": 0.0,
        },
    ]

    async def exercise():
        app = create_server(retriever).streamable_http_app()
        url = "http://127.0.0.1:8001/mcp"
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
            ) as http_client:
                async with streamable_http_client(url, http_client=http_client) as (
                    read,
                    write,
                    _,
                ):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = (await session.list_tools()).tools
                        assert [tool.name for tool in tools] == ["search_esg_evidence"]
                        assert tools[0].annotations.readOnlyHint is True

                        call = {
                            "company_id": " IBK ",
                            "disclosure_item": "Scope 1 emissions",
                            "limit": 3,
                        }
                        result = await session.call_tool("search_esg_evidence", call)
                        assert not result.isError
                        assert result.structuredContent == {
                            "company_id": "IBK",
                            "evidence": [
                                {
                                    "chunk_id": "IBK-64",
                                    "text": "Direct greenhouse gas emissions.",
                                    "page_number": 64,
                                }
                            ],
                        }
                        retriever.search.assert_called_once_with(
                            "Scope 1 emissions", n_results=3, where={"company_id": "IBK"}
                        )

                        retriever.search.return_value = []
                        empty = await session.call_tool("search_esg_evidence", call)
                        assert not empty.isError
                        assert empty.structuredContent == {"company_id": "IBK", "evidence": []}

                        for invalid in (
                            {**call, "disclosure_item": "  "},
                            {**call, "limit": 6},
                        ):
                            error = await session.call_tool("search_esg_evidence", invalid)
                            assert error.isError, invalid
                        assert retriever.search.call_count == 2

    asyncio.run(exercise())

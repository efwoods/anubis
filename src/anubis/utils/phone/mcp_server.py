"""In-process Streamable HTTP MCP server for personal-avatar phone tools.

Mounted at ``POST /mcp/phone``. Authenticated with the owner's API key.
``think`` attaches the same tools as native LangChain callables and must
never HTTP-call this endpoint (that would deadlock the API process).
"""

from __future__ import annotations

from typing import Any

from src.anubis.utils.phone.places import lookup_local_place
from src.anubis.utils.phone.tools import (
    LOOKUP_TRAVEL_TOOL_NAMES,
    SIP_TOOL_NAMES,
    phone_record_from_accounts,
    phone_tools_as_mcp_descriptors,
    serialize_tool_result,
)
from src.anubis.utils.phone.travel import estimate_travel


def mcp_initialize_result() -> dict[str, Any]:
    """JSON-RPC ``initialize`` result."""
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "anubis-phone", "version": "1.0.0"},
    }


async def handle_phone_mcp_request(
    body: dict[str, Any],
    *,
    context: Any,
    store: Any,
    user_id: str,
    assistant_id: str,
    accounts: list[dict[str, Any]],
    is_personal_avatar: bool,
) -> dict[str, Any]:
    """Dispatch one JSON-RPC MCP message. SIP tools stay hidden until connected."""
    request_id = body.get("id")
    method = str(body.get("method") or "")
    params = body.get("params") if isinstance(body.get("params"), dict) else {}

    if not is_personal_avatar:
        return _error(request_id, -32000, "Phone tools belong to the personal avatar.")

    include_sip = phone_record_from_accounts(accounts) is not None

    if method == "initialize":
        return _result(request_id, mcp_initialize_result())
    if method == "notifications/initialized":
        return _result(request_id, {})
    if method == "tools/list":
        return _result(
            request_id, {"tools": phone_tools_as_mcp_descriptors(include_sip)}
        )
    if method == "tools/call":
        name = str(params.get("name") or "")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        if name in SIP_TOOL_NAMES and not include_sip:
            return _error(
                request_id,
                -32601,
                "SIP tools attach only after a Phone connection is verified.",
            )
        if name not in LOOKUP_TRAVEL_TOOL_NAMES and name not in SIP_TOOL_NAMES:
            return _error(request_id, -32601, f"Unknown tool {name!r}.")
        if name == "place_call":
            return _error(request_id, -32601, "Unknown tool 'place_call'.")
        try:
            payload = await _call_tool(
                name,
                arguments,
                context=context,
                store=store,
                user_id=user_id,
                assistant_id=assistant_id,
            )
        except Exception as call_error:
            return _error(request_id, -32000, str(call_error))
        return _result(
            request_id,
            {
                "content": [{"type": "text", "text": serialize_tool_result(payload)}],
                "isError": False,
            },
        )
    return _error(request_id, -32601, f"Unknown method {method!r}.")


async def _call_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    context: Any,
    store: Any,
    user_id: str,
    assistant_id: str,
) -> Any:
    if name == "lookup_local_place":
        return await lookup_local_place(
            str(arguments.get("name") or ""),
            str(arguments.get("city") or ""),
            context,
        )
    if name == "estimate_travel":
        return await estimate_travel(
            str(arguments.get("destination") or ""),
            context,
            origin=arguments.get("origin"),
            store=store,
            user_id=user_id,
            assistant_id=assistant_id,
        )
    if name in SIP_TOOL_NAMES:
        return {
            "status": "use_native_tool",
            "message": (
                "SIP tools run as native LangChain tools on the personal avatar "
                "turn. They are not invoked through this HTTP loop."
            ),
            "tool": name,
        }
    raise ValueError(f"Unknown tool {name!r}.")


def _result(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }

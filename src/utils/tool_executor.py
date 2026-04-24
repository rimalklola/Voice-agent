"""Tool execution with timeout handling and result validation."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# Tool response schemas for validation
TOOL_RESPONSE_SCHEMAS = {
    "ask_to_reserve": {
        "required_keys": ["status", "answer"],
        "status_values": ["ok", "missing_fields", "needs_confirmation", "duplicate", "invalid_fields", "rate_limited", "error"],
    },
    "get_project_info": {
        "required_keys": ["status"],
        "status_values": ["ok", "error"],
    },
    "get_property_specs": {
        "required_keys": ["status"],
        "status_values": ["ok", "error"],
    },
    "get_project_facts": {
        "required_keys": ["status"],
        "status_values": ["ok", "error"],
    },
    "end_call": {
        "required_keys": [],
    },
}


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""

    pass


class ToolTimeoutError(ToolExecutionError):
    """Raised when tool execution times out."""

    pass


class ToolValidationError(ToolExecutionError):
    """Raised when tool output validation fails."""

    pass


def validate_tool_output(tool_name: str, output: Any) -> bool:
    """
    Validate tool output matches expected schema.
    
    Args:
        tool_name: Name of the tool
        output: Output from the tool
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(output, dict):
        logger.error(
            "Tool output is not a dict",
            extra={"tool_name": tool_name, "output_type": type(output).__name__},
        )
        return False

    schema = TOOL_RESPONSE_SCHEMAS.get(tool_name)
    if schema is None:
        logger.debug(f"No validation schema for tool {tool_name}; skipping validation")
        return True

    # Check required keys
    for key in schema.get("required_keys", []):
        if key not in output:
            logger.error(
                f"Tool output missing required key '{key}'",
                extra={"tool_name": tool_name, "keys": list(output.keys())},
            )
            return False

    # Validate status value if applicable
    if "status_values" in schema and "status" in output:
        status = output.get("status")
        if status not in schema["status_values"]:
            logger.error(
                f"Tool status '{status}' not in allowed values",
                extra={
                    "tool_name": tool_name,
                    "status": status,
                    "allowed": schema["status_values"],
                },
            )
            return False

    return True


async def execute_tool_with_timeout(
    tool_func: Callable,
    tool_name: str,
    args: Dict[str, Any],
    timeout_s: float = 10.0,
    fallback_error_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute tool function with timeout handling.
    
    Args:
        tool_func: Async tool function to execute
        tool_name: Name of the tool (for logging/validation)
        args: Arguments to pass to the tool
        timeout_s: Timeout in seconds
        fallback_error_message: Message to return on timeout/error
        
    Returns:
        Tool output dict
    """
    if fallback_error_message is None:
        fallback_error_message = f"Le service {tool_name} a pris trop de temps à répondre. Réessayez."

    try:
        logger.debug(
            f"Executing tool with timeout",
            extra={"tool_name": tool_name, "timeout_s": timeout_s},
        )

        result = await asyncio.wait_for(
            tool_func(args),
            timeout=timeout_s,
        )

        # Validate output
        if not validate_tool_output(tool_name, result):
            logger.error(
                f"Tool {tool_name} returned invalid output",
                extra={"tool_name": tool_name, "output": str(result)[:200]},
            )
            return {
                "status": "error",
                "answer": "Une erreur interne s'est produite. Réessayez.",
            }

        return result

    except asyncio.TimeoutError:
        logger.error(
            f"Tool {tool_name} timed out",
            extra={"tool_name": tool_name, "timeout_s": timeout_s},
        )
        raise ToolTimeoutError(f"Tool {tool_name} exceeded {timeout_s}s timeout") from None

    except ToolValidationError as exc:
        logger.error(
            f"Tool validation error",
            extra={"tool_name": tool_name, "error": str(exc)},
        )
        raise

    except Exception as exc:
        logger.exception(
            f"Tool {tool_name} raised exception",
            extra={"tool_name": tool_name, "error_type": type(exc).__name__},
        )
        # Return graceful error response
        return {
            "status": "error",
            "answer": fallback_error_message,
        }


def get_tool_timeout_config(tool_name: str, default_timeout_s: float = 10.0) -> float:
    """
    Get timeout configuration for a specific tool.
    
    Args:
        tool_name: Name of the tool
        default_timeout_s: Default timeout if not configured
        
    Returns:
        Timeout in seconds
    """
    # Tool-specific timeouts (can be extended with config)
    tool_timeouts = {
        "ask_to_reserve": 8.0,  # CSV write is fast
        "get_project_info": 12.0,  # Retrieval can take time
        "get_property_specs": 8.0,  # YAML lookup is fast
        "get_project_facts": 8.0,  # YAML lookup is fast
        "end_call": 5.0,  # Hangup should be quick
    }

    return tool_timeouts.get(tool_name, default_timeout_s)

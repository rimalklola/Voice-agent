"""Enhanced health checks for production monitoring."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.utils.circuit_breaker import get_circuit_breaker_status

logger = logging.getLogger(__name__)


class HealthStatus:
    """Represents service health status."""

    def __init__(self):
        self.timestamp = datetime.now(timezone.utc)
        self.status = "healthy"  # healthy, degraded, unhealthy
        self.components: Dict[str, str] = {}
        self.errors: list[str] = []
        self.details: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON response."""
        return {
            "status": self.status,
            "timestamp": self.timestamp.isoformat(),
            "components": self.components,
            "circuit_breakers": self.details.get("circuit_breakers", {}),
            "errors": self.errors[:5] if self.errors else [],  # Top 5 errors
        }


async def check_gemini_connection(timeout_s: float = 2.0) -> tuple[bool, Optional[str]]:
    """
    Check Gemini API connectivity.

    Returns:
        (success: bool, error_message: str or None)
    """
    try:
        from google import genai
        from src.utils.config import config

        api_key = config.project.gemini_api_key
        if not api_key:
            return False, "GEMINI_API_KEY not configured"
        client = genai.Client(api_key=api_key)
        # Lightweight connectivity check via model listing
        await asyncio.wait_for(
            asyncio.to_thread(lambda: list(client.models.list())),
            timeout=timeout_s,
        )
        return True, None
    except asyncio.TimeoutError:
        return False, "Gemini API timeout"
    except Exception as exc:
        return False, f"Gemini connection failed: {str(exc)}"


async def check_lancedb_connection(timeout_s: float = 2.0) -> tuple[bool, Optional[str]]:
    """
    Check LanceDB connectivity and table access.
    
    Returns:
        (success: bool, error_message: str or None)
    """
    try:
        from src.retrieval.lancedb_client import connect_table

        # Try to connect and read from the table
        async def _connect():
            table = connect_table()
            if table is None:
                return False
            # Use metadata/row-count check; avoids requiring full-text indices.
            _ = await asyncio.to_thread(table.count_rows)
            return True

        success = await asyncio.wait_for(_connect(), timeout=timeout_s)
        return success, None

    except asyncio.TimeoutError:
        return False, "LanceDB timeout"
    except Exception as exc:
        return False, f"LanceDB connection failed: {str(exc)}"


async def check_config_validity() -> tuple[bool, Optional[str]]:
    """
    Check configuration is loaded and valid.

    Returns:
        (success: bool, error_message: str or None)
    """
    try:
        from src.utils.config import config

        if not config.project.gemini_api_key:
            return False, "GEMINI_API_KEY not configured"

        if not config.realtime.temperature:
            return False, "realtime.temperature not configured"

        return True, None

    except Exception as exc:
        return False, f"Config check failed: {str(exc)}"


async def perform_health_check(
    check_openai: bool = True,
    check_lancedb: bool = True,
    check_config: bool = True,
    timeout_s: float = 5.0,
) -> HealthStatus:
    """
    Perform comprehensive health check.
    
    Args:
        check_openai: Whether to check OpenAI connectivity
        check_lancedb: Whether to check LanceDB connectivity
        check_config: Whether to check configuration
        timeout_s: Overall timeout for all checks
        
    Returns:
        HealthStatus object
    """
    health = HealthStatus()

    checks = []

    if check_config:
        checks.append(
            _check_and_record("configuration", check_config_validity(), health)
        )

    if check_lancedb:
        checks.append(
            _check_and_record("lancedb", check_lancedb_connection(timeout_s=2.0), health)
        )

    if check_openai:
        checks.append(
            _check_and_record("gemini", check_gemini_connection(timeout_s=2.0), health)
        )

    # Run all checks concurrently with timeout
    try:
        await asyncio.wait_for(
            asyncio.gather(*checks, return_exceptions=True),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        health.status = "degraded"
        health.errors.append("Health check exceeded timeout")
        logger.warning("Health check timed out")

    # Add circuit breaker status
    try:
        health.details["circuit_breakers"] = await get_circuit_breaker_status()
    except Exception as exc:
        logger.warning("Failed to get circuit breaker status", exc_info=exc)

    # Determine overall health status
    if any(v == "unhealthy" for v in health.components.values()):
        if health.status != "unhealthy":
            health.status = "unhealthy"
    elif any(v == "degraded" for v in health.components.values()):
        if health.status != "unhealthy":
            health.status = "degraded"

    logger.info(
        "Health check completed",
        extra={"status": health.status, "components": health.components},
    )

    return health


async def _check_and_record(
    component_name: str,
    check_coro,
    health: HealthStatus,
) -> None:
    """Helper to run check and record result."""
    try:
        success, error_msg = await check_coro
        if success:
            health.components[component_name] = "healthy"
        else:
            health.components[component_name] = "unhealthy"
            if error_msg:
                health.errors.append(f"{component_name}: {error_msg}")
                logger.warning(
                    f"Health check failed for {component_name}",
                    extra={"component": component_name, "error": error_msg},
                )
    except Exception as exc:
        health.components[component_name] = "unhealthy"
        error_msg = f"{component_name}: {str(exc)}"
        health.errors.append(error_msg)
        logger.exception(
            f"Exception during health check for {component_name}",
            extra={"component": component_name},
        )

"""Circuit breaker pattern for handling persistent service failures gracefully."""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Service failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascading failures by stopping requests to a failing service
    and allowing recovery time.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_s: float = 60.0,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name (for logging)
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes in half-open state before closing
            timeout_s: Time to wait in open state before entering half-open
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_s = timeout_s

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Any exception raised by the function
        """
        async with self._lock:
            await self._check_state_transition()
            
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. Service unavailable."
                )

        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as exc:
            await self._record_failure()
            raise

    async def _check_state_transition(self) -> None:
        """Check and transition state if needed."""
        if self.state == CircuitState.OPEN:
            time_since_failure = time.monotonic() - self.last_failure_time
            if time_since_failure >= self.timeout_s:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(
                    f"Circuit breaker '{self.name}' transitioned to HALF_OPEN",
                    extra={"circuit_name": self.name},
                )

    async def _record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info(
                        f"Circuit breaker '{self.name}' closed (service recovered)",
                        extra={"circuit_name": self.name},
                    )
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0

    async def _record_failure(self) -> None:
        """Record a failed call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()

            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.error(
                        f"Circuit breaker '{self.name}' opened after {self.failure_count} failures",
                        extra={"circuit_name": self.name, "failure_count": self.failure_count},
                    )
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' re-opened (service still failing)",
                    extra={"circuit_name": self.name},
                )

    def get_state(self) -> dict:
        """Get current state for monitoring."""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "time_since_last_failure": time.monotonic() - self.last_failure_time,
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


# Global circuit breakers for key services
gemini_live_breaker = CircuitBreaker(
    name="gemini_live",
    failure_threshold=5,
    success_threshold=2,
    timeout_s=60.0,
)


async def get_circuit_breaker_status() -> dict:
    """Get status of all circuit breakers."""
    return {
        "gemini_live": gemini_live_breaker.get_state(),
    }

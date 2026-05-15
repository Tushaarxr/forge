"""Retry decorators for transient failure handling.

Provides exponential backoff retry for network operations.
"""

import asyncio
import functools
import logging
import time
from typing import Callable, TypeVar, Any

import httpx

from forge.exceptions import BrainError, WorkerError

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    retriable_exceptions: tuple = (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException),
) -> Callable:
    """Decorator that retries a function with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        retriable_exceptions: Tuple of exception types to retry on

    Returns:
        Decorated function

    Example:
        @retry_with_backoff(max_attempts=3, base_delay=2.0)
        async def call_api():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retriable_exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = min(base_delay * (exponential_base ** attempt), max_delay)
                        status_code = getattr(e, "response", None)

                        # Check for rate limit
                        if status_code and status_code.status_code == 429:
                            retry_after = status_code.headers.get("retry-after")
                            if retry_after:
                                try:
                                    delay = max(float(retry_after), delay)
                                except ValueError:
                                    pass

                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__}: "
                            f"{type(e).__name__}. Waiting {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")

            # All retries exhausted
            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retriable_exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = min(base_delay * (exponential_base ** attempt), max_delay)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__}: "
                            f"{type(e).__name__}. Waiting {delay:.1f}s"
                        )
                        time.sleep(delay)

            raise last_exception

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def retry_on_error(
    max_attempts: int = 3,
    delay: float = 1.0,
    error_types: tuple = (Exception,),
    backoff: bool = False,
) -> Callable:
    """Generic retry decorator with configurable options.

    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between retries (seconds)
        backoff: If True, double delay after each retry
        error_types: Tuple of exception types to catch

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except error_types as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__}: {e}"
                        )
                        await asyncio.sleep(current_delay)
                        if backoff:
                            current_delay *= 2

            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except error_types as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__}: {e}"
                        )
                        time.sleep(current_delay)
                        if backoff:
                            current_delay *= 2

            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
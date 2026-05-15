"""tests/test_retry.py — Unit tests for retry decorators."""

import asyncio
import pytest
import time
from unittest.mock import MagicMock, AsyncMock
import httpx

from forge.retry import retry_with_backoff, retry_on_error


class TestRetryWithBackoff:
    """Test retry_with_backoff decorator."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """Function succeeds on first attempt, no retries."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.1)
        async def succeed_once():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await succeed_once()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_http_status_error_429(self):
        """Retries on 429 status code."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.1)
        async def retry_on_429():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                mock_response = MagicMock()
                mock_response.status_code = 429
                mock_response.headers = {}
                raise httpx.HTTPStatusError("rate limited", request=MagicMock(), response=mock_response)
            return "success"

        result = await retry_on_429()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_connect_error(self):
        """Retries on connection error."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.1)
        async def retry_on_connect():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ConnectError("connection failed")
            return "success"

        result = await retry_on_connect()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """Retries on timeout exception."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.1)
        async def retry_on_timeout():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.TimeoutException("timeout")
            return "success"

        result = await retry_on_timeout()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_max_attempts_exhausted_raises(self):
        """Raises exception when all attempts exhausted."""
        @retry_with_backoff(max_attempts=2, base_delay=0.1)
        async def always_fails():
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.headers = {}
            raise httpx.HTTPStatusError("server error", request=MagicMock(), response=mock_response)

        with pytest.raises(httpx.HTTPStatusError):
            await always_fails()

    @pytest.mark.asyncio
    async def test_non_retriable_exception_not_retried(self):
        """Non-retriable exceptions are not retried."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.1)
        async def raise_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retriable")

        with pytest.raises(ValueError):
            await raise_value_error()
        assert call_count == 1  # Only called once, not retried

    @pytest.mark.asyncio
    async def test_retry_after_header_used_when_present(self):
        """Uses retry-after header when present."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=1.0)
        async def with_retry_after():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                mock_response = MagicMock()
                mock_response.status_code = 429
                mock_response.headers = {"retry-after": "0.05"}  # Very short delay for test
                raise httpx.HTTPStatusError("rate limited", request=MagicMock(), response=mock_response)
            return "success"

        start = time.time()
        result = await with_retry_after()
        elapsed = time.time() - start
        assert result == "success"
        assert call_count == 2
        # Should have waited at least 0.05 seconds
        assert elapsed >= 0.04


class TestRetryOnError:
    """Test retry_on_error decorator."""

    @pytest.mark.asyncio
    async def test_success_without_retry(self):
        """Succeeds without retry when no errors."""
        @retry_on_error(max_attempts=3, delay=0.1)
        async def succeed():
            return "ok"

        result = await succeed()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retry_on_specified_exception(self):
        """Retries on specified exception type."""
        call_count = 0

        @retry_on_error(max_attempts=3, delay=0.1, error_types=(ValueError,))
        async def retry_value_error():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("test error")
            return "recovered"

        result = await retry_value_error()
        assert result == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises(self):
        """Raises when max attempts exhausted."""
        @retry_on_error(max_attempts=2, delay=0.1, error_types=(ValueError,))
        async def always_fail():
            raise ValueError("persistent")

        with pytest.raises(ValueError):
            await always_fail()

    @pytest.mark.asyncio
    async def test_backoff_doubles_delay(self):
        """With backoff=True, delay doubles each retry."""
        call_count = 0

        @retry_on_error(max_attempts=3, delay=0.1, backoff=True)
        async def with_backoff():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("test")
            return "done"

        start = time.time()
        result = await with_backoff()
        elapsed = time.time() - start
        assert result == "done"
        # Approximate: 0.1 + 0.2 = 0.3 seconds minimum
        assert elapsed >= 0.25

    @pytest.mark.asyncio
    async def test_no_backoff_constant_delay(self):
        """Without backoff, delay stays constant."""
        call_count = 0

        @retry_on_error(max_attempts=3, delay=0.1, backoff=False)
        async def without_backoff():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("test")
            return "done"

        start = time.time()
        result = await without_backoff()
        elapsed = time.time() - start
        assert result == "done"
        # Should be roughly 0.2 seconds (2 retries * 0.1)
        assert elapsed < 0.4  # Much less than with backoff


class TestRetrySyncFunction:
    """Test retry decorator works with synchronous functions."""

    def test_sync_function_success(self):
        """Synchronous function succeeds without retry."""
        @retry_with_backoff(max_attempts=3, base_delay=0.1)
        def sync_success():
            return "ok"

        result = sync_success()
        assert result == "ok"

    def test_sync_function_retries(self):
        """Synchronous function retries on failure."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.1)
        def sync_retry():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ConnectError("test")
            return "recovered"

        result = sync_retry()
        assert result == "recovered"
        assert call_count == 2

    def test_sync_exhausted_raises(self):
        """Synchronous function raises when exhausted."""
        @retry_with_backoff(max_attempts=2, base_delay=0.1)
        def sync_fail():
            raise httpx.ConnectError("test")

        with pytest.raises(httpx.ConnectError):
            sync_fail()


class TestRetryWithDifferentExceptions:
    """Test retry with different exception types."""

    @pytest.mark.asyncio
    async def test_handles_500_error(self):
        """Handles 500 internal server error."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.1)
        async def handle_500():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                mock_response = MagicMock()
                mock_response.status_code = 500
                mock_response.headers = {}
                raise httpx.HTTPStatusError("server error", request=MagicMock(), response=mock_response)
            return "recovered"

        result = await handle_500()
        assert result == "recovered"

    @pytest.mark.asyncio
    async def test_handles_503_error(self):
        """Handles 503 service unavailable."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.1)
        async def handle_503():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                mock_response = MagicMock()
                mock_response.status_code = 503
                mock_response.headers = {}
                raise httpx.HTTPStatusError("unavailable", request=MagicMock(), response=mock_response)
            return "recovered"

        result = await handle_503()
        assert result == "recovered"
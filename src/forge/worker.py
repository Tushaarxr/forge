"""Worker class for LM Studio local execution."""

import logging
import os
import re
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")


class Worker:
    """Local LLM worker using LM Studio (OpenAI-compatible API at localhost:1234)."""

    SYSTEM_PROMPT = (
        "You are a precise coding assistant. You receive a task and relevant context.\n"
        "Output ONLY the requested code or changes. Use this exact format for file changes:\n\n"
        "<<<FILE: path/to/file.py>>>\n"
        "<complete file content>\n"
        "<<<END FILE>>>\n\n"
        "If multiple files change, use multiple FILE blocks.\n"
        "Never explain unless asked. Never add placeholder comments like '# TODO: implement'."
    )

    def __init__(self) -> None:
        """Initialize the Worker with LM Studio configuration."""
        self.base_url = LM_STUDIO_BASE_URL.rstrip("/")
        self.model = os.getenv("LOCAL_MODEL", "qwen3.5-9b-instruct")
        self.temperature = float(os.getenv("LOCAL_TEMPERATURE", "0.6"))
        self.top_p = float(os.getenv("LOCAL_TOP_P", "0.95"))
        self.ctx_size = int(os.getenv("LOCAL_CTX_SIZE", "8192"))
        self.max_tokens = int(os.getenv("LOCAL_MAX_TOKENS", "2048"))
        self._client: httpx.AsyncClient | None = None
        self._started_at: float = 0.0

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client if open."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def execute(
        self, task: str, context: str, active_file: str | None = None
    ) -> dict[str, Any]:
        """Execute a coding task via LM Studio.

        Args:
            task: The coding task description
            context: Relevant file/project context
            active_file: Optional currently active file path

        Returns:
            dict with raw_response, file_changes, tokens_used, elapsed_seconds, error (optional)
        """
        self._started_at = time.monotonic()

        if active_file:
            user_content = f"Context:\n{context}\n\nTask for {active_file}:\n{task}"
        else:
            user_content = f"Context:\n{context}\n\nTask:\n{task}"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        est_tokens = self.estimate_context_tokens(context, task)
        if est_tokens > 6000:
            logger.warning(f"Estimated context tokens ({est_tokens}) may be too large")

        try:
            response = await self.client.post(
                "/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_tokens": self.max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()

            raw_response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            file_changes = self.parse_file_blocks(raw_response)
            elapsed = time.monotonic() - self._started_at
            tokens_used = data.get("usage", {}).get("completion_tokens", 0)

            return {
                "raw_response": raw_response,
                "file_changes": file_changes,
                "tokens_used": tokens_used,
                "elapsed_seconds": elapsed,
                "tokens_per_second": tokens_used / max(elapsed, 1e-9),
            }

        except httpx.ConnectError:
            msg = (
                f"LM Studio not reachable at {self.base_url} — "
                "start the server and load a model."
            )
            logger.error(msg)
            return {
                "raw_response": "",
                "file_changes": [],
                "tokens_used": 0,
                "elapsed_seconds": time.monotonic() - self._started_at,
                "tokens_per_second": 0.0,
                "error": msg,
            }

        except httpx.HTTPStatusError as e:
            msg = f"HTTP {e.response.status_code}: {e.response.text[:500]}"
            logger.error(f"HTTP error: {msg}")
            return {
                "raw_response": "",
                "file_changes": [],
                "tokens_used": 0,
                "elapsed_seconds": time.monotonic() - self._started_at,
                "tokens_per_second": 0.0,
                "error": msg,
            }

        except httpx.TimeoutException:
            logger.error("Request timed out")
            return {
                "raw_response": "",
                "file_changes": [],
                "tokens_used": 0,
                "elapsed_seconds": time.monotonic() - self._started_at,
                "tokens_per_second": 0.0,
                "error": "Request timed out",
            }

        except Exception as e:
            logger.exception("Unexpected error during execution")
            return {
                "raw_response": "",
                "file_changes": [],
                "tokens_used": 0,
                "elapsed_seconds": time.monotonic() - self._started_at,
                "tokens_per_second": 0.0,
                "error": str(e),
            }

    async def health_check(self) -> dict[str, Any]:
        """Check if LM Studio is running and list available models.

        Returns:
            dict with ok (bool), models (list), and optional error string
        """
        try:
            response = await self.client.get("/v1/models")
            response.raise_for_status()
            data = response.json()
            models = [m.get("id", "") for m in data.get("data", [])]
            return {"ok": True, "models": models}

        except httpx.ConnectError:
            msg = (
                f"LM Studio not reachable at {self.base_url} — "
                "start the server and load a model."
            )
            return {"ok": False, "models": [], "error": msg}

        except httpx.HTTPStatusError as e:
            return {"ok": False, "models": [], "error": f"HTTP {e.response.status_code}"}

        except Exception as e:
            logger.exception("Health check failed")
            return {"ok": False, "models": [], "error": str(e)}

    def parse_file_blocks(self, response: str) -> list[dict[str, str]]:
        """Extract <<<FILE: path>>> ... <<<END FILE>>> blocks from LLM response.

        Returns:
            List of {"file_path": ..., "content": ...} dicts
        """
        pattern = r"<<<FILE:\s*([^>]+)>>>(.*?)<<<END FILE>>>"
        matches = re.findall(pattern, response, re.DOTALL)

        file_changes = [
            {"file_path": fp.strip(), "content": content.strip("\n")}
            for fp, content in matches
        ]

        if not file_changes and "<<<FILE:" in response:
            logger.warning("Response has FILE blocks but missing END FILE markers — attempting fallback parse")
            fallback = r"<<<FILE:\s*([^>]+)>>>(.*?)(?=<<<|$)"
            matches = re.findall(fallback, response, re.DOTALL)
            file_changes = [
                {"file_path": fp.strip(), "content": content.strip("\n")}
                for fp, content in matches
            ]

        return file_changes

    def estimate_context_tokens(self, context: str, task: str) -> int:
        """Rough estimate of token count (chars / 4)."""
        total_chars = len(self.SYSTEM_PROMPT) + len(context) + len(task)
        return total_chars // 4

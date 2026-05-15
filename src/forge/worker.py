"""Worker class for LM Studio local execution."""

import logging
import os
import re
import time
import typing
from typing import Any

import httpx

from forge.exceptions import WorkerError
from forge.retry import retry_with_backoff

logger = logging.getLogger(__name__)

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")


class Worker:
    """Local LLM worker using LM Studio (OpenAI-compatible API at localhost:1234)."""

    SYSTEM_PROMPT = (
        "You are a master software engineer and UI/UX designer. You receive a task and "
        "the current state of relevant files.\n\n"
        "DESIGN MANDATE:\n"
        "- When building web interfaces, aim for 'Premium Visual Excellence'.\n"
        "- Use modern, harmonious color palettes (e.g., sleek dark modes, vibrant gradients).\n"
        "- Use professional typography (e.g., Inter, Roboto, Outfit) via Google Fonts.\n"
        "- Ensure mobile-responsive layouts and high-quality spacing/padding.\n"
        "- Avoid 'basic' or 'MVP' looking designs. WOW the user with aesthetics.\n\n"
        "OUTPUT FORMATS:\n"
        "Output ONLY the requested changes using one of these two formats:\n\n"
        "FORMAT 1 — Create or fully rewrite a file:\n"
        "<<<FILE: path/to/file>>>\n"
        "<complete file content>\n"
        "<<<END FILE>>>\n\n"
        "FORMAT 2 — Small patch to an existing file (under 30 lines changed):\n"
        "<<<PATCH: path/to/file>>>\n"
        "<<<FIND>>>\n"
        "<exact existing lines to replace — must match file exactly>\n"
        "<<<REPLACE>>>\n"
        "<new lines to put in their place>\n"
        "<<<END PATCH>>>\n\n"
        "Rules:\n"
        "- Use FORMAT 1 when creating a new file or making large structural changes.\n"
        "- Use FORMAT 2 when making small, targeted changes to an existing file.\n"
        "- NEVER truncate output. NEVER use '...' or '# rest of file'. Always complete.\n"
        "- Never explain unless asked. Never add placeholder comments like '# TODO'."
    )

    def __init__(self) -> None:
        """Initialize the Worker with LM Studio configuration."""
        self.base_url = LM_STUDIO_BASE_URL.rstrip("/")
        self.model = os.getenv("LOCAL_MODEL", "qwen3.5-9b-instruct")
        self.temperature = float(os.getenv("LOCAL_TEMPERATURE", "0.6"))
        self.top_p = float(os.getenv("LOCAL_TOP_P", "0.95"))
        self.ctx_size = int(os.getenv("LOCAL_CTX_SIZE", "8192"))
        # FIX 1: Increased default from 2048 → 4096 to prevent mid-file truncation.
        # A growing HTML/Python file easily exceeds 2048 tokens when the worker must
        # emit the entire file inside a FILE block.
        self.max_tokens = int(os.getenv("LOCAL_MAX_TOKENS", "4096"))
        self._client: httpx.AsyncClient | None = None
        self._started_at: float = 0.0

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            # Use connection pooling for better performance
            limits = httpx.Limits(
                max_keepalive_connections=3,
                max_connections=5,
                keepalive_expiry=30.0,
            )
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(120.0, connect=10.0),
                limits=limits,
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client if open."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def execute(
        self, task: str, context: str, active_file: str | None = None,
        stream_callback: typing.Callable[[str], None] | None = None
    ) -> dict[str, Any]:
        """Execute a coding task via LM Studio.

        Args:
            task: The coding task description
            context: Relevant file/project context
            active_file: Optional currently active file path

        Returns:
            dict with raw_response, file_changes, patch_changes, tokens_used,
            elapsed_seconds, error (optional)
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
            request_data = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            }

            raw_response = ""
            tokens_used = 0

            if stream_callback:
                request_data["stream"] = True
                async with self.client.stream(
                    "POST", "/v1/chat/completions", json=request_data
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            import json
                            try:
                                chunk = json.loads(line[6:])
                                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if delta:
                                    raw_response += delta
                                    stream_callback(delta)
                            except Exception:
                                pass
                tokens_used = len(raw_response) // 4
            else:
                response = await self.client.post(
                    "/v1/chat/completions",
                    json=request_data,
                )
                response.raise_for_status()
                data = response.json()
                raw_response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                tokens_used = data.get("usage", {}).get("completion_tokens", 0)

            file_changes = self.parse_file_blocks(raw_response)
            patch_changes = self.parse_patch_blocks(raw_response)
            elapsed = time.monotonic() - self._started_at

            return {
                "raw_response": raw_response,
                "file_changes": file_changes,
                "patch_changes": patch_changes,
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
                "patch_changes": [],
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
                "patch_changes": [],
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
                "patch_changes": [],
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
                "patch_changes": [],
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

    def parse_patch_blocks(self, response: str) -> list[dict[str, str]]:
        """Extract <<<PATCH: path>>> <<<FIND>>> ... <<<REPLACE>>> ... <<<END PATCH>>> blocks.

        FIX 7: Patch format lets the worker emit only changed sections of large files,
        avoiding the need to regenerate the entire file and staying within token limits.

        Returns:
            List of {"file_path": ..., "find": ..., "replace": ...} dicts
        """
        patches: list[dict[str, str]] = []
        pattern = r"<<<PATCH:\s*(.+?)>>>\s*<<<FIND>>>(.*?)<<<REPLACE>>>(.*?)<<<END PATCH>>>"
        for match in re.finditer(pattern, response, re.DOTALL):
            patches.append({
                "file_path": match.group(1).strip(),
                "find": match.group(2).strip(),
                "replace": match.group(3).strip(),
            })
        return patches

    def apply_patch(self, file_path: str, find: str, replace: str) -> bool:
        """Apply a find/replace patch to a file in-place.

        FIX 7: Applies surgical edits without rewriting the entire file.
        Uses flexible whitespace matching to handle minor formatting differences.

        Args:
            file_path: Path to the file to patch
            find: Exact string to find in the file
            replace: String to substitute in

        Returns:
            True if patch applied successfully, False if find string not present
            (patch mismatch — caller should fall back to full-file context on retry).
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Try flexible matching first
            if self._patches_match(find, content):
                # Find the exact position using original find string
                idx = content.find(find)
                if idx >= 0:
                    new_content = content[:idx] + replace + content[idx + len(find):]
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    logger.info(f"Applied patch to {file_path}")
                    return True

            # Fall back to exact match
            if find in content:
                new_content = content.replace(find, replace, 1)  # first occurrence only
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                logger.info(f"Applied patch to {file_path}")
                return True

            logger.warning(
                f"Patch mismatch for {file_path} — FIND string not found in file. "
                "Worker will receive full file content on next retry."
            )
            return False
        except Exception as e:
            logger.error(f"Patch apply failed for {file_path}: {e}")
            return False

    def estimate_context_tokens(self, context: str, task: str) -> int:
        """Rough estimate of token count (chars / 4)."""
        total_chars = len(self.SYSTEM_PROMPT) + len(context) + len(task)
        return total_chars // 4

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace for patch matching.

        Replaces multiple whitespace with single space, removes leading/trailing.
        This makes patch matching more robust against minor whitespace differences.
        """
        return re.sub(r'\s+', ' ', text).strip()

    def _patches_match(self, find: str, content: str) -> bool:
        """Check if find string exists in content, with flexible whitespace matching."""
        # First try exact match
        if find in content:
            return True

        # Try normalized match
        normalized_find = self._normalize_whitespace(find)
        normalized_content = self._normalize_whitespace(content)

        # Check if normalized find is in normalized content
        # We need to check if each line in find appears in content (order preserved)
        find_lines = [l for l in normalized_find.split('\n') if l.strip()]
        if not find_lines:
            return False

        content_lines = normalized_content.split('\n')
        idx = 0
        for find_line in find_lines:
            found = False
            while idx < len(content_lines):
                if find_line in content_lines[idx]:
                    found = True
                    idx = idx + 1
                    break
                idx += 1
            if not found:
                return False
        return True

    async def self_review(self, task: str, response: str) -> dict[str, Any]:
        """Verify if the generated response satisfies the task.
        
        Simple heuristic check for now.
        """
        if not response:
            return {"passed": False, "reason": "Empty response"}
        
        # Check if response has any of our markers
        if "<<<FILE:" in response or "<<<PATCH:" in response:
            return {"passed": True}
            
        # If it's a very short response without markers, it might have failed
        if len(response) < 20:
            return {"passed": False, "reason": "Response too short and missing markers"}
            
        return {"passed": True}

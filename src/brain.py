"""Brain class for forge coding agent - Gemini 2.0 Flash master brain."""

import json
import logging
import os
import sys
from typing import Any

import httpx


logger = logging.getLogger(__name__)

# ── Provider configuration ─────────────────────────────────────────────────────
MASTER_PROVIDER = os.getenv("MASTER_PROVIDER", "gemini")  # gemini | anthropic
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MASTER_MODEL = os.getenv("MASTER_MODEL", "gemini-2.0-flash")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _require_gemini_key() -> str:
    """Return GEMINI_API_KEY or exit with a clear error message."""
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        print(
            "\n[ERROR] GEMINI_API_KEY is not set.\n"
            "  Set it with:  set GEMINI_API_KEY=<your_key>  (Windows)\n"
            "  or create a .env file in your project's .forge/ directory.\n"
        )
        sys.exit(1)
    return key


class Brain:
    """Core orchestration brain — defaults to Gemini 2.0 Flash (free tier)."""

    def __init__(self) -> None:
        """Initialize the Brain with configuration."""
        self._provider = MASTER_PROVIDER.lower()
        self._model = MASTER_MODEL
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Lazy-initialize the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=90.0)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Strip markdown fences, parse JSON, return fallback dict on error."""
        text = text.strip()
        # Strip ```json ... ``` fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            start = 1 if lines[0].startswith("```") else 0
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            text = "\n".join(lines[start:end]).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}. Raw text: {text[:200]}")
            return {"reasoning": text, "error": str(e)}

    async def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        """Make a Gemini API call and return the text response with fallback support."""
        api_key = _require_gemini_key()
        client = await self._ensure_client()
        models = [m.strip() for m in self._model.split(",") if m.strip()]
        
        last_error = None
        for model in models:
            url = f"{GEMINI_BASE_URL}/{model}:generateContent"
            payload = {
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "generationConfig": {"temperature": 0.2},
            }
            try:
                response = await client.post(
                    url,
                    headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )
            except httpx.HTTPStatusError as e:
                last_error = e
                # 429 is Too Many Requests (Quota Exceeded)
                if e.response.status_code == 429:
                    logger.warning(f"Quota exceeded for model {model}, trying next if available...")
                    continue
                # If it's a 404, maybe the model doesn't exist, try next
                elif e.response.status_code == 404:
                    logger.warning(f"Model {model} not found (404), trying next...")
                    continue
                # If it's another error, we probably shouldn't blindly retry on other models, but let's just fail
                raise
            except Exception as e:
                last_error = e
                logger.warning(f"Request failed for model {model}: {e}, trying next...")
                continue
                
        if last_error:
            raise last_error
        raise RuntimeError("No models available to try")

    async def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Make an Anthropic API call and return the text response with fallback support."""
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        client = await self._ensure_client()
        models = [m.strip() for m in self._model.split(",") if m.strip()]
        
        last_error = None
        for model in models:
            try:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": key,
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": model,
                        "max_tokens": 4096,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": user_prompt}],
                    },
                )
                response.raise_for_status()
                return response.json().get("content", [{}])[0].get("text", "")
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429 or e.response.status_code == 404:
                    logger.warning(f"Model {model} failed with {e.response.status_code}, trying next...")
                    continue
                raise
            except Exception as e:
                last_error = e
                logger.warning(f"Model {model} request failed: {e}, trying next...")
                continue
                
        if last_error:
            raise last_error
        raise RuntimeError("No models available to try")

    async def _call(self, system_prompt: str, user_prompt: str) -> str:
        """Dispatch to the configured provider."""
        if self._provider == "gemini":
            return await self._call_gemini(system_prompt, user_prompt)
        elif self._provider == "anthropic":
            return await self._call_anthropic(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown provider: {self._provider}")

    async def plan(
        self,
        goal: str,
        context: str,
        project_summary: dict[str, Any] | str,
    ) -> dict[str, Any]:
        """Plan task into sub-tasks.

        Returns dict with: reasoning, sub_tasks, affected_files, risks
        """
        system_prompt = (
            "You are a senior software architect. Your worker is a local 9B model "
            "with an 8K context window. Break tasks into small, single-file focused "
            "sub-tasks. Each sub-task must be completable in one model call. "
            "Respond in JSON only with no markdown fences."
        )

        if isinstance(project_summary, dict):
            summary_str = json.dumps(project_summary, indent=2)
        else:
            summary_str = str(project_summary)

        user_prompt = (
            f"Goal: {goal}\n\n"
            f"Project Context:\n{context}\n\n"
            f"Project Summary:\n{summary_str}\n\n"
            "Plan the work into ordered sub-tasks. Return JSON with:\n"
            "- reasoning: your thinking process\n"
            "- sub_tasks: list of {id, description, active_file, needs_context, estimated_lines}\n"
            "- affected_files: files likely to change\n"
            "- risks: things to watch out for"
        )

        try:
            result_text = await self._call(system_prompt, user_prompt)
            return self._parse_json(result_text)
        except httpx.HTTPStatusError as e:
            logger.error(f"API error during planning ({e.response.status_code}): {e.response.text[:300]}")
            return {"reasoning": "API error", "sub_tasks": [], "affected_files": [], "risks": []}
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return {"reasoning": f"Planning failed: {e}", "sub_tasks": [], "affected_files": [], "risks": []}

    async def review(
        self,
        task: str,
        output: str,
        changed_files: list[str],
    ) -> dict[str, Any]:
        """Review completed work.

        Returns dict with: passed, score, issues, retry_prompt, learnings
        """
        system_prompt = (
            "You are a senior code reviewer. Check for correctness, security issues, "
            "and whether the task was actually completed. Be specific. "
            "Respond in JSON only with no markdown fences."
        )

        user_prompt = (
            f"Task: {task}\n\n"
            f"Output:\n{output[:3000]}\n\n"
            f"Changed Files: {', '.join(changed_files)}\n\n"
            "Review the work. Return JSON with:\n"
            "- passed: bool indicating if task is complete\n"
            "- score: int 0-10\n"
            "- issues: list of specific problems\n"
            "- retry_prompt: improved prompt if not passed\n"
            "- learnings: things to store in context"
        )

        try:
            result_text = await self._call(system_prompt, user_prompt)
            return self._parse_json(result_text)
        except httpx.HTTPStatusError as e:
            logger.error(f"API error during review: {e.response.status_code}")
            return {"passed": False, "score": 5, "issues": ["API unavailable"], "retry_prompt": task, "learnings": []}
        except Exception as e:
            logger.error(f"Review failed: {e}")
            return {"passed": False, "score": 0, "issues": [str(e)], "retry_prompt": task, "learnings": []}

    async def summarise(
        self,
        changes: list[dict[str, Any]],
        project_summary: dict[str, Any],
    ) -> dict[str, Any]:
        """Summarize changes made.

        Returns dict with: summary, key_decisions, next_suggested, risk_flags
        """
        system_prompt = (
            "You are a technical writer. Summarise what changed and why. Write for a "
            "developer who will read this in a git log. "
            "Respond in JSON only with no markdown fences."
        )

        user_prompt = (
            f"Changes:\n{json.dumps(changes, indent=2)}\n\n"
            f"Project Summary:\n{json.dumps(project_summary, indent=2)}\n\n"
            "Summarize the changes. Return JSON with:\n"
            "- summary: human-readable paragraph\n"
            "- key_decisions: architectural decisions made\n"
            "- next_suggested: what to do next\n"
            "- risk_flags: anything the human should review"
        )

        try:
            result_text = await self._call(system_prompt, user_prompt)
            return self._parse_json(result_text)
        except httpx.HTTPStatusError as e:
            logger.error(f"API error during summarization: {e.response.status_code}")
            return {"summary": "Summarization failed", "key_decisions": [], "next_suggested": [], "risk_flags": []}
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return {"summary": f"Failed: {e}", "key_decisions": [], "next_suggested": [], "risk_flags": []}

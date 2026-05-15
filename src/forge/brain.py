"""Brain class for forge coding agent - Gemini 2.5 Flash master brain."""

import asyncio
import json
import logging
import os
import sys
from typing import Any

import httpx

from forge.exceptions import BrainError
from forge.retry import retry_with_backoff


logger = logging.getLogger(__name__)

# ── Provider configuration ─────────────────────────────────────────────────────
MASTER_PROVIDER = os.getenv("MASTER_PROVIDER", "gemini")  # gemini | anthropic
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MASTER_MODEL = os.getenv("MASTER_MODEL", "gemini-2.5-flash")
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
        """Lazy-initialize the HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            # Use connection pooling for better performance
            limits = httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0,
            )
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(90.0, connect=30.0),
                limits=limits,
            )
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
        for attempt in range(3):
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
                    if e.response.status_code == 429:
                        if model != models[-1]:
                            logger.warning(f"Quota exceeded for {model}, trying next fallback...")
                            continue  # Try next model immediately
                        else:
                            # Last model in list, wait before next attempt cycle
                            wait_time = (attempt + 1) * 3
                            logger.warning(f"Quota exceeded for all models. Waiting {wait_time}s before next attempt cycle...")
                            await asyncio.sleep(wait_time)
                    elif e.response.status_code == 404:
                        logger.warning(f"Model {model} not found (404), trying next fallback...")
                        continue
                    else:
                        raise
                except Exception as e:
                    last_error = e
                    logger.warning(f"Request failed for {model}: {e}, trying next fallback...")
                    continue
        
        if last_error:
            raise last_error
        raise RuntimeError("No models available to try")

    async def test_connection(self) -> tuple[bool, str]:
        """Test API connectivity and basic quota. Returns (success, message)."""
        try:
            # Try a tiny request
            res = await self._call("You are a connectivity tester.", "Reply with 'OK'")
            if "OK" in res:
                return True, "Connected successfully"
            return False, f"Unexpected response: {res[:50]}"
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                return False, "Rate limit hit (429). You likely exceeded the 'Requests Per Minute' (RPM) limit, even if you have daily quota left. Using the 'Flash Trio' fallbacks will help avoid this."
            return False, f"API error ({e.response.status_code}): {e.response.text[:100]}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    async def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Make an Anthropic API call and return the text response with fallback support."""
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        client = await self._ensure_client()
        models = [m.strip() for m in self._model.split(",") if m.strip()]
        
        last_error = None
        for model in models:
            for attempt in range(3): # Try each model up to 3 times on 429
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
                    data = response.json()
                    return data.get("content", [{}])[0].get("text", "")
                except httpx.HTTPStatusError as e:
                    last_error = e
                    if e.response.status_code == 429:
                        wait_time = (attempt + 1) * 3
                        logger.warning(f"Anthropic rate limit for {model} (Attempt {attempt+1}/3). Waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    raise
                except Exception as e:
                    last_error = e
                    logger.warning(f"Anthropic request failed for {model}: {e}, trying next model...")
                    break
                
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
            "with an 8K context window. Break tasks into meaningful, file-focused "
            "sub-tasks. For web projects, prioritize 'Visual Excellence' and 'Premium Design'—"
            "avoid fragmenting UI components that should be built together for consistency.\n"
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

    async def plan_full_project(
        self,
        goal: str,
        project_summary: dict[str, Any] | str,
    ) -> list[dict[str, Any]]:
        """Generate a complete ordered task list for building a full application.

        Distinct from plan() which plans a single task. This method generates
        the entire roadmap from empty folder to working, runnable application.

        Returns:
            List of task dicts with: id, description, active_file, estimated_lines
        """
        system_prompt = (
            "You are a master project planning architect. Your goal is to generate a task list that is "
            "optimised for both quality and execution speed.\n\n"
            "ADAPTIVE TASK DENSITY:\n"
            "- For SMALL projects (landing pages, single-file scripts, simple utilities): DO NOT over-chunk. "
            "Prefer 1-3 comprehensive tasks. Fragmenting a single-page build into 15 tasks reduces quality.\n"
            "- For LARGE projects (multi-file applications, full-stack apps): Use granular chunking (10-30 tasks) "
            "to stay within the worker's context and output limits.\n\n"
            "GUIDELINES:\n"
            "- Order tasks by dependency (scaffold -> model -> logic -> UI).\n"
            "- Each task must be: completable in a single model call, scoped to 1-2 files maximum.\n"
            "- For UI/Web tasks: Group related sections together (e.g., 'Build full landing page structure and styling') "
            "rather than splitting header/footer/hero into separate tasks. This ensures design consistency.\n"
            "- Respond in JSON only with no markdown fences."
        )

        if isinstance(project_summary, dict):
            summary_str = json.dumps(project_summary, indent=2)
        else:
            summary_str = str(project_summary)

        user_prompt = (
            f"Goal: {goal}\n\n"
            f"Current Project State:\n{summary_str}\n\n"
            "Generate the complete ordered task list to build this application from scratch.\n"
            "Return JSON with a single key 'tasks' containing a list of objects, each with:\n"
            "- id: integer (1-based)\n"
            "- description: clear, actionable task description\n"
            "- active_file: primary file this task creates/modifies (string or null)\n"
            "- estimated_lines: rough estimate of new lines of code\n"
            "- category: one of: scaffold | model | logic | route | test | config\n"
            "- depends_on: list of task IDs that must complete successfully before this "
            "task can run (empty list [] if no dependencies). Use this for tasks that "
            "build directly on a prior task's output.\n\n"
            "Ensure tasks are ordered so each one builds on completed prior tasks."
        )

        try:
            result_text = await self._call(system_prompt, user_prompt)
            parsed = self._parse_json(result_text)
            tasks = parsed.get("tasks", [])
            if not tasks and isinstance(parsed, list):
                tasks = parsed
            # Ensure sequential IDs
            for i, t in enumerate(tasks, 1):
                t["id"] = i
            return tasks
        except httpx.HTTPStatusError as e:
            logger.error(f"API error during full project planning ({e.response.status_code}): {e.response.text[:300]}")
            return []
        except Exception as e:
            logger.error(f"Full project planning failed: {e}")
            raise

    async def refine_text_plan(self, text_plan: str, goal: str, project_summary: dict[str, Any]) -> list[dict[str, Any]]:
        """Refine a raw list of tasks into structured AutoTask dictionaries."""
        system_prompt = (
            "You are a project planning architect. Convert the raw task list into a structured JSON list.\n\n"
            "CRITICAL: Be 'Scale-Aware'. If the raw list is excessively granular for a simple project "
            "(e.g., 20 tasks for a single-page site), consolidate them into fewer, more meaningful tasks. "
            "For EACH task provided, you must provide:\n"
            "- description: clear, actionable task description\n"
            "- active_file: the primary file this task creates/modifies (or null if unknown)\n"
            "- estimated_lines: rough estimate of new lines of code (integer)\n"
            "- category: one of: scaffold | model | logic | route | test | config\n"
            "- depends_on: list of task IDs (1-indexed) that must complete first\n\n"
            "Return only a JSON object with a 'tasks' key containing the list."
        )
        user_prompt = (
            f"Project Goal: {goal}\n\n"
            f"Project Context: {json.dumps(project_summary)}\n\n"
            f"User's Raw Task List:\n{text_plan}\n\n"
            "Convert these into a structured task list."
        )

        try:
            result_text = await self._call(system_prompt, user_prompt)
            parsed = self._parse_json(result_text)
            tasks = parsed.get("tasks", [])
            if not tasks and isinstance(parsed, list):
                tasks = parsed
            # Ensure sequential IDs
            for i, t in enumerate(tasks, 1):
                t["id"] = i
            return tasks
        except Exception as e:
            logger.error(f"Failed to refine text plan: {e}")
            # Fallback: just return a basic logic task for each line if Gemini fails
            lines = [l.strip() for l in text_plan.splitlines() if l.strip()]
            return [{"id": i+1, "description": l, "category": "logic", "estimated_lines": 0} for i, l in enumerate(lines)]

    async def review(
        self,
        task: str,
        output: str,
        changed_files: list[str],
        file_contents: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Review completed work.

        FIX 3: Accepts file_contents — actual file state read from disk after the
        worker wrote it. When present, Gemini uses this as ground truth instead of
        the potentially-truncated raw LLM output.

        Returns dict with: passed, score, issues, retry_prompt, learnings
        """
        system_prompt = (
            "You are a senior code reviewer. Check for correctness, security issues, "
            "and whether the task was actually completed. Be specific. "
            "When both raw worker output and actual file contents are provided, evaluate "
            "the FILE CONTENTS as the ground truth — the raw output may be truncated "
            "but the written file may be complete and correct. "
            "Respond in JSON only with no markdown fences."
        )

        user_prompt = (
            f"Task: {task}\n\n"
            f"Worker raw output (may be truncated):\n{output[:2000]}\n\n"
            f"Changed Files: {', '.join(changed_files)}\n\n"
        )

        # FIX 3: Include actual file content from disk as ground truth
        if file_contents:
            user_prompt += "ACTUAL FILE STATE ON DISK (use this as ground truth):\n"
            for path, content in file_contents.items():
                user_prompt += f"\n--- {path} ---\n{content}\n"
            user_prompt += "\n"

        user_prompt += (
            "Review the work. Return JSON with:\n"
            "- passed: bool indicating if task is complete\n"
            "- score: int 0-10\n"
            "- issues: list of specific problems found\n"
            "- retry_prompt: improved prompt describing exactly what to fix, if not passed\n"
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

    def _build_batch_prompt(self, batch: list[dict], goal: str) -> str:
        """
        Compress batch into a minimal token prompt.
        
        Target: under 2000 tokens for a 5-task batch.
        
        Format:
        GOAL: <goal in one line>
        
        TASKS COMPLETED:
        [1] <task description, max 80 chars>
        FILES: file1.py (+12/-3)  file2.py (+5/-0)
        DIFF:
        <unified diff, max 30 lines per task>
        
        [2] <task description>
        FILES: ...
        DIFF:
        ...
        
        REVIEW EACH TASK. Return JSON:
        {
          "batch_passed": true/false,
          "tasks": [
            {"id": 1, "ok": true, "score": 8, "issue": null},
            {"id": 2, "ok": false, "score": 4, "issue": "unclosed div at line 34"},
            ...
          ],
          "critical_blockers": [],
          "batch_summary": "one sentence max",
          "retry_tasks": [2],
          "learnings": ["key decision 1", "key decision 2"]
        }
        
        Rules:
        - "issue" must be under 120 chars. One issue only — the most critical.
        - "learnings" max 3 items, each under 100 chars.
        - "batch_summary" max 120 chars.
        - If ok=true, issue must be null.
        """
        lines = [f"GOAL: {goal[:100]}"]
        lines.append("")
        lines.append("TASKS COMPLETED:")
        
        for item in batch:
            lines.append(f"\n[{item['id']}] {item['task'][:80]}")
            
            # File change stats: show +added/-removed line counts compactly
            file_stats = []
            for fp in item.get("files_changed", []):
                diff = item.get("diffs", {}).get(fp, "")
                added = diff.count("\n+") 
                removed = diff.count("\n-")
                file_stats.append(f"{fp}(+{added}/-{removed})")
            lines.append(f"FILES: {' '.join(file_stats)}")
            
            # Include diff but cap per-task
            for fp, diff in item.get("diffs", {}).items():
                if diff:
                    diff_lines = diff.split("\n")[:25]  # 25 lines per file per task
                    lines.append(f"DIFF {fp}:")
                    lines.append("\n".join(diff_lines))
        
        lines.append("\nReview all tasks above. JSON only:")
        return "\n".join(lines)

    async def batch_review(self, batch: list[dict], goal: str) -> dict:
        """
        Review multiple completed tasks in one API call.
        Uses a compressed prompt format to maximise information per token.
        """
        system_prompt = (
            "You are a code reviewer. Review multiple coding tasks at once.\n"
            "Be concise. Use the scoring table format exactly as shown. JSON only, no prose."
        )
        user_prompt = self._build_batch_prompt(batch, goal)
        
        try:
            result_text = await self._call(system_prompt, user_prompt)
            return self._parse_json(result_text)
        except httpx.HTTPStatusError as e:
            logger.error(f"API error during batch review: {e.response.status_code}")
            return {"batch_passed": False, "tasks": [], "critical_blockers": ["API unavailable"], "batch_summary": "API Error", "retry_tasks": [], "learnings": []}
        except Exception as e:
            logger.error(f"Batch review failed: {e}")
            return {"batch_passed": False, "tasks": [], "critical_blockers": [str(e)], "batch_summary": "Review Error", "retry_tasks": [], "learnings": []}

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
            msg = f"Summarization failed (HTTP {e.response.status_code}): {e.response.text[:100]}"
            logger.error(msg)
            return {"summary": msg, "key_decisions": [], "next_suggested": [], "risk_flags": []}
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return {"summary": f"Summarization failed: {e}", "key_decisions": [], "next_suggested": [], "risk_flags": []}

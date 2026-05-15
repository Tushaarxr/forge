"""Custom exceptions for the Forge coding agent.

Provides specific error types for better error handling and debugging.
"""

from __future__ import annotations


class ForgeError(Exception):
    """Base exception for all Forge errors.

    All custom exceptions inherit from this base class for easy catching.
    """

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConfigurationError(ForgeError):
    """Raised when configuration is invalid or missing.

    Examples:
        - Missing required environment variable
        - Invalid config file format
        - Invalid API key
    """
    pass


class BrainError(ForgeError):
    """Raised when Master Brain (Gemini/Anthropic) operations fail.

    Examples:
        - API key missing
        - Rate limit exceeded (429)
        - API returns error response
        - Network timeout
    """
    pass


class WorkerError(ForgeError):
    """Raised when Local Worker (LM Studio) operations fail.

    Examples:
        - LM Studio not running
        - Model not loaded
        - Execution timeout
        - Invalid model response
    """
    pass


class VectorStoreError(ForgeError):
    """Raised when vector store operations fail.

    Examples:
        - FAISS index corruption
        - Embedding model load failure
        - Search with empty index
    """
    pass


class ProjectGraphError(ForgeError):
    """Raised when project graph operations fail.

    Examples:
        - File parsing error
        - Graph serialization error
        - Circular dependency detected
    """
    pass


class PersistentMemoryError(ForgeError):
    """Raised when persistent memory operations fail.

    Examples:
        - Database corruption
        - Disk write failure
        - Index reconstruction failure
    """
    pass


class AutoRunnerError(ForgeError):
    """Raised when auto runner operations fail.

    Examples:
        - Plan parsing error
        - Checkpoint failure
        - Task execution failure
    """
    pass


class ContextEngineError(ForgeError):
    """Raised when context engine operations fail.

    Examples:
        - Vector search failure
        - Graph traversal error
        - Context budget exceeded
    """
    pass


class ValidationError(ForgeError):
    """Raised when input validation fails.

    Examples:
        - Invalid file path (path traversal attempt)
        - Invalid task format
        - Invalid plan format
    """
    pass
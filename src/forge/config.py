"""Configuration management for Forge.

Provides Pydantic-based configuration with validation and environment variable handling.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from forge.exceptions import ConfigurationError


class ForgeSettings(BaseSettings):
    """Forge configuration settings with validation.

    Loads from environment variables with validation.
    """

    # Master Brain Configuration
    master_provider: Literal["gemini", "anthropic"] = Field(
        default="gemini",
        description="Provider for master brain (gemini or anthropic)"
    )
    gemini_api_key: str = Field(default="", description="Gemini API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    master_model: str = Field(
        default="gemini-2.5-flash,gemini-2.5-flash-lite,gemini-3.1-flash-lite",
        description="Master brain model(s)"
    )

    # Local Worker Configuration
    local_model: str = Field(
        default="qwen3.5-9b-instruct",
        description="Local model name for LM Studio"
    )
    local_temperature: float = Field(
        default=0.6,
        ge=0.0,
        le=2.0,
        description="Temperature for local model"
    )
    local_top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Top-p for local model"
    )
    local_ctx_size: int = Field(
        default=8192,
        ge=512,
        le=32768,
        description="Context length for local model"
    )
    local_max_tokens: int = Field(
        default=4096,
        ge=256,
        le=16384,
        description="Max tokens for local model response"
    )

    # LM Studio Configuration
    lm_studio_base_url: str = Field(
        default="http://localhost:1234",
        description="LM Studio base URL"
    )

    # Vector Store Configuration
    chunk_size: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Chunk size for vector store"
    )
    embed_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )

    # Auto Runner Configuration
    checkpoint_every: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Pause for review every N tasks"
    )
    max_tasks: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum tasks per run"
    )

    # Paths
    graph_path: str = Field(
        default=".forge/project_graph.json",
        description="Path to project graph"
    )
    faiss_index_path: str = Field(
        default=".forge/vectors/faiss.index",
        description="Path to FAISS index"
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="WARNING",
        description="Logging level"
    )

    class Config:
        env_prefix = ""
        case_sensitive = False

    @field_validator("gemini_api_key", "anthropic_api_key", mode="before")
    @classmethod
    def validate_api_keys(cls, v: str) -> str:
        """Validate API keys are not empty if provider requires them."""
        return v or os.getenv(f"{cls.__name__.upper()}_API_KEY", "")

    @field_validator("master_model")
    @classmethod
    def validate_master_model(cls, v: str) -> str:
        """Ensure at least one model is specified."""
        if not v.strip():
            raise ConfigurationError("master_model cannot be empty")
        return v


def get_forge_config() -> ForgeSettings:
    """Get Forge configuration with validation.

    Returns:
        ForgeSettings: Validated configuration

    Raises:
        ConfigurationError: If required config is missing or invalid
    """
    try:
        return ForgeSettings()
    except Exception as e:
        raise ConfigurationError(f"Configuration validation failed: {e}")


def validate_required_env_vars() -> list[str]:
    """Validate that required environment variables are set.

    Returns:
        List of missing required variables (empty if all present)
    """
    required = ["GEMINI_API_KEY"]
    missing = []

    for var in required:
        if not os.getenv(var):
            missing.append(var)

    return missing


def get_missing_config_message() -> str:
    """Get user-friendly message about missing configuration.

    Returns:
        Formatted message for display to user
    """
    missing = validate_required_env_vars()
    if not missing:
        return ""

    msg = "Missing required configuration:\n"
    for var in missing:
        msg += f"  - {var}\n"
    msg += "\nRun 'forge setup' to configure or set environment variables."
    return msg
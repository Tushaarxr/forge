"""tests/test_config.py — Unit tests for configuration management."""

import os
import pytest
from unittest.mock import patch


class TestForgeSettingsDefaults:
    """Test ForgeSettings default values."""

    def test_default_master_provider(self):
        """Default master provider is gemini."""
        from forge.config import ForgeSettings
        with patch.dict(os.environ, {}, clear=True):
            settings = ForgeSettings()
            assert settings.master_provider == "gemini"

    def test_default_local_model(self):
        """Default local model is qwen3.5-9b-instruct."""
        from forge.config import ForgeSettings
        with patch.dict(os.environ, {}, clear=True):
            settings = ForgeSettings()
            assert settings.local_model == "qwen3.5-9b-instruct"

    def test_default_lm_studio_url(self):
        """Default LM Studio URL is localhost:1234."""
        from forge.config import ForgeSettings
        with patch.dict(os.environ, {}, clear=True):
            settings = ForgeSettings()
            assert settings.lm_studio_base_url == "http://localhost:1234"

    def test_default_temperature(self):
        """Default temperature is 0.6."""
        from forge.config import ForgeSettings
        with patch.dict(os.environ, {}, clear=True):
            settings = ForgeSettings()
            assert settings.local_temperature == 0.6

    def test_default_top_p(self):
        """Default top_p is 0.95."""
        from forge.config import ForgeSettings
        with patch.dict(os.environ, {}, clear=True):
            settings = ForgeSettings()
            assert settings.local_top_p == 0.95

    def test_default_ctx_size(self):
        """Default context size is 8192."""
        from forge.config import ForgeSettings
        with patch.dict(os.environ, {}, clear=True):
            settings = ForgeSettings()
            assert settings.local_ctx_size == 8192

    def test_default_max_tokens(self):
        """Default max tokens is 4096."""
        from forge.config import ForgeSettings
        with patch.dict(os.environ, {}, clear=True):
            settings = ForgeSettings()
            assert settings.local_max_tokens == 4096

    def test_default_chunk_size(self):
        """Default chunk size is 512."""
        from forge.config import ForgeSettings
        with patch.dict(os.environ, {}, clear=True):
            settings = ForgeSettings()
            assert settings.chunk_size == 512

    def test_default_embed_model(self):
        """Default embed model is all-MiniLM-L6-v2."""
        from forge.config import ForgeSettings
        with patch.dict(os.environ, {}, clear=True):
            settings = ForgeSettings()
            assert settings.embed_model == "all-MiniLM-L6-v2"

    def test_default_checkpoint_every(self):
        """Default checkpoint_every is 5."""
        from forge.config import ForgeSettings
        with patch.dict(os.environ, {}, clear=True):
            settings = ForgeSettings()
            assert settings.checkpoint_every == 5

    def test_default_max_tasks(self):
        """Default max_tasks is 50."""
        from forge.config import ForgeSettings
        with patch.dict(os.environ, {}, clear=True):
            settings = ForgeSettings()
            assert settings.max_tasks == 50

    def test_default_log_level(self):
        """Default log level is WARNING."""
        from forge.config import ForgeSettings
        with patch.dict(os.environ, {}, clear=True):
            settings = ForgeSettings()
            assert settings.log_level == "WARNING"


class TestNumericValidation:
    """Test numeric field validation."""

    def test_temperature_upper_bound(self):
        """Temperature cannot exceed 2.0."""
        from forge.config import ForgeSettings
        with pytest.raises(ValueError):
            ForgeSettings(local_temperature=2.5)

    def test_temperature_lower_bound(self):
        """Temperature cannot be below 0.0."""
        from forge.config import ForgeSettings
        with pytest.raises(ValueError):
            ForgeSettings(local_temperature=-0.1)

    def test_top_p_upper_bound(self):
        """top_p cannot exceed 1.0."""
        from forge.config import ForgeSettings
        with pytest.raises(ValueError):
            ForgeSettings(local_top_p=1.5)

    def test_top_p_lower_bound(self):
        """top_p cannot be below 0.0."""
        from forge.config import ForgeSettings
        with pytest.raises(ValueError):
            ForgeSettings(local_top_p=-0.1)

    def test_ctx_size_upper_bound(self):
        """ctx_size cannot exceed 32768."""
        from forge.config import ForgeSettings
        with pytest.raises(ValueError):
            ForgeSettings(local_ctx_size=40000)

    def test_ctx_size_lower_bound(self):
        """ctx_size cannot be below 512."""
        from forge.config import ForgeSettings
        with pytest.raises(ValueError):
            ForgeSettings(local_ctx_size=256)

    def test_max_tokens_upper_bound(self):
        """max_tokens cannot exceed 16384."""
        from forge.config import ForgeSettings
        with pytest.raises(ValueError):
            ForgeSettings(local_max_tokens=20000)

    def test_max_tokens_lower_bound(self):
        """max_tokens cannot be below 256."""
        from forge.config import ForgeSettings
        with pytest.raises(ValueError):
            ForgeSettings(local_max_tokens=128)

    def test_chunk_size_upper_bound(self):
        """chunk_size cannot exceed 2048."""
        from forge.config import ForgeSettings
        with pytest.raises(ValueError):
            ForgeSettings(chunk_size=3000)

    def test_chunk_size_lower_bound(self):
        """chunk_size cannot be below 128."""
        from forge.config import ForgeSettings
        with pytest.raises(ValueError):
            ForgeSettings(chunk_size=64)

    def test_checkpoint_every_upper_bound(self):
        """checkpoint_every cannot exceed 50."""
        from forge.config import ForgeSettings
        with pytest.raises(ValueError):
            ForgeSettings(checkpoint_every=100)

    def test_checkpoint_every_lower_bound(self):
        """checkpoint_every cannot be below 1."""
        from forge.config import ForgeSettings
        with pytest.raises(ValueError):
            ForgeSettings(checkpoint_every=0)

    def test_max_tasks_upper_bound(self):
        """max_tasks cannot exceed 500."""
        from forge.config import ForgeSettings
        with pytest.raises(ValueError):
            ForgeSettings(max_tasks=1000)

    def test_max_tasks_lower_bound(self):
        """max_tasks cannot be below 1."""
        from forge.config import ForgeSettings
        with pytest.raises(ValueError):
            ForgeSettings(max_tasks=0)


class TestEnvironmentVariableLoading:
    """Test environment variable loading."""

    def test_gemini_api_key_from_env(self):
        """gemini_api_key loads from GEMINI_API_KEY env var."""
        from forge.config import ForgeSettings
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-123"}, clear=True):
            settings = ForgeSettings(_case_sensitive=False)
            assert settings.gemini_api_key == "test-key-123"

    def test_anthropic_api_key_from_env(self):
        """anthropic_api_key loads from ANTHROPIC_API_KEY env var."""
        from forge.config import ForgeSettings
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "anthropic-key-456"}, clear=True):
            settings = ForgeSettings(_case_sensitive=False)
            assert settings.anthropic_api_key == "anthropic-key-456"

    def test_local_model_from_env(self):
        """local_model loads from LOCAL_MODEL env var."""
        from forge.config import ForgeSettings
        with patch.dict(os.environ, {"LOCAL_MODEL": "custom-model"}, clear=True):
            settings = ForgeSettings(_case_sensitive=False)
            assert settings.local_model == "custom-model"

    def test_master_provider_from_env(self):
        """master_provider loads from MASTER_PROVIDER env var."""
        from forge.config import ForgeSettings
        with patch.dict(os.environ, {"MASTER_PROVIDER": "anthropic"}, clear=True):
            settings = ForgeSettings(_case_sensitive=False)
            assert settings.master_provider == "anthropic"

    def test_master_model_from_env(self):
        """master_model loads from MASTER_MODEL env var."""
        from forge.config import ForgeSettings
        with patch.dict(os.environ, {"MASTER_MODEL": "gemini-2.0-pro"}, clear=True):
            settings = ForgeSettings(_case_sensitive=False)
            assert settings.master_model == "gemini-2.0-pro"

    def test_lm_studio_url_from_env(self):
        """lm_studio_base_url loads from LM_STUDIO_BASE_URL env var."""
        from forge.config import ForgeSettings
        with patch.dict(os.environ, {"LM_STUDIO_BASE_URL": "http://localhost:8080"}, clear=True):
            settings = ForgeSettings(_case_sensitive=False)
            assert settings.lm_studio_base_url == "http://localhost:8080"


class TestValidationFunctions:
    """Test validation functions."""

    def test_validate_required_env_vars_returns_empty_when_set(self):
        """Returns empty list when GEMINI_API_KEY is set."""
        from forge.config import validate_required_env_vars
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}, clear=True):
            result = validate_required_env_vars()
            assert result == []

    def test_validate_required_env_vars_returns_missing(self):
        """Returns list with GEMINI_API_KEY when not set."""
        from forge.config import validate_required_env_vars
        with patch.dict(os.environ, {}, clear=True):
            result = validate_required_env_vars()
            assert "GEMINI_API_KEY" in result

    def test_get_missing_config_message_returns_empty_when_configured(self):
        """Returns empty string when config is present."""
        from forge.config import get_missing_config_message
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test"}, clear=True):
            result = get_missing_config_message()
            assert result == ""

    def test_get_missing_config_message_returns_message_when_missing(self):
        """Returns formatted message when config is missing."""
        from forge.config import get_missing_config_message
        with patch.dict(os.environ, {}, clear=True):
            result = get_missing_config_message()
            assert "GEMINI_API_KEY" in result
            assert "forge setup" in result

    def test_get_forge_config_does_not_raise_with_defaults(self):
        """get_forge_config returns settings with defaults."""
        from forge.config import get_forge_config
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise
            settings = get_forge_config()
            assert settings.local_model == "qwen3.5-9b-instruct"


class TestMasterModelValidation:
    """Test master_model validation."""

    def test_empty_master_model_raises(self):
        """Empty master_model string raises ConfigurationError."""
        from forge.config import ForgeSettings
        with pytest.raises(Exception):  # ConfigurationError
            ForgeSettings(master_model="   ")

    def test_valid_master_model_comma_separated(self):
        """Valid master_model with multiple models works."""
        from forge.config import ForgeSettings
        settings = ForgeSettings(master_model="model1, model2, model3")
        assert settings.master_model == "model1, model2, model3"
"""tests/test_exceptions.py — Unit tests for custom exception types."""

import pytest
from forge.exceptions import (
    ForgeError,
    ConfigurationError,
    BrainError,
    WorkerError,
    VectorStoreError,
    ProjectGraphError,
    PersistentMemoryError,
    AutoRunnerError,
    ContextEngineError,
    ValidationError,
)


class TestExceptionHierarchy:
    """Test exception inheritance hierarchy."""

    def test_all_exceptions_inherit_from_forge_error(self):
        """All custom exceptions inherit from ForgeError."""
        exceptions = [
            ConfigurationError("test"),
            BrainError("test"),
            WorkerError("test"),
            VectorStoreError("test"),
            ProjectGraphError("test"),
            PersistentMemoryError("test"),
            AutoRunnerError("test"),
            ContextEngineError("test"),
            ValidationError("test"),
        ]
        for exc in exceptions:
            assert isinstance(exc, ForgeError)

    def test_forge_error_is_exception_subclass(self):
        """ForgeError is a subclass of Exception."""
        assert issubclass(ForgeError, Exception)


class TestForgeError:
    """Test ForgeError base class functionality."""

    def test_message_attribute_set(self):
        """ForgeError stores message in attribute."""
        err = ForgeError("test message")
        assert err.message == "test message"

    def test_message_in_str(self):
        """String representation includes message."""
        err = ForgeError("test message")
        assert "test message" in str(err)

    def test_details_dict_default_empty(self):
        """Details defaults to empty dict when not provided."""
        err = ForgeError("test")
        assert err.details == {}

    def test_details_dict_stored(self):
        """Details dict is stored when provided."""
        err = ForgeError("test", details={"key": "value"})
        assert err.details == {"key": "value"}

    def test_str_includes_details(self):
        """String representation includes details when present."""
        err = ForgeError("test message", details={"code": 123})
        result = str(err)
        assert "test message" in result
        assert "code=123" in result

    def test_str_without_details_no_parens(self):
        """String without details doesn't have parentheses."""
        err = ForgeError("test message")
        result = str(err)
        assert result == "test message"


class TestSpecificExceptions:
    """Test that specific exceptions can be raised and caught."""

    def test_raise_and_catch_configuration_error(self):
        """ConfigurationError can be raised and caught."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("config missing")

    def test_raise_and_catch_brain_error(self):
        """BrainError can be raised and caught."""
        with pytest.raises(BrainError):
            raise BrainError("API failed")

    def test_raise_and_catch_worker_error(self):
        """WorkerError can be raised and caught."""
        with pytest.raises(WorkerError):
            raise WorkerError("LM Studio not running")

    def test_raise_and_catch_vector_store_error(self):
        """VectorStoreError can be raised and caught."""
        with pytest.raises(VectorStoreError):
            raise VectorStoreError("index corrupted")

    def test_raise_and_catch_project_graph_error(self):
        """ProjectGraphError can be raised and caught."""
        with pytest.raises(ProjectGraphError):
            raise ProjectGraphError("parse failed")

    def test_raise_and_catch_persistent_memory_error(self):
        """PersistentMemoryError can be raised and caught."""
        with pytest.raises(PersistentMemoryError):
            raise PersistentMemoryError("DB corrupted")

    def test_raise_and_catch_auto_runner_error(self):
        """AutoRunnerError can be raised and caught."""
        with pytest.raises(AutoRunnerError):
            raise AutoRunnerError("plan failed")

    def test_raise_and_catch_context_engine_error(self):
        """ContextEngineError can be raised and caught."""
        with pytest.raises(ContextEngineError):
            raise ContextEngineError("budget exceeded")

    def test_raise_and_catch_validation_error(self):
        """ValidationError can be raised and caught."""
        with pytest.raises(ValidationError):
            raise ValidationError("invalid input")


class TestExceptionChaining:
    """Test exception can be caught by base type."""

    def test_catch_forge_error_gets_all(self):
        """Catching ForgeError catches all specific exceptions."""
        exceptions = [
            ConfigurationError("test"),
            BrainError("test"),
            WorkerError("test"),
        ]
        for exc in exceptions:
            with pytest.raises(ForgeError):
                raise exc


class TestExceptionDetails:
    """Test exception details functionality."""

    def test_multiple_details_keys(self):
        """Multiple details keys are all included in str."""
        err = ForgeError("test", details={"a": 1, "b": 2, "c": 3})
        result = str(err)
        assert "a=1" in result
        assert "b=2" in result
        assert "c=3" in result

    def test_details_with_none_values(self):
        """Details with None values are handled."""
        err = ForgeError("test", details={"key": None})
        # Should not crash
        result = str(err)
        assert "test" in result
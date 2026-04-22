"""
Tests for Thread Validator — structural integrity checks.
"""

import pytest

from src.validator.thread_validator import (
    ThreadValidator,
    ValidationError,
    ValidationResult,
    VALID_ROLES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role: str, content: str | list) -> dict:
    return {"role": role, "content": content}


def _tool_use_msg() -> dict:
    """Assistant message with tool_use block."""
    return {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me search that."},
            {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "test"}},
        ]
    }


def _tool_result_msg() -> dict:
    """User message with tool_result block."""
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "Search results here."},
        ]
    }


# ---------------------------------------------------------------------------
# Valid Thread Tests
# ---------------------------------------------------------------------------

class TestValidThreads:

    def test_simple_valid_thread(self):
        messages = [
            _msg("user", "Hello"),
            _msg("assistant", "Hi there"),
            _msg("user", "Question?"),
            _msg("assistant", "Answer."),
        ]
        validator = ThreadValidator()
        result = validator.validate(messages)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_valid_with_system(self):
        messages = [
            _msg("system", "You are helpful."),
            _msg("user", "Hello"),
            _msg("assistant", "Hi"),
        ]
        validator = ThreadValidator()
        result = validator.validate(messages)
        assert result.valid is True
        assert result.has_system_message is True

    def test_valid_with_tool_chain(self):
        messages = [
            _msg("user", "Search for X"),
            _tool_use_msg(),
            _tool_result_msg(),
            _msg("assistant", "Here are the results."),
        ]
        validator = ThreadValidator()
        result = validator.validate(messages)
        assert result.valid is True

    def test_single_user_message(self):
        messages = [_msg("user", "Hello")]
        validator = ThreadValidator()
        result = validator.validate(messages)
        assert result.valid is True

    def test_validate_or_raise_passes(self):
        messages = [
            _msg("user", "Hello"),
            _msg("assistant", "Hi"),
        ]
        validator = ThreadValidator()
        result = validator.validate_or_raise(messages)
        assert result.valid is True


# ---------------------------------------------------------------------------
# Invalid First Message Tests
# ---------------------------------------------------------------------------

class TestFirstMessage:

    def test_starts_with_assistant(self):
        messages = [
            _msg("assistant", "I shouldn't be first"),
            _msg("user", "Hello"),
        ]
        validator = ThreadValidator()
        result = validator.validate(messages)
        assert result.valid is False
        assert any("starts with assistant" in e for e in result.errors)

    def test_starts_with_system_then_assistant(self):
        """System → assistant without user in between."""
        messages = [
            _msg("system", "You are helpful."),
            _msg("assistant", "Hi"),
        ]
        validator = ThreadValidator()
        result = validator.validate(messages)
        assert result.valid is False
        assert any("expected 'user' after system" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Role Alternation Tests
# ---------------------------------------------------------------------------

class TestRoleAlternation:

    def test_consecutive_users(self):
        messages = [
            _msg("user", "first"),
            _msg("user", "second"),
            _msg("assistant", "response"),
        ]
        validator = ThreadValidator()
        result = validator.validate(messages)
        assert result.valid is False
        assert any("consecutive" in e and "user" in e for e in result.errors)

    def test_consecutive_assistants(self):
        messages = [
            _msg("user", "question"),
            _msg("assistant", "first answer"),
            _msg("assistant", "second answer"),
        ]
        validator = ThreadValidator()
        result = validator.validate(messages)
        assert result.valid is False
        assert any("consecutive" in e and "assistant" in e for e in result.errors)

    def test_system_exempt_from_alternation(self):
        """System at position 0 doesn't count for alternation."""
        messages = [
            _msg("system", "System prompt"),
            _msg("user", "Hello"),
            _msg("assistant", "Hi"),
        ]
        validator = ThreadValidator()
        result = validator.validate(messages)
        assert result.valid is True


# ---------------------------------------------------------------------------
# Empty Content Tests
# ---------------------------------------------------------------------------

class TestEmptyContent:

    def test_empty_string_content(self):
        messages = [
            _msg("user", ""),
            _msg("assistant", "response"),
        ]
        validator = ThreadValidator()
        result = validator.validate(messages)
        assert result.valid is False
        assert any("empty content" in e for e in result.errors)

    def test_whitespace_only_content(self):
        messages = [
            _msg("user", "   "),
            _msg("assistant", "response"),
        ]
        validator = ThreadValidator()
        result = validator.validate(messages)
        assert result.valid is False

    def test_missing_content_field(self):
        messages = [{"role": "user"}]
        validator = ThreadValidator()
        result = validator.validate(messages)
        assert result.valid is False
        assert any("missing 'content'" in e for e in result.errors)

    def test_empty_content_list(self):
        messages = [_msg("user", [])]
        validator = ThreadValidator()
        result = validator.validate(messages)
        assert result.valid is False


# ---------------------------------------------------------------------------
# Invalid Role Tests
# ---------------------------------------------------------------------------

class TestInvalidRoles:

    def test_missing_role(self):
        messages = [{"content": "hello"}]
        validator = ThreadValidator()
        result = validator.validate(messages)
        assert result.valid is False
        assert any("missing 'role'" in e for e in result.errors)

    def test_invalid_role_value(self):
        messages = [_msg("bot", "hello")]
        validator = ThreadValidator()
        result = validator.validate(messages)
        assert result.valid is False
        assert any("invalid role" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Tool Chain Tests
# ---------------------------------------------------------------------------

class TestToolChainIntegrity:

    def test_tool_use_without_result_strict(self):
        """In strict mode, missing tool_result is an error."""
        messages = [
            _msg("user", "Search for X"),
            _tool_use_msg(),
            _msg("user", "Never mind"),  # no tool_result
        ]
        validator = ThreadValidator(strict=True)
        result = validator.validate(messages)
        assert any("tool_use requires tool_result" in e for e in result.errors)

    def test_tool_use_without_result_lenient(self):
        """In lenient mode, missing tool_result is just a warning."""
        messages = [
            _msg("user", "Search for X"),
            _tool_use_msg(),
            _msg("user", "Never mind"),
        ]
        validator = ThreadValidator(strict=False)
        result = validator.validate(messages)
        assert len(result.warnings) > 0
        # Warnings don't make it invalid in lenient mode
        # (but alternation error will still fire)

    def test_tool_use_at_end_of_thread(self):
        """tool_use as last message — no tool_result possible."""
        messages = [
            _msg("user", "Search for X"),
            _tool_use_msg(),
        ]
        validator = ThreadValidator(strict=True)
        result = validator.validate(messages)
        assert any("thread ends" in e for e in result.errors)

    def test_valid_tool_chain(self):
        """Proper tool_use → tool_result sequence passes."""
        messages = [
            _msg("user", "Search for X"),
            _tool_use_msg(),
            _tool_result_msg(),
            _msg("assistant", "Found it."),
        ]
        validator = ThreadValidator(strict=True)
        result = validator.validate(messages)
        # No tool chain errors
        tool_errors = [e for e in result.errors if "tool" in e.lower()]
        assert len(tool_errors) == 0


# ---------------------------------------------------------------------------
# Empty Thread Tests
# ---------------------------------------------------------------------------

class TestEmptyThread:

    def test_empty_strict(self):
        validator = ThreadValidator(strict=True)
        result = validator.validate([])
        assert result.valid is False

    def test_empty_lenient(self):
        validator = ThreadValidator(strict=False)
        result = validator.validate([])
        assert result.valid is True
        assert len(result.warnings) > 0


# ---------------------------------------------------------------------------
# ValidationError Tests
# ---------------------------------------------------------------------------

class TestValidationError:

    def test_validate_or_raise_raises(self):
        messages = [
            _msg("assistant", "I shouldn't be first"),
        ]
        validator = ThreadValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_or_raise(messages)
        assert len(exc_info.value.errors) > 0

    def test_error_message_readable(self):
        messages = [
            _msg("assistant", "bad start"),
            _msg("assistant", "double assistant"),
        ]
        validator = ThreadValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_or_raise(messages)
        error_msg = str(exc_info.value)
        assert "validation failed" in error_msg.lower()


# ---------------------------------------------------------------------------
# Result Metadata Tests
# ---------------------------------------------------------------------------

class TestResultMetadata:

    def test_message_count(self):
        messages = [
            _msg("user", "a"),
            _msg("assistant", "b"),
            _msg("user", "c"),
        ]
        validator = ThreadValidator()
        result = validator.validate(messages)
        assert result.message_count == 3

    def test_role_sequence(self):
        messages = [
            _msg("system", "sys"),
            _msg("user", "u"),
            _msg("assistant", "a"),
        ]
        validator = ThreadValidator()
        result = validator.validate(messages)
        assert result.role_sequence == ["system", "user", "assistant"]

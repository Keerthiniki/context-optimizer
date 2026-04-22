from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Validation error
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised when assembled thread fails structural integrity checks."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        msg = f"Thread validation failed with {len(errors)} error(s):\n"
        msg += "\n".join(f"  - {e}" for e in errors)
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of thread validation."""
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    message_count: int = 0
    has_system_message: bool = False
    role_sequence: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Valid roles
# ---------------------------------------------------------------------------

VALID_ROLES = {"system", "user", "assistant"}


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ThreadValidator:
    """
    Validates assembled conversation threads for Claude API compatibility.

    Usage:
        validator = ThreadValidator()
        result = validator.validate(messages)       # returns ValidationResult
        validator.validate_or_raise(messages)       # raises ValidationError if invalid
    """

    def __init__(self, strict: bool = True):
        """
        Args:
            strict: If True, warnings are promoted to errors.
        """
        self.strict = strict

    def validate(self, messages: list[dict]) -> ValidationResult:
        """
        Run all validation checks and return a ValidationResult.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.

        Returns:
            ValidationResult with valid flag, errors, and warnings.
        """
        result = ValidationResult(message_count=len(messages))

        if not messages:
            result.warnings.append("Thread is empty — no messages to validate.")
            if self.strict:
                result.valid = False
                result.errors.append("Thread is empty.")
            return result

        # Collect role sequence
        result.role_sequence = [m.get("role", "MISSING") for m in messages]

        # Run each check
        self._check_valid_roles(messages, result)
        self._check_first_message(messages, result)
        self._check_no_empty_content(messages, result)
        self._check_role_alternation(messages, result)
        self._check_tool_chain_integrity(messages, result)

        # Final verdict
        if result.errors:
            result.valid = False

        return result

    def validate_or_raise(self, messages: list[dict]) -> ValidationResult:
        """
        Validate and raise ValidationError if thread is invalid.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.

        Returns:
            ValidationResult if valid.

        Raises:
            ValidationError if any checks fail.
        """
        result = self.validate(messages)
        if not result.valid:
            raise ValidationError(result.errors)
        return result

    # --- Individual checks ---

    def _check_valid_roles(self, messages: list[dict], result: ValidationResult) -> None:
        """Every message must have a role from VALID_ROLES."""
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if role is None:
                result.errors.append(f"Message {i}: missing 'role' field.")
            elif role not in VALID_ROLES:
                result.errors.append(
                    f"Message {i}: invalid role '{role}'. "
                    f"Must be one of: {', '.join(sorted(VALID_ROLES))}."
                )

    def _check_first_message(self, messages: list[dict], result: ValidationResult) -> None:
        """Thread must start with system or user, never assistant."""
        first_role = messages[0].get("role")
        if first_role == "system":
            result.has_system_message = True
        elif first_role == "user":
            pass  # valid
        elif first_role == "assistant":
            result.errors.append(
                "Thread starts with assistant message — "
                "must start with system or user."
            )

    def _check_no_empty_content(self, messages: list[dict], result: ValidationResult) -> None:
        """No message should have empty or missing content."""
        for i, msg in enumerate(messages):
            content = msg.get("content")
            if content is None:
                result.errors.append(f"Message {i}: missing 'content' field.")
            elif isinstance(content, str) and content.strip() == "":
                result.errors.append(f"Message {i} ({msg.get('role', '?')}): empty content.")
            elif isinstance(content, list) and len(content) == 0:
                result.errors.append(f"Message {i} ({msg.get('role', '?')}): empty content list.")

    def _check_role_alternation(self, messages: list[dict], result: ValidationResult) -> None:
        """
        Enforce user/assistant alternation.
        System message at position 0 is exempt.
        """
        # Find where alternation checking starts
        start = 0
        if messages[0].get("role") == "system":
            start = 1
            if len(messages) < 2:
                return
            # Message after system must be user
            if messages[1].get("role") != "user":
                result.errors.append(
                    f"Message 1: expected 'user' after system message, "
                    f"got '{messages[1].get('role')}'."
                )
                return

        # Check alternation from start
        for i in range(start + 1, len(messages)):
            prev_role = messages[i - 1].get("role")
            curr_role = messages[i].get("role")

            # Skip system messages in alternation check
            if prev_role == "system":
                continue

            if curr_role == prev_role:
                result.errors.append(
                    f"Message {i}: consecutive '{curr_role}' messages "
                    f"(positions {i-1} and {i}). Alternation violated."
                )

    def _check_tool_chain_integrity(self, messages: list[dict], result: ValidationResult) -> None:
        """
        If an assistant message contains tool_use, the next user message
        should contain tool_result. Warn if not (may have been compressed).
        """
        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue

            content = msg.get("content", "")
            if not isinstance(content, list):
                continue

            has_tool_use = any(
                isinstance(b, dict) and b.get("type") == "tool_use"
                for b in content
            )

            if not has_tool_use:
                continue

            # Look for tool_result in next message
            if i + 1 >= len(messages):
                result.warnings.append(
                    f"Message {i}: tool_use with no following message."
                )
                if self.strict:
                    result.errors.append(
                        f"Message {i}: tool_use at end of thread — "
                        f"tool_result expected but thread ends."
                    )
                continue

            next_msg = messages[i + 1]
            next_content = next_msg.get("content", "")

            if isinstance(next_content, list):
                has_tool_result = any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in next_content
                )
            else:
                has_tool_result = False

            if not has_tool_result:
                result.warnings.append(
                    f"Message {i}: tool_use not followed by tool_result "
                    f"in message {i+1}. Chain may be broken."
                )
                if self.strict:
                    result.errors.append(
                        f"Message {i}: tool_use requires tool_result in "
                        f"message {i+1}, but none found."
                    )

"""Session-scoped conversation and filesystem state."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any

from app.confirmation import ConfirmationRequest
from app.filesystem_context import FilesystemContext
from app.llm import ToolCall


@dataclass
class PendingConfirmation:
    tool_name: str
    arguments: dict[str, Any]
    request: ConfirmationRequest


@dataclass
class ConversationState:
    filesystem_context: FilesystemContext | None = None
    history: list[tuple[str, str]] = field(default_factory=list)
    pending_confirmation: PendingConfirmation | None = None

    def record_turn(self, user_message: str, assistant_message: str) -> None:
        self.history.append((user_message, assistant_message))
        if len(self.history) > 50:
            del self.history[:-50]

    def prepare_tool_call(self, tool_call: ToolCall, user_message: str) -> ToolCall:
        """Preserve explicit turn arguments and resolve pronoun targets structurally."""
        arguments = dict(tool_call.arguments)
        tokens = self._tokens(user_message)
        if tool_call.name in {"create_file", "create_directory"}:
            target = self._target_after(tokens, {"in", "inside", "into", "to"})
            if target and "location" not in arguments:
                arguments["location"] = target
            quoted = self._quoted_values(user_message)
            if tool_call.name == "create_file" and quoted and "content" not in arguments:
                arguments["content"] = quoted[-1]
        elif tool_call.name == "write_file":
            quoted = self._quoted_values(user_message)
            if quoted:
                arguments["content"] = quoted[-1]
            target = self._target_after(tokens, {"to", "in", "into"})
            if target:
                arguments["path"] = target
            elif arguments.get("path") == arguments.get("content"):
                arguments["path"] = "it"
        elif tool_call.name in {"read_file", "delete_file", "list_directory"}:
            target = self._target_after(tokens, {"to", "in", "from"})
            if target:
                arguments["path" if tool_call.name != "list_directory" else "path"] = target
            elif any(token in {"it", "this", "that"} for token in tokens):
                arguments["path"] = "it"
        elif tool_call.name == "search_files":
            explicit_location = self._target_after(tokens, {"in", "inside", "under", "at"})
            if explicit_location:
                arguments["root"] = explicit_location
            elif self.filesystem_context is not None:
                arguments["root"] = str(self.filesystem_context.search_root())
        elif tool_call.name in {"rename_file", "move_file"}:
            destination = self._target_after(tokens, {"to", "into"})
            if destination:
                arguments["destination"] = destination
            if any(token in {"it", "this", "that"} for token in tokens):
                arguments["source"] = "it"
        return ToolCall(name=tool_call.name, arguments=arguments, call_id=tool_call.call_id)

    def resolve_filesystem_intent(self, tool_call: ToolCall, user_message: str) -> ToolCall:
        """Correct only clear filesystem operation intent before argument resolution."""
        tokens = self._tokens(user_message)
        operation = next((token for token in tokens if token in {
            "create", "read", "write", "delete", "rename", "move", "list", "search",
        }), None)
        tool_name = {
            "create": "create_file",
            "read": "read_file",
            "write": "write_file",
            "delete": "delete_file",
            "rename": "rename_file",
            "move": "move_file",
            "list": "list_directory",
            "search": "search_files",
        }.get(operation)
        if tool_name is None:
            return self.prepare_tool_call(tool_call, user_message)
        return self.prepare_tool_call(
            ToolCall(name=tool_name, arguments=tool_call.arguments, call_id=tool_call.call_id),
            user_message,
        )

    @staticmethod
    def _tokens(message: str) -> list[str]:
        try:
            return [token.strip(".,!?;:").lower() for token in shlex.split(message)]
        except ValueError:
            return message.lower().split()

    @staticmethod
    def _quoted_values(message: str) -> list[str]:
        values: list[str] = []
        current: list[str] = []
        quote: str | None = None
        for char in message:
            if quote is None and char in {'"', "'"}:
                quote = char
            elif quote == char:
                values.append("".join(current))
                current = []
                quote = None
            elif quote is not None:
                current.append(char)
        return values

    @staticmethod
    def _target_after(tokens: list[str], prepositions: set[str]) -> str | None:
        for index, token in enumerate(tokens):
            if token in prepositions and index + 1 < len(tokens):
                target = tokens[index + 1:]
                while target and target[0] in {"the", "file", "folder", "directory"}:
                    target.pop(0)
                if target:
                    return " ".join(target)
        return None

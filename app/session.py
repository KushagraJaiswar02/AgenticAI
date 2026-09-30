"""Session-scoped conversation and filesystem state."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.confirmation import ConfirmationRequest
from app.filesystem_context import FilesystemContext
from app.browser_context import BrowserContext
from app.llm import ConversationMessage, ToolCall
from app.path_resolver import PathResolver


@dataclass
class PendingConfirmation:
    tool_name: str
    arguments: dict[str, Any]
    request: ConfirmationRequest


@dataclass
class ConversationState:
    filesystem_context: FilesystemContext | None = None
    browser_context: BrowserContext | None = None
    history: list[tuple[str, str]] = field(default_factory=list)
    message_history: list[ConversationMessage] = field(default_factory=list)
    pending_tool_events: list[ConversationMessage] = field(default_factory=list)
    history_limit: int = 10
    pending_confirmation: PendingConfirmation | None = None

    def record_turn(self, user_message: str, assistant_message: str) -> None:
        self.history.append((user_message, assistant_message))
        self.message_history.append(ConversationMessage("user", user_message))
        self.message_history.extend(self.pending_tool_events)
        self.pending_tool_events.clear()
        self.message_history.append(ConversationMessage("assistant", assistant_message))
        if len(self.history) > 50:
            del self.history[:-50]
        self._trim_message_history()

    def record_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        self.pending_tool_events.append(ConversationMessage("assistant", f"TOOL_CALL {tool_name} {json.dumps(arguments, sort_keys=True, default=str)}"))

    def record_tool_result(self, tool_name: str, result: dict[str, Any]) -> None:
        self.pending_tool_events.append(ConversationMessage("assistant", f"TOOL_RESULT {tool_name} {json.dumps(result, sort_keys=True, default=str)}"))

    def recent_messages(self, max_turns: int | None = None) -> list[ConversationMessage]:
        """Return bounded prior dialogue plus factual tool events in order."""
        limit = max_turns if max_turns is not None else self.history_limit
        if self.message_history:
            return list(self.message_history[-(limit * 2):])
        turns = self.history[-limit:]
        return [message for user_message, assistant_message in turns for message in (
            ConversationMessage("user", user_message), ConversationMessage("assistant", assistant_message)
        )]

    def _trim_message_history(self) -> None:
        del self.message_history[: max(0, len(self.message_history) - (self.history_limit * 2))]

    def prepare_tool_call(self, tool_call: ToolCall, user_message: str) -> ToolCall:
        """Preserve explicit turn arguments and resolve pronoun targets structurally."""
        arguments = dict(tool_call.arguments)
        tokens = self._tokens(user_message)
        lowered = self._lower_tokens(tokens)
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
            if self._mentions_recent_file(lowered):
                arguments["path"] = "file we just created"
            else:
                target = self._target_after(tokens, {"to", "in", "into"})
            if not self._mentions_recent_file(lowered) and target:
                arguments["path"] = target
            elif arguments.get("path") == arguments.get("content"):
                arguments["path"] = "it"
        elif tool_call.name in {"read_file", "delete_file", "list_directory"}:
            target = self._target_after(tokens, {"to", "in", "from"})
            path = arguments.get("path")
            pronoun = any(token in {"it", "this", "that"} for token in lowered)
            if pronoun:
                arguments["path"] = "it"
            elif not path and target:
                arguments["path"] = target
            elif path and target and self._is_bare_basename(path):
                arguments["path"] = str(Path(target) / path)
        elif tool_call.name == "search_files":
            explicit_location = self._target_after(tokens, {"in", "inside", "under", "at"})
            if explicit_location:
                arguments["root"] = explicit_location
            elif self.filesystem_context is not None:
                arguments["root"] = str(self.filesystem_context.search_root())
        elif tool_call.name == "change_directory":
            target = self._target_after(tokens, {"to", "inside", "in", "into"})
            if target:
                arguments["location"] = target.removesuffix(" folder").removesuffix(" directory")
        elif tool_call.name in {"rename_file", "move_file"}:
            destination = self._target_after(tokens, {"to", "into"})
            if destination:
                arguments["destination"] = destination
            if any(token in {"it", "this", "that"} for token in lowered):
                arguments["source"] = "it"
        return ToolCall(name=tool_call.name, arguments=arguments, call_id=tool_call.call_id)

    def resolve_filesystem_intent(self, tool_call: ToolCall, user_message: str) -> ToolCall:
        """Correct only clear filesystem operation intent before argument resolution."""
        tokens = self._tokens(user_message)
        valid_tools = {
            "create_file", "create_directory", "read_file", "write_file", "delete_file",
            "rename_file", "move_file", "list_directory", "search_files", "change_directory",
        }
        if tool_call.name in valid_tools and not self._obviously_incompatible(tool_call.name, tokens):
            return self.prepare_tool_call(tool_call, user_message)
        lowered = self._lower_tokens(tokens)
        operation = next((token for token in lowered if token in {
            "create", "read", "write", "delete", "rename", "move", "list", "search", "change", "go",
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
            "change": "change_directory",
            "go": "change_directory",
        }.get(operation)
        if tool_name is None:
            return self.prepare_tool_call(tool_call, user_message)
        return self.prepare_tool_call(
            ToolCall(name=tool_name, arguments=tool_call.arguments, call_id=tool_call.call_id),
            user_message,
        )

    @staticmethod
    def _obviously_incompatible(tool_name: str, tokens: list[str]) -> bool:
        lowered = ConversationState._lower_tokens(tokens)
        operation = next((token for token in lowered if token in {
            "create", "read", "write", "delete", "rename", "move", "list", "search", "change", "go",
        }), None)
        if tool_name == "list_directory" and operation in {"read", "write", "delete", "rename", "move"}:
            return True
        if operation == "create" and tool_name in {"create_file", "create_directory"}:
            wants_directory = any(token in {"folder", "directory"} for token in lowered)
            wants_file = "file" in lowered
            return (wants_directory and tool_name == "create_file") or (wants_file and tool_name == "create_directory")
        return False

    @staticmethod
    def _tokens(message: str) -> list[str]:
        try:
            tokens = shlex.split(message, posix=False)
        except ValueError:
            tokens = message.split()
        return [ConversationState._strip_matching_quotes(token.strip(".,!?;:")) for token in tokens]

    @staticmethod
    def _lower_tokens(tokens: list[str]) -> list[str]:
        return [token.lower() for token in tokens]

    @staticmethod
    def _strip_matching_quotes(token: str) -> str:
        if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
            return token[1:-1]
        return token

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
        lowered = [token.lower() for token in tokens]
        wanted = {item.lower() for item in prepositions}
        for index, token in enumerate(lowered):
            if token in wanted and index + 1 < len(tokens):
                target = tokens[index + 1:]
                while target and target[0].lower() in {"the", "file", "folder", "directory"}:
                    target.pop(0)
                if target:
                    return " ".join(target)
        return None

    @staticmethod
    def _is_bare_basename(value: Any) -> bool:
        if not isinstance(value, str) or not value or value in {"..", "."} or ".." in value:
            return False
        if any(separator in value for separator in ("\\", "/")):
            return False
        return not (len(value) >= 2 and value[1] == ":")

    @staticmethod
    def _mentions_recent_file(tokens: list[str]) -> bool:
        return (
            "created" in tokens and "file" in tokens
        ) or ("previous" in tokens and "file" in tokens)

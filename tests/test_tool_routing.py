from app.browser_context import BrowserContext
from app.browser_tools import create_browser_tools
from app.fake_provider import FakeLLMProvider
from app.llm import LLMResponse, ToolCall
from app.orchestrator import Orchestrator
from app.session import ConversationState
from app.tools import ToolRegistry


class FakeBrowserBackend:
    def open(self, browser: str) -> None:
        del browser

    def navigate(self, url: str) -> tuple[str, str]:
        return url, "GitHub"

    def current_url(self) -> str:
        return "https://github.com"

    def current_title(self) -> str:
        return "GitHub"


def test_registered_open_url_is_not_remapped_by_go_verb():
    state = ConversationState()
    call = ToolCall("open_url", {"url": "https://github.com"})

    result = state.resolve_filesystem_intent(call, "Go to https://github.com", {"open_url"})

    assert result == call


def test_registered_open_url_is_not_remapped_by_search_verb():
    state = ConversationState()
    call = ToolCall("open_url", {"url": "https://github.com"})

    result = state.resolve_filesystem_intent(call, "search for jarvis on google", {"open_url"})

    assert result == call


def test_registered_current_page_is_not_remapped_by_list_verb():
    state = ConversationState()
    call = ToolCall("get_current_page", {})

    result = state.resolve_filesystem_intent(call, "list what page I am on", {"get_current_page"})

    assert result == call


def test_filesystem_tool_still_remaps_incompatible_intent():
    state = ConversationState()
    call = ToolCall("list_directory", {"path": "ARCHITECTURE.md"})

    result = state.resolve_filesystem_intent(call, "read ARCHITECTURE.md from docs", {"list_directory"})

    assert result.name == "read_file"


def test_unknown_tool_name_still_remaps_to_filesystem_intent():
    state = ConversationState()
    call = ToolCall("file_reader", {"path": "notes.txt"})

    result = state.resolve_filesystem_intent(call, "read notes.txt", set())

    assert result.name == "read_file"


def test_orchestrator_executes_registered_open_url_without_change_directory():
    context = BrowserContext(_backend=FakeBrowserBackend())
    context.open_browser()
    audit = None
    registry = ToolRegistry(create_browser_tools(context))
    audit = registry.audit_logger
    provider = FakeLLMProvider([
        LLMResponse(tool_call=ToolCall("open_url", {"url": "https://github.com"})),
        LLMResponse(text="Opened GitHub."),
    ])

    result = Orchestrator(provider, registry).process("Go to https://github.com")

    assert result == "Opened GitHub."
    assert audit.records[-1]["tool_name"] == "open_url"
    assert not any(record["tool_name"] == "change_directory" for record in audit.records)

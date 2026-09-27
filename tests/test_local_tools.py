import pytest

from app.tools import (
    CalculatorTool,
    DateTool,
    SystemInfoTool,
    TimeTool,
    ToolExecutionError,
    ToolRegistry,
)


def test_time_tool_returns_local_time_structure() -> None:
    result = TimeTool().execute({})
    assert result.success is True
    assert set(result.data) == {"time", "timezone"}
    assert len(result.data["time"]) == 8
    assert result.data["timezone"] == "local"


def test_date_tool_returns_date_structure() -> None:
    result = DateTool().execute({})
    assert result.success is True
    assert set(result.data) == {"date", "day"}
    assert len(result.data["date"]) == 10
    assert result.data["day"]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 2", 4),
        ("25 * 17", 425),
        ("(25 + 15) * 3", 120),
        ("144 / 12", 12),
        ("10 % 3", 1),
    ],
)
def test_calculator_tool_supports_safe_arithmetic(expression: str, expected: int | float) -> None:
    result = CalculatorTool().execute({"expression": expression})
    assert result.success is True
    assert result.data == {"expression": expression, "result": expected}


@pytest.mark.parametrize(
    "expression",
    [
        '__import__("os")',
        "open('secret.txt')",
        "os.system('whoami')",
        "globals()",
        "lambda: 1",
        "1 / 0",
    ],
)
def test_calculator_tool_rejects_unsafe_or_invalid_expressions(expression: str) -> None:
    with pytest.raises(ToolExecutionError):
        CalculatorTool().execute({"expression": expression})


def test_system_info_tool_returns_non_sensitive_runtime_fields() -> None:
    result = SystemInfoTool().execute({})
    assert result.success is True
    assert set(result.data) == {"os", "platform", "architecture", "python_version"}
    assert not any("key" in value.lower() for value in result.data.values())
    assert not any("password" in value.lower() for value in result.data.values())


def test_registry_definitions_and_dispatch_cover_local_tools() -> None:
    registry = ToolRegistry([TimeTool(), DateTool(), CalculatorTool(), SystemInfoTool()])
    assert registry.list() == ["calculator", "date", "system_info", "time"]
    assert {definition.name for definition in registry.definitions()} == {
        "calculator",
        "date",
        "system_info",
        "time",
    }
    assert registry.execute("calculator", {"expression": "2 + 2"}).data["result"] == 4
    assert registry.execute("time", {}).success is True
    assert registry.execute("date", {}).success is True
    assert registry.execute("system_info", {}).success is True

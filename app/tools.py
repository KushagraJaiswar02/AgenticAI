"""Minimal V0.2 tool abstraction and the first concrete WeatherTool."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable

import httpx
from pydantic import BaseModel, ValidationError

from app.errors import ApplicationError
from app.llm import ToolDefinition


class ToolError(ApplicationError):
    """Raised when a tool registration or execution fails."""


class UnknownToolError(ToolError):
    """Raised when a tool name is not registered."""


class ToolArgumentError(ToolError):
    """Raised when a tool receives malformed arguments."""


class ToolExecutionError(ToolError):
    """Raised when a tool fails while executing."""


@dataclass(frozen=True)
class ToolExecutionResult:
    """Normalized result returned from a tool execution."""

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None


class Tool(ABC):
    """Abstract tool contract for JARVIS V0.2."""

    name: str = ""
    description: str = ""
    input_model: type[BaseModel] | None = None

    @abstractmethod
    def execute(self, arguments: BaseModel | dict[str, Any]) -> ToolExecutionResult:
        """Execute the tool with validated arguments."""
        raise NotImplementedError

    def call(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        """Validate the payload and delegate to the tool implementation."""
        if self.input_model is None:
            raise ToolArgumentError(f"Tool '{self.name}' does not define an input model")
        try:
            validated = self.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolArgumentError(f"Tool '{self.name}' received invalid arguments") from exc
        return self.execute(validated)


class ToolRegistry:
    """Maintain a small set of explicitly registered tools."""

    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register a tool by name, rejecting duplicates."""
        if not isinstance(tool, Tool):
            raise ToolError("Only Tool instances may be registered")
        if tool.name in self._tools:
            raise ToolError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Return a tool if it exists."""
        return self._tools.get(name)

    def list(self) -> list[str]:
        """Return the names of all registered tools."""
        return sorted(self._tools)

    def definitions(self) -> list[ToolDefinition]:
        """Expose the registry's tool schemas to provider-neutral LLM calls."""
        return [
            ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters={
                    "type": "object",
                    "properties": {
                        name: {
                            "type": value.get("type", "string"),
                            "description": value.get("description", ""),
                        }
                        for name, value in getattr(tool.input_model, "model_json_schema", lambda: {"properties": {}})().get("properties", {}).items()
                    },
                    "required": getattr(tool.input_model, "model_json_schema", lambda: {"required": []})().get("required", []),
                },
            )
            for tool in self._tools.values()
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        """Execute a registered tool by name."""
        tool = self.get(name)
        if tool is None:
            raise UnknownToolError(f"Tool '{name}' is not registered")
        try:
            return tool.call(arguments)
        except ToolExecutionError:
            raise
        except ApplicationError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise ToolExecutionError(f"Tool '{name}' failed unexpectedly") from exc


class WeatherInput(BaseModel):
    """Input schema for the weather lookup tool."""

    location: str


class WeatherTool(Tool):
    """Fetch the current weather for a location using Open-Meteo."""

    name = "weather"
    description = "Get the current weather for a specific location."
    input_model = WeatherInput

    def execute(self, arguments: WeatherInput | dict[str, Any]) -> ToolExecutionResult:
        location = arguments.location.strip() if isinstance(arguments, WeatherInput) else str(arguments["location"]).strip()
        if not location:
            raise ToolArgumentError("Weather tool requires a non-empty location")

        try:
            geocode_response = httpx.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1, "language": "en", "format": "json"},
                timeout=10,
            )
            geocode_response.raise_for_status()
            geocode_payload = geocode_response.json()
            results = geocode_payload.get("results") or []
            if not results:
                raise ToolExecutionError(f"No weather data found for '{location}'")
            location_data = results[0]
            latitude = location_data.get("latitude")
            longitude = location_data.get("longitude")
            if latitude is None or longitude is None:
                raise ToolExecutionError(f"Weather lookup for '{location}' failed")

            forecast_response = httpx.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,weather_code",
                    "timezone": "auto",
                },
                timeout=10,
            )
            forecast_response.raise_for_status()
            forecast_payload = forecast_response.json()
            current = forecast_payload.get("current") or {}
            if not current:
                raise ToolExecutionError(f"Weather lookup for '{location}' failed")

            return ToolExecutionResult(
                success=True,
                data={
                    "location": location,
                    "latitude": latitude,
                    "longitude": longitude,
                    "temperature_c": current.get("temperature_2m"),
                    "apparent_temperature_c": current.get("apparent_temperature"),
                    "humidity_percent": current.get("relative_humidity_2m"),
                    "wind_kph": current.get("wind_speed_10m"),
                    "weather_code": current.get("weather_code"),
                },
            )
        except Exception as exc:
            if isinstance(exc, ToolExecutionError):
                raise
            raise ToolExecutionError(f"Weather service unavailable for '{location}'") from exc

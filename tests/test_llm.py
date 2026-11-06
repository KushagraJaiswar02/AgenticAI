from app.llm import LLMProvider, LLMResponse


class FakeLLM(LLMProvider):
    def generate(self, prompt: str, tools=None, *, tool_call=None, tool_result=None) -> LLMResponse:
        return LLMResponse(text=f"reply: {prompt}")


def test_llm_interface_can_be_mocked() -> None:
    assert FakeLLM().generate("hello").text == "reply: hello"

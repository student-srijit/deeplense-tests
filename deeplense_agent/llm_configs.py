"""LLM backends for the agent. All run locally through Ollama's OpenAI-compatible API."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

OLLAMA_URL = "http://localhost:11434/v1"


@dataclass(frozen=True)
class LLMConfig:
    name: str
    ollama_model: str

    def build(self) -> OpenAIChatModel:
        return OpenAIChatModel(self.ollama_model, provider=OllamaProvider(base_url=OLLAMA_URL))


# The agent needs a model that supports tool calling (check `ollama show <model>`:
# "Capabilities" must list "tools").
LLAMA3 = LLMConfig("llama3-8b", "llama3")  # no tool support: kept to show the failure
QWEN25 = LLMConfig("qwen2.5-7b", "qwen2.5:7b")  # supports tools

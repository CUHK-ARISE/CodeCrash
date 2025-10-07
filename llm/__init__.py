from llm.base import LLMChat
from llm.format import Message
from llm.openai_api import OpenAIChat
from llm.platform_api import AzureChat, DeepInfraChat, DeepSeekChat, GeminiChat, QwenChat
from llm.anthropic_api import AnthropicChat
from llm.sglang_api import SGLangChat


__all__ = [
    "LLMChat",
    "Message",
    "OpenAIChat",
    "AzureChat",
    "DeepInfraChat",
    "DeepSeekChat",
    "GeminiChat",
    "QwenChat",
    "AnthropicChat",
    "SGLangChat",
]

def get_platform(platform: str) -> LLMChat:
    platform = platform.lower()
    if platform == "openai":
        return OpenAIChat
    elif platform == "azure":
        return AzureChat
    elif platform == "deepinfra":
        return DeepInfraChat
    elif platform == "deepseek":
        return DeepSeekChat
    elif platform == "gemini":
        return GeminiChat
    elif platform == "qwen":
        return QwenChat
    elif platform == "anthropic":
        return AnthropicChat
    elif platform == "sglang":
        return SGLangChat
    else:
        raise ValueError(f"Unsupported platform: {platform}")
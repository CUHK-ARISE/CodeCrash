import os
from openai import OpenAI, AzureOpenAI

from llm import OpenAIChat


class AzureChat(OpenAIChat):
    def __init__(self, model_name: str, **kwargs) -> None:
        client = AzureOpenAI(
            api_key=os.getenv(f"AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv(f"AZURE_ENDPOINT"),
            api_version=os.getenv(f"AZURE_VERSION"),
        )
        super().__init__(model_name, client, **kwargs)


class DeepInfraChat(OpenAIChat):
    def __init__(self, model_name: str, **kwargs) -> None:
        client = OpenAI(
            api_key=os.getenv(f"DEEPINFRA_API_KEY"),
            base_url="https://api.deepinfra.com/v1/openai",
        )
        super().__init__(model_name, client, max_n=4, **kwargs)


class DeepSeekChat(OpenAIChat):
    def __init__(self, model_name: str, **kwargs) -> None:
        client = OpenAI(
            api_key=os.getenv(f"DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        super().__init__(model_name, client, max_n=1, **kwargs)


class GeminiChat(OpenAIChat):
    def __init__(self, model_name: str, **kwargs) -> None:
        client = OpenAI(
            api_key=os.getenv(f"GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        super().__init__(model_name, client, **kwargs)


class QwenChat(OpenAIChat):
    def __init__(self, model_name: str, **kwargs) -> None:
        client = OpenAI(
            api_key=os.getenv(f"QWEN_API_KEY"),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        super().__init__(model_name, client, **kwargs)

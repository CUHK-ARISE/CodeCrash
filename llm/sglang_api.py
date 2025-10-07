from sglang import OpenAI
from llm.openai_api import OpenAIChat

class SGLangChat(OpenAIChat):
    def __init__(self, model_name: str, **kwargs) -> None:
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:30000/v1"
        )
        super().__init__(model_name, client, **kwargs)

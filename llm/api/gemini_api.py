import os
import time
from typing import List, Union
from tenacity import (retry, stop_after_attempt, wait_random_exponential)

import google.generativeai as genai
from llm import LLMChat, Message

class GeminiChat(LLMChat):
    client = genai.configure(
        api_key=os.getenv(f"GOOGLE_API_KEY")
    )
    
    def __init__(self, model_name:str, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
        print(os.getenv(f"GOOGLE_API_KEY"))

    def get_msg(self, messages: List[Message]) -> List[dict]:
        return [msg.to_gemini_format() for msg in messages]
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def chat(self, messages: List[Message], n: int = 1) -> Union[List[str], str]:
        
        input_prompt = self.get_msg(messages)
        self.write_records(messages[-1].content, title="INPUT")
        
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(
            input_prompt,
            generation_config=genai.types.GenerationConfig(
                **self.get_gemini_conf(n=n)
            )
        )
        
        time.sleep(self.delay)
        return [
            c.content.parts[0].text 
            for c in response.candidates 
            if c.content and c.content.parts and len(c.content.parts) > 0
        ]
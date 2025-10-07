import os
import time
from typing import List, Optional

from openai import OpenAI
from llm import LLMChat, Message


class OpenAIChat(LLMChat):
    def __init__(self, model_name: str, client: Optional[OpenAI] = None, **kwargs) -> None:
        if client is None:
            openai_client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        self.client = client or openai_client
        super().__init__(model_name, **kwargs)
    
    def get_msg(self, messages: List[Message]) -> List[dict]:
        message_list = [msg.to_openai_format() for msg in messages]
        return message_list
    
    def chat_generic(self, input_prompt: List[dict], n: int) -> List[str]:
        responses = self.client.chat.completions.create(
            model=self.model_name,
            messages=input_prompt,
            **self.get_openai_conf(n=n)
        )
        self.write_records(responses.choices[0].message.content, title="RESPONSE")
        time.sleep(self.delay)        
        return [c.message.content for c in responses.choices]
    
    
    def chat_streaming(self, input_prompt: List[dict], n: int, stream_print: bool) -> List[str]:
        response_stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=input_prompt,
            stream=True,
            **self.get_openai_conf(n=n),
        )
        
        response_content = ""
        reasoning_content = ""
        
        for chunk in response_stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content != None:
                reasoning_content += delta.reasoning_content
                if stream_print:
                    print(delta.reasoning_content, end="", flush=True)
            else:
                response_content += delta.content
                if stream_print:
                    print(delta.content, end="", flush=True)
            
        self.write_records(reasoning_content, title="RESPONSE")
        time.sleep(self.delay)
        
        if reasoning_content == "":
            return [response_content]
        else:
            return [f"<think>{reasoning_content}</think>\n\n{response_content}"]
    
    
    def chat(self, messages: List[Message], n: int = 1, stream_print: bool = False) -> List[str]:
        input_prompt = self.get_msg(messages)
        self.write_records(messages[-1].content, title="INPUT")
        if self.stream:
            return self.chat_streaming(input_prompt, n, stream_print)
        else:
            return self.chat_generic(input_prompt, n)

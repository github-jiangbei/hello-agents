# my_llm.py
import os
from typing import Any, Iterator, Optional
from openai import OpenAI
from hello_agents import HelloAgentsLLM

class MyLLM(HelloAgentsLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs
    ):
        # 检查provider是否为我们想处理的'modelscope'
        if provider == "modelscope":
            print("正在使用自定义的 ModelScope Provider")
            self.provider = "modelscope"
            
            # 解析 ModelScope 的凭证
            self.api_key = api_key or os.getenv("MODELSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
            self.base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api-inference.modelscope.cn/v1/"
            
            # 验证凭证是否存在
            if not self.api_key:
                raise ValueError(
                    "ModelScope API key not found. Please set MODELSCOPE_API_KEY or LLM_API_KEY."
                )

            # 设置默认模型和其他参数
            self.model = model or os.getenv("LLM_MODEL_ID") or "Qwen/Qwen3.5-35B-A3B"
            self.temperature = kwargs.get('temperature', 0.7)
            self.max_tokens = kwargs.get('max_tokens')
            self.timeout = kwargs.get('timeout', 60)
            self.kwargs = kwargs
            
            # 使用获取的参数创建OpenAI客户端实例
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

        else:
            # 如果不是 modelscope, 则完全使用父类的原始逻辑来处理
            super().__init__(model=model, api_key=api_key, base_url=base_url, provider=provider, **kwargs)

    def think(self, messages: list[dict[str, str]], temperature: Optional[float] = None) -> Iterator[str]:
        """流式调用 ModelScope，跳过结束时的空 choices chunk。"""
        if self.provider != "modelscope":
            yield from super().think(messages, temperature)
            return

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        for chunk in response:
            content = self._extract_stream_content(chunk)
            if content:
                yield content

    def stream_invoke(self, messages: list[dict[str, str]], **kwargs) -> Iterator[str]:
        """流式调用LLM，保持和 HelloAgentsLLM.stream_invoke 一致的接口。"""
        temperature = kwargs.get("temperature")
        yield from self.think(messages, temperature)

    @classmethod
    def _extract_stream_content(cls, chunk: Any) -> str:
        choices = cls._get_value(chunk, "choices") or []
        if not choices:
            return ""

        delta = cls._get_value(choices[0], "delta")
        if not delta:
            return ""

        content = cls._get_value(delta, "content")
        return content or ""

    @staticmethod
    def _get_value(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

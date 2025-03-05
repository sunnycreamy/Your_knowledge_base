#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Any, List, Mapping, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import zhipuai
from dotenv import load_dotenv
import os

# 继承自 langchain.llms.base.LLM
class ZhipuAILLM(LLM):
    # 默认选用 glm-4
    model: str = "glm-4"
    # 温度系数
    temperature: float = 0.7
    # API_Key
    api_key: Optional[str] = None
    
    def __init__(self, model: str = "glm-4", temperature: float = 0.7, api_key: Optional[str] = None):
        super().__init__()
        # 加载环境变量
        load_dotenv()
        
        # 设置 API key
        self.api_key = api_key or os.getenv('ZHIPUAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided either through argument or environment variable.")
        
        zhipuai.api_key = self.api_key
        self.model = model
        self.temperature = temperature
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        try:
            response = zhipuai.ZhipuAI().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling ZhipuAI API: {e}")
            return f"抱歉，调用 API 时出现错误：{str(e)}"

    # 首先定义一个返回默认参数的方法
    @property
    def _default_params(self) -> Dict[str, Any]:
        """获取调用API的默认参数。"""
        normal_params = {
            "temperature": self.temperature,
            }
        # print(type(self.model_kwargs))
        return {**normal_params}

    @property
    def _llm_type(self) -> str:
        return "zhipuai"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}

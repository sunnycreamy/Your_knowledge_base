from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional


from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator
import zhipuai
import os
from dotenv import load_dotenv
from pydantic import Field, validator

logger = logging.getLogger(__name__)

class ZhipuAIEmbeddings(Embeddings):
    """智谱 AI Embeddings 封装类"""
    
    def __init__(self, api_key: str = None):
        """初始化函数"""
        # 加载环境变量
        load_dotenv()
        
        # 设置 API key
        self.api_key = api_key or os.getenv('ZHIPUAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "ZHIPUAI_API_KEY 未设置。请在环境变量或 .env 文件中设置。"
            )
        
        # 初始化 ZhipuAI 客户端
        self.client = zhipuai.ZhipuAI(api_key=self.api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将文档转换为向量"""
        try:
            embeddings = []
            for text in texts:
                response = self.client.embeddings.create(
                    model="text_embedding",
                    input=text
                )
                embeddings.append(response.data[0].embedding)
            return embeddings
        except Exception as e:
            print(f"Embedding documents error: {e}")
            return []

    def embed_query(self, text: str) -> List[float]:
        """将查询转换为向量"""
        try:
            response = self.client.embeddings.create(
                model="text_embedding",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding query error: {e}")
            return []
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError("Please use `embed_documents`. Official does not support asynchronous requests")

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError("Please use `aembed_query`. Official does not support asynchronous requests")
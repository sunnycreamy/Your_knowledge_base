import requests
from models.llm.zhipuai_llm import ZhipuAILLM
from config.config import OLLAMA_URL, TEMPERATURE, AVAILABLE_MODELS, EMBEDDING_MODEL
import streamlit as st
# 直接从 langchain_ollama 导入需要的组件
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
# 从 zhipuai 直接导入
try:
    from zhipuai import ZhipuAI as ChatZhipuAI
except ImportError:
    try:
        from zhipuai import ChatZhipuAI
    except ImportError:
        ChatZhipuAI = None

import logging

logger = logging.getLogger(__name__)

def get_llm(selected_model, api_key=None):
    """获取 LLM 实例"""
    logger.info(f"初始化LLM，选择的模型: {selected_model}")
    
    try:
        if selected_model == "智谱GLM4":
            if not api_key:
                logger.error("使用智谱GLM4模型需要提供API Key")
                return None
            logger.info("使用智谱GLM4模型")
            llm = ZhipuAILLM(
                model="glm-4-flash",
                temperature=TEMPERATURE,
                api_key=api_key
            )
        else:
            if not check_ollama_status():
                logger.error("Ollama服务未运行")
                return None
            
            logger.info(f"使用Ollama模型: {selected_model}")
            llm = ChatOllama(
                base_url=OLLAMA_URL,
                model=selected_model,
                temperature=TEMPERATURE
            )
            
        # 验证 LLM 实例
        if llm is None:
            logger.error("LLM 实例创建失败")
            return None
            
        logger.info(f"成功创建 LLM 实例: {type(llm)}")
        return llm
        
    except Exception as e:
        logger.error(f"初始化 LLM 时发生错误: {str(e)}", exc_info=True)
        return None

def check_ollama_status():
    """检查Ollama服务状态"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        status = response.status_code == 200
        print(f"DEBUG: Ollama status check: {status}")  # 调试信息
        return status
    except Exception as e:
        print(f"DEBUG: Ollama status check error: {e}")  # 调试信息
        return False

def get_available_ollama_models():
    """获取可用的Ollama模型列表"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            print(f"DEBUG: Available Ollama models: {model_names}")  # 调试信息
            return model_names
        return []
    except Exception as e:
        print(f"DEBUG: Error getting Ollama models: {e}")  # 调试信息
        return []

def get_embedding_model():
    """获取 BGE-M3 嵌入模型"""
    try:
        return OllamaEmbeddings(
            base_url=OLLAMA_URL,
            model=EMBEDDING_MODEL  # 使用配置中指定的模型
        )
    except Exception as e:
        st.error(f"初始化 BGE-M3 Embeddings 失败: {str(e)}")
        return None 
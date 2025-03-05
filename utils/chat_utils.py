import streamlit as st
from utils.imports import (
    ConversationalRetrievalChain,
    Ollama,
    ChatZhipuAI
)
from config.config import OLLAMA_URL

def get_llm(model_name, api_key=None):
    """
    获取语言模型实例
    
    Args:
        model_name (str): 模型名称
        api_key (str, optional): API密钥
    
    Returns:
        BaseLLM: 语言模型实例
    """
    try:
        if model_name == "智谱GLM4":
            if not api_key:
                st.error("使用智谱GLM4需要提供API Key")
                return None
            return ChatZhipuAI(
                model_name="glm-4",
                temperature=0.7,
                api_key=api_key
            )
        else:
            return Ollama(model=model_name,
            base_url=OLLAMA_URL,
            temperature=0.7
            )
    except Exception as e:
        st.error(f"初始化模型失败: {str(e)}")
        return None

def get_chat_qa_chain(llm, vectorstore):
    """
    创建问答链
    
    Args:
        llm (BaseLLM): 语言模型实例
        vectorstore (VectorStore): 向量存储实例
    
    Returns:
        ConversationalRetrievalChain: 问答链实例
    """
    if not llm:
        st.error("LLM 模型未初始化")
        return None
        
    if not vectorstore:
        st.error("向量存储未初始化")
        return None

    try:
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
            verbose=True
        )
    except Exception as e:
        st.error(f"创建问答链失败: {str(e)}")
        return None 
from utils.imports import (
    ConversationalRetrievalChain,
    ConversationBufferMemory
)
import streamlit as st
from utils.api_handler import APIHandler
from utils.vectordb_utils import get_vectordb, rebuild_vectordb_for_files
from utils.model_utils import get_embedding_model
from config.config import VECTOR_DB_PATH
import logging
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

logger = logging.getLogger(__name__)

# 定义系统提示和人类提示模板
SYSTEM_TEMPLATE = """你是一个专业的助手。使用以下上下文来回答用户的问题。如果你不知道答案，就说你不知道，不要试图编造答案。

上下文: {context}

历史对话: {chat_history}
"""

HUMAN_TEMPLATE = """问题: {question}"""

# 创建聊天提示模板
CHAT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)
])

def initialize_chat_history():
    """
    初始化或获取聊天历史
    
    Returns:
        list: 聊天历史列表
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    return st.session_state.messages

def get_chat_qa_chain(llm, vectordb):
    """
    创建聊天问答链
    
    Args:
        llm: 语言模型实例
        vectordb: 向量数据库实例
    
    Returns:
        ConversationalRetrievalChain: 问答链实例
    """
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    
    # 修改检索参数
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3
        }
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        return_generated_question=False,
        combine_docs_chain_kwargs={"prompt": CHAT_PROMPT}
    )

def generate_response(user_input: str, qa_chain) -> tuple[str, list]:
    """生成回答"""
    try:
        # 使用 qa_chain 处理问题
        response = qa_chain({"question": user_input})
        
        # 从响应中提取答案和来源文档
        answer = response.get('answer', '')
        source_docs = response.get('source_documents', [])
        
        return answer, source_docs
        
    except Exception as e:
        logger.error(f"处理问题时出错: {str(e)}", exc_info=True)
        return f"处理问题时出错: {str(e)}", []

def clean_response(response) -> str:
    """清理模型响应"""
    try:
        logger.info(f"原始响应: {response}")  # 添加日志
        
        if isinstance(response, str):
            return response.strip()
            
        # 如果是 ChatZhipuAI 的响应
        if hasattr(response, 'content'):
            return response.content.strip()
            
        # 如果是 ChatOllama 的响应
        if hasattr(response, 'message'):
            return response.message.content.strip()
            
        logger.error(f"未知的响应类型: {type(response)}")
        return str(response)
        
    except Exception as e:
        logger.error(f"清理响应时发生错误: {str(e)}", exc_info=True)
        return f"处理响应时出错: {str(e)}"

def display_chat_message(message, is_user=False):
    """
    显示聊天消息
    
    Args:
        message (str): 消息内容
        is_user (bool): 是否为用户消息
    """
    avatar = "🧑‍💻" if is_user else "🤖"
    with st.chat_message(name="user" if is_user else "assistant", avatar=avatar):
        st.markdown(message)

def display_source_documents(documents):
    """
    显示来源文档
    
    Args:
        documents (list): 来源文档列表
    """
    if documents:
        st.markdown("### 参考来源")
        
        # 使用 tabs 代替 expander
        tabs = st.tabs([f"来源 {i+1}" for i in range(len(documents))])
        
        for i, (tab, doc) in enumerate(zip(tabs, documents)):
            with tab:
                st.markdown(doc.page_content)
                # if hasattr(doc.metadata, 'source') and doc.metadata.get('source'):
                #     st.caption(f"来源: {doc.metadata['source']}") 
                if doc.metadata and 'source' in doc.metadata:
                    st.caption(f"来源: {doc.metadata['source']}")
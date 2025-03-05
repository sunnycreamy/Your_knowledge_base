import streamlit as st
from config.config import APP_TITLE, APP_ICON, LAYOUT, INITIAL_SIDEBAR_STATE

# 必须是第一个 Streamlit 命令
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE
)

# 在页面顶部添加标题
st.title(APP_TITLE)

# 设置 Pydantic 配置
from typing import Any, Dict
import pydantic
from pydantic import BaseModel

# 设置全局 Pydantic 配置
pydantic.config.ConfigDict.arbitrary_types_allowed = True

# 添加基础配置类
class BaseConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            # 添加自定义的 JSON 编码器（如果需要）
        }

# 确保所有使用 Pydantic 的类都继承这个配置
class Config(BaseConfig):
    pass

# 标准库
import os
import json
import logging
import shutil
from pathlib import Path
from contextlib import contextmanager


# 事件循环处理
import asyncio
from functools import partial
from concurrent.futures import ThreadPoolExecutor

# 创建线程池执行器
thread_pool = ThreadPoolExecutor(max_workers=4)

# 确保只有一个事件循环
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# 设置线程池执行器
loop.set_default_executor(thread_pool)  # 使用显式的线程池执行器

# 防止 Streamlit 创建新的事件循环
def run_async(coro):
    """运行异步代码的包装器"""
    try:
        return loop.run_until_complete(coro)
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            # 如果循环已关闭，创建新的循环
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(coro)
        raise


# ========================== #
#       第三方库导入
# ========================== #
# 第三方库
import langchain
langchain.verbose = False

# 适配新版本 langchain 结构
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain.chains import ConversationalRetrievalChain
except ImportError:
    from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain


from langchain_community.vectorstores import Chroma as ChromaDB

try:
    from langchain_community.embeddings.ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings

# 文档加载器
try:
    from langchain_community.document_loaders.pdf import PyPDFLoader
except ImportError:
    from langchain_community.document_loaders import PyPDFLoader

from langchain.schema import (
    Document,
    HumanMessage,
    AIMessage
)

# 配置模块
try:
    from config.config import (
        VECTOR_DB_PATH,
        APP_TITLE,
        APP_ICON,
        AVAILABLE_MODELS,
        DEFAULT_MODEL,
        KNOWLEDGE_BASE_PATH
    )
except ModuleNotFoundError:
    raise ImportError("⚠️ 请确保 config/config.py 存在")

## 🌟 组件相关
from components.ui import (
    setup_sidebar,
    file_selector,
    show_file_upload,
    show_system_status,
    display_chat_message 
)
from components.chat import (
    initialize_chat_history,
    get_chat_qa_chain,
    generate_response,
    display_source_documents
)
from components.file_manager import (
    create_file_manager,
    initialize_file_manager,
    show_file_manager_dialog
)

## 🌟 工具类
from utils.api_handler import APIHandler
from utils.model_utils import get_llm, get_embedding_model
from utils.file_utils import load_documents, get_knowledge_base_files
from utils.manager_utils import ensure_knowledge_base_structure
from utils.vectordb_utils import (
    add_documents_to_vectordb,
    rebuild_vectordb_for_files,
    get_vectordb
)
from utils.logger import logger
# 数据库连接
try:
    from services.database import SessionLocal, Base, engine
except ModuleNotFoundError:
    raise ImportError("⚠️ `services/database.py` 可能不存在，请检查文件路径")
from services.google_drive_service import GoogleDriveService


# 创建数据库表
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

def init_drive_service():
    """初始化 Google Drive 服务"""
    if 'drive_service' not in st.session_state:
        st.session_state.drive_service = GoogleDriveService()

def show_vector_store_management():
    st.header("向量库管理")
    
    # 增量添加文档到向量库
    st.button("增量添加文档到向量库")
    
    # 向量库维护说明
    with st.expander("向量库维护说明"):
        st.write("""
        1. 向量库用于存储文档的语义表示
        2. 支持增量添加新文档
        3. 定期维护可确保最佳性能
        """)

def show_feature_selection():
    st.header("功能选择")
    
    # Google Drive同步选项
    drive_sync = st.checkbox("Google Drive 同步")
    
    if drive_sync:
        show_drive_sync()

def show_drive_sync():
    st.subheader("Google Drive 同步")
    
    init_drive_service()
    
    if st.button("同步 Google Drive"):
        try:
            # 确保已认证
            st.session_state.drive_service.authenticate()
            
            # 执行同步
            success = st.session_state.drive_service.sync_drive_files()
            
            if success:
                st.success("同步完成！")
                # 自动更新向量库
                with st.spinner("正在更新向量库..."):
                    
                    try:
                        # 使用已经验证可用的函数获取文件
                        files = get_knowledge_base_files()
                        if files:
                            vectordb = add_documents_to_vectordb(files)
                            if vectordb:
                                st.success("向量库更新成功！")
                            else:
                                st.error("向量库更新失败")
                        else:
                            st.warning("没有找到需要处理的文件")
                    except Exception as e:
                        st.error(f"更新向量库时出错: {str(e)}")
        except Exception as e:
            st.error(f"同步失败: {str(e)}")

def initialize_session_state():
    """初始化会话状态"""
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_file_manager" not in st.session_state:
        st.session_state.show_file_manager = False
    if "selected_docs_vectordb" not in st.session_state:
        st.session_state.selected_docs_vectordb = None

def clean_response(response):
    """清理响应文本，去除不需要的符号和元数据"""
    if isinstance(response, dict):
        content = response.get("answer") or response.get("content") or str(response)
    else:
        content = str(response)
    
    # 清理响应文本
    if "additional_kwargs=" in content:
        content = content.split("additional_kwargs=")[0]  # 移除元数据部分
    
    # 移除 think 标签
    content = content.replace("content='<think>", "")
    # content = content.replace("</think>", "")
    content = content.replace("<think>", "")
    content=content.replace("content='", "")
    content = content.replace("\\n", "\n")  # 处理换行符
    content = content.strip("'\" ")  # 移除首尾的引号和空格
    return content

def clear_text():
    """清空输入框的回调函数"""
    if st.session_state.user_input:
        st.session_state.user_question = st.session_state.user_input
        st.session_state.user_input = ""  # 只清空输入框，不清空问题

def generate_response(llm, prompt, search_results=None, source_type=None):
    """生成回答"""
    try:
        response = llm.invoke(prompt)
        if response is None:
            return "⚠️ AI 没有返回结果，请稍后再试", [], source_type
        
        # 如果提供了搜索结果，将其转换为原文档格式
        source_documents = []
        
        if search_results and 'google' in search_results and 'data' in search_results['google']:
            for item in search_results['google']['data'][:5]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                
                # 创建源文档对象
                source_documents.append(
                    Document(
                        page_content=f"📌 {title}\n📄 摘要：{snippet}\n🔗 来源：{link}\n",
                        metadata={"source": link, "title": title}
                    )
                )
        
        return clean_response(response), source_documents, source_type
    except Exception as e:
        logger.error(f"生成响应时出错: {str(e)}", exc_info=True)
        return f"⚠️ 发生错误，无法生成回答: {str(e)}", [], source_type

def format_session_state():
    """格式化 session_state 为易读的字符串格式"""
    try:
        
        formatted_lines = []
        for key, value in st.session_state.items():
            try:
                if isinstance(value, (str, int, float, bool, type(None))):
                    formatted_value = str(value)
                elif hasattr(value, '__class__'):
                    class_name = value.__class__.__name__
                    if class_name == 'UploadedFile':
                        formatted_value = f"UploadedFile(name={value.name})"
                    else:
                        formatted_value = f"{class_name}({str(value)})"
                else:
                    formatted_value = str(value)
                
                formatted_lines.append(f"{key}: {formatted_value}")
            except Exception as e:
                formatted_lines.append(f"{key}: <错误: {str(e)}>")
        
        return "\n".join(formatted_lines)
    except Exception as e:
        return f"格式化 session state 时出错: {str(e)}"

# 创建一个用户友好的状态显示
def show_user_friendly_status(message=None):
    # 创建一个状态容器
    if "status_container" not in st.session_state:
        st.session_state.status_container = st.empty()
    
    # 如果有消息，显示一个友好的加载指示器
    if message:
        with st.session_state.status_container:
            st.markdown(f"<div class='stProgress'><div class='stProgressIndicator'>⏳</div> {message}</div>", 
                       unsafe_allow_html=True)
    else:
        # 清空状态容器
        st.session_state.status_container.empty()
        
# show_user_friendly_status('正在加载...')
# show_user_friendly_status('加载完成')


def handle_free_chat(llm, user_question, chat_history):
    """处理自由对话的主要逻辑"""
    logger.info("开始自由对话模式处理")

    # 获取格式化的对话历史
    formatted_history = format_chat_history(chat_history)

    # **第一步：LLM 直接回答**
    answer, source_documents = get_llm_direct_response(llm, user_question, formatted_history)

    # **第二步：检查 LLM 是否确定答案**
    if should_perform_search(user_question, answer):
        try:
            logger.info(f"🔍 触发搜索，查询：{user_question}")

            # **执行搜索**
            search_results = perform_web_search(user_question)

            # **检查搜索是否成功**
            if search_results:
                logger.info(f"✅ 成功获取搜索结果: {search_results}")

                # **如果 LLM 先回答了内容，把 LLM 的回答也放进 Prompt**
                search_prompt = f"""
                {formatted_history}
                
                【当前问题】
                {user_question}

                【AI初步回答】（如果可信，可以参考）
                {answer}

                🔍【最新搜索结果】：
                {format_search_results(search_results)}

                📋 【回答要求】：
                1. **你的回答必须严格基于搜索结果**
                2. **你不能使用 LLM 内部知识，除非搜索结果不足**
                3. **请直接引用搜索结果中的最新信息，并提供来源**
                4. **如果搜索结果提到具体时间和人物，你必须使用这些信息**
                """

                # **让 LLM 结合搜索信息重新回答**
                new_answer, new_source_docs = generate_response(llm, search_prompt, search_results, "web_search")[:2]
                
                if new_answer:
                    answer = new_answer
                    source_documents = new_source_docs

        except Exception as e:
            logger.error(f"❌ 搜索增强回答失败: {str(e)}", exc_info=True)
            # **搜索失败时，仍然使用 LLM 回答**
    
    return answer, source_documents

def format_chat_history(chat_history):
    """格式化对话历史"""
    if not chat_history:
        return ""
    
    history_messages = []
    for msg in chat_history[-5:]:  # 只保留最近5条对话
        prefix = "用户：" if msg['role'] == "user" else "AI："
        history_messages.append(f"{prefix}{msg['content']}")
    
    return "【历史对话】\n" + "\n".join(history_messages) + "\n"

def get_llm_direct_response(llm, user_question, formatted_history):
    """获取LLM的直接回答"""
    logger.info("尝试 LLM 直接回答")
    
    prompt = f"""
    {formatted_history}
    
    【当前问题】
    {user_question}

    📋 【回答要求】：
    1. **如果你能100%确定答案，请直接回答**
    2. **如果问题涉及最新信息（如“现在”、“今年”、“最近”），请直接说："请查询最新数据"**
    3. **如果你不确定答案，请直接说："我不确定，请查询最新信息"**
    4. **不要编造信息**
    """
    
    answer, source_documents = generate_response(llm, prompt, None, "direct")[:2]

    # ✅ 记录 LLM 的回答，以便调试
    logger.info(f"💬 LLM 直接回答: {answer}")

    return answer, source_documents

def should_perform_search(user_question, llm_answer):
    """判断是否需要进行搜索"""
    # 时效性关键词
    time_keywords = [
        "最新", "现在", "今天", "最近", "目前",
        "2025", "明年", "今年", "上个月", "这个月"
    ]
    
    # 需要实时数据的主题
    topic_keywords = [
        "股票", "基金", "房价", "油价", "汇率",
        "天气", "疫情", "新闻", "政策", "行情",
        "价格", "上市", "发布", "更新", "公告"
    ]
    
    # LLM 表示不确定的标志
    uncertainty_signals = [
        "我不确定", "需要查询", "需要核实",
        "最新信息", "实时数据", "建议查询"
    ]
    
    needs_search = (
        any(keyword in user_question for keyword in time_keywords) or
        any(keyword in user_question for keyword in topic_keywords) or
        any(signal in llm_answer for signal in uncertainty_signals)
    )
    
    if needs_search:
        logger.info(f"🔍 触发搜索原因: 问题或回答中包含需要实时信息的关键词")
    
    return needs_search

def perform_web_search(query):
    """执行网络搜索"""
    try:
        api_handler = APIHandler()  # 确保 APIHandler 已正确配置
        search_results = api_handler.search_web(query)
        
        if not search_results or 'google' not in search_results or 'data' not in search_results['google']:
            logger.warning("搜索返回数据格式错误或无结果")
            return None

        filtered_results = search_results['google']['data'][:5]
        if not filtered_results:
            logger.warning("搜索结果为空")

        # ✅ 打印日志，检查搜索结果是否有效
        logger.info(f"🔍 搜索结果：{filtered_results}")
        
        return filtered_results
            
    except Exception as e:
        logger.error(f"网络搜索失败: {str(e)}", exc_info=True)
        return None

def format_search_results(search_results):
    """格式化搜索结果，确保 LLM 能正确解析"""
    if not search_results:
        return "⚠️ 未找到相关搜索结果"
        
    formatted_results = []
    
    for item in search_results:
        title = item.get("title", "未知标题")
        snippet = item.get("snippet", "暂无摘要")
        link = item.get("link", "#")
        
        formatted_results.append(f"""
📌 **{title}**
📝 **摘要**：{snippet}
🔗 **来源**：[{link}]({link})
""")
    
    return "\n".join(formatted_results)

def main():
    initialize_session_state()
    
    # 调试信息
    # logger.debug(f"Session State:\n{format_session_state()}")
    
    # 检查关键变量
    if "selected_model" not in st.session_state:
        st.error("未初始化 `selected_model`，正在使用默认值")
        st.session_state.selected_model = DEFAULT_MODEL
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
        
    # 设置侧边栏
    selected_model, api_key = setup_sidebar()
    
   # 检查Google Drive同步是否被选中
    if hasattr(st.session_state, 'drive_sync_checked') and st.session_state.drive_sync_checked:
        # 使用占位符显示Google Drive同步内容
        with st.session_state.drive_sync_content.container():
            show_drive_sync()
            
            st.markdown("---")  # 添加分隔线
        
    
    # 聊天模式选择
    chat_mode = st.radio(
        "选择聊天模式",
        ["知识库对话", "文档对话", "自由对话"],
        key="chat_mode"
    )
    
    # 文档对话模式的文件选择
    if chat_mode == "文档对话":
        st.info("请选择要对话的文档")
        # 获取已上传的文件列表
        available_files = get_knowledge_base_files()
        
        if not available_files:
            st.warning("暂无可用文档，请先在侧边栏上传文档")
        else:
            selected_files = st.multiselect(
                "选择要对话的文档（可多选）",
                options=available_files,
                format_func=lambda x: os.path.basename(x)
            )
            
            if selected_files:
                with st.spinner("准备文档对话..."):
                    try:
                        # 使用永久向量库而不是创建临时向量库
                        if not VECTOR_DB_PATH.exists():
                            st.error("向量库路径不存在，请先构建向量库")
                            st.stop()
                        
                        # 加载永久向量库
                        vectordb = ChromaDB(
                            persist_directory=str(VECTOR_DB_PATH),
                            embedding_function=get_embedding_model()
                        )
                        
                        # 创建过滤器，只检索选定的文档
                        file_filter = {"source": {"$in": selected_files}}
                        
                        # 使用过滤器创建检索器
                        retriever = vectordb.as_retriever(
                            search_kwargs={
                                "k": 5,
                                "filter": file_filter  # 应用过滤器
                            }
                        )
                        
                        # 存储检索器而不是整个向量库
                        st.session_state.selected_docs_retriever = retriever
                        
                        # st.success(f"成功准备 {len(selected_files)} 个文档！")
                            
                    except Exception as e:
                        st.error(f"准备文档时出错: {str(e)}")
                        st.session_state.selected_docs_retriever = None
    
    # 对话区域
    st.markdown("### 对话区域")
    
    if chat_mode == "自由对话":
        st.info("在这里，您可以与AI助手进行自由对话，不受知识库限制。")
    
    # 创建对话容器和输入区域
    chat_container = st.container()
    with st.container():
        user_input = st.text_input("请输入您的问题", key="user_input", on_change=clear_text)
    
    # 显示对话历史
    with chat_container:
        for message in st.session_state.get("chat_history", []):
            if message["role"] == "user":
                display_chat_message("user", message["content"])
            else:
                display_chat_message("assistant", message["content"])
                if "source_documents" in message:
                    display_source_documents(message["source_documents"])
            st.markdown("---")
    
    # 处理用户问题
    if st.session_state.user_question:
        try:
            with st.spinner():
                llm = get_llm(st.session_state.selected_model, st.session_state.api_key)
                if not llm:
                    st.error("模型初始化失败，无法加载 LLM")
                    st.stop()
                
                if chat_mode == "自由对话":
                    try:
                        answer, source_documents = handle_free_chat(
                            llm,
                            st.session_state.user_question,
                            st.session_state.get("chat_history", [])
                        )
                    except Exception as e:
                        logger.error(f"自由对话处理失败: {str(e)}", exc_info=True)
                        answer = "抱歉，我现在无法正确处理您的问题，请稍后再试"
                        source_documents = []
                        
                elif chat_mode == "知识库对话":
                    try:
                        if not VECTOR_DB_PATH.exists():
                            st.error("知识库路径不存在")
                            st.stop()
                            
                        # 尝试使用内存模式作为备选方案
                        try:
                            # 先尝试加载持久化向量库
                            vectordb = ChromaDB(
                                persist_directory=str(VECTOR_DB_PATH),
                                embedding_function=get_embedding_model()
                            )
                        except Exception as e:
                            st.warning(f"加载持久化向量库失败: {str(e)}")
                            logger.info("正在使用内存模式作为备选方案...")
                            
                            # 获取所有文档
                            files = get_knowledge_base_files()
                            if not files:
                                st.warning("知识库中没有文件")
                                st.stop()
                                
                            # 加载文档
                            documents = load_documents(files)
                            
                            # 使用内存模式
                            client = chromadb.Client()
                            vectordb = ChromaDB(
                                client=client,
                                collection_name="knowledge_base",
                                embedding_function=get_embedding_model()
                            )
                            
                            # 添加文档
                            vectordb.add_documents(documents)
                            
                        # 创建问答链
                        qa_chain = get_chat_qa_chain(llm, vectordb)
                        response = qa_chain.invoke({"question": st.session_state.user_question})
                        answer = response['answer']
                        source_documents = response.get('source_documents', [])
                    except Exception as e:
                        logger.error(f"处理知识库对话时出错: {str(e)}", exc_info=True)
                        st.error(f"处理问题时出错: {str(e)}")
                        st.stop()
                else:  # 文档对话模式
                    if not st.session_state.selected_docs_retriever:
                        st.warning("请先选择文档")
                        st.stop()
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=st.session_state.selected_docs_retriever,
                        memory=ConversationBufferMemory(
                            memory_key="chat_history",
                            return_messages=True
                        )
                    )
                    response = qa_chain({"question": st.session_state.user_question})
                    answer = response['answer']
                    source_documents = response.get('source_documents', [])

                # 更新对话历史
                st.session_state.chat_history.extend([
                    {"role": "user", "content": st.session_state.user_question},
                    {"role": "assistant", "content": answer, "source_documents": source_documents}
                ])
                logger.info("对话历史已更新")
                
                # 清空用户问题
                st.session_state.user_question = ""
                
                # 刷新页面
                st.rerun()

        except Exception as e:
            error_msg = f"处理问题时出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)
            st.session_state.user_question = ""


if __name__ == "__main__":
    # 确保日志目录存在
    os.makedirs("logs", exist_ok=True)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/app.log"),
            # 如果你不想在控制台看到日志，可以注释掉下面这行
            logging.StreamHandler()
        ]
    )

    # 获取logger
    logger = logging.getLogger("knowledge_base")

    main() 
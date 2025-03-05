from utils.text_splitter import get_text_splitter
from utils.file_utils import load_documents, get_knowledge_base_files
from utils.model_utils import get_embedding_model
from config.config import VECTOR_DB_PATH
import os
import streamlit as st
import chromadb
from chromadb.config import Settings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.config import (
    OLLAMA_URL, 
    # CHUNK_SIZE, 
    # CHUNK_OVERLAP, 
    TEXT_SEPARATORS,
    EMBEDDING_MODEL
)
from utils.imports import (
    OllamaEmbeddings
)
from pathlib import Path
from langchain_chroma import Chroma as ChromaDB
import shutil
import time
import tempfile

# 全局设置
CHROMA_SETTINGS = Settings(anonymized_telemetry=False)


def ensure_vector_db_structure():
    """确保向量库目录结构存在"""
    try:
        # 创建主目录
        vector_db_path = Path(VECTOR_DB_PATH)
        vector_db_path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        st.error(f"创建向量库目录结构失败: {str(e)}")
        return False


def get_vectordb():
    """获取向量库实例"""
    try:
        # 确保目录结构存在
        if not ensure_vector_db_structure():
            return None
            
        # 如果session_state中已有vectordb，直接返回
        if 'vectordb' in st.session_state:
            return st.session_state.vectordb
            
        # 检查向量库是否存在
        vector_db_path = Path(VECTOR_DB_PATH)
        if not vector_db_path.exists() or not any(vector_db_path.iterdir()):
            st.warning("向量库不存在，请先上传文档")
            return None
            
        # 获取嵌入模型
        embedding_model = get_embedding_model()
        if not embedding_model:
            st.error("初始化嵌入模型失败")
            return None
            
        try:
            # 尝试连接现有向量库
            client = chromadb.PersistentClient(
                path=str(vector_db_path),
                settings=CHROMA_SETTINGS
            )
            
            # 获取集合
            collection = client.get_or_create_collection("knowledge_base")
            
            # 使用 LangChain 的 ChromaDB 包装器
            vectordb = ChromaDB(
                client=client,
                collection_name="knowledge_base",
                embedding_function=embedding_model
            )
            
            # 保存到会话状态
            st.session_state.vectordb = vectordb
            return vectordb
            
        except Exception as e:
            # 如果出现兼容性错误，使用内存模式作为备选方案
            st.warning(f"加载持久化向量库失败: {str(e)}")
            logger.info("正在使用内存模式作为临时解决方案...")
            
            # 获取所有文档
            files = get_knowledge_base_files()
            if not files:
                st.warning("知识库中没有文件")
                return None
                
            # 加载文档
            documents = load_documents(files)
            
            # 创建临时目录
            temp_dir = tempfile.mkdtemp()
            
            # 使用临时目录创建持久化客户端
            client = chromadb.PersistentClient(
                path=temp_dir,
                settings=CHROMA_SETTINGS
            )
            
            # 确保集合名称唯一
            collection_name = f"knowledge_base_{int(time.time())}"
            collection = client.get_or_create_collection(collection_name)
            
            vectordb = ChromaDB(
                client=client,
                collection_name=collection_name,
                embedding_function=embedding_model
            )
            
            # 添加文档
            vectordb.add_documents(documents)
            
            # 保存到会话状态
            st.session_state.vectordb = vectordb
            return vectordb
            
    except Exception as e:
        st.error(f"获取向量库失败: {str(e)}")
        return None


def add_documents_to_vectordb(file_paths: list):
    """增量添加文档到向量数据库"""
    try:
        embedding_model = get_embedding_model()
        if not embedding_model:
            st.error("初始化嵌入模型失败")
            return None

        vector_db_path = Path(VECTOR_DB_PATH)
        vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # 使用单个状态组件
        status = st.status("正在处理文档和更新向量库...")
        
        try:
            # 加载或创建向量库
            status.write("正在加载向量库...")
            
            # 使用 PersistentClient 而不是旧的接口
            client = chromadb.PersistentClient(
                path=str(vector_db_path),
                settings=CHROMA_SETTINGS
            )
            
            # 获取或创建集合
            collection = client.get_or_create_collection("knowledge_base")
            
            # 使用 LangChain 的 ChromaDB 包装器
            vectordb = ChromaDB(
                client=client,
                collection_name="knowledge_base",
                embedding_function=embedding_model
            )
            
            # 处理新文档
            for file_path in file_paths:
                # 检查文件类型
                file_path = Path(file_path)
                suffix = file_path.suffix.lower()
                if suffix in ['.pdf', '.docx', '.txt', '.md']:  # 支持多种文件格式
                    status.write(f"正在处理文件: {file_path.name}...")
                    
                    try:
                        # 获取文本分割器
                        text_splitter = get_text_splitter(str(file_path))
                        
                        # 加载和分割文档
                        doc_texts = text_splitter.split_documents(load_documents([str(file_path)]))
                        
                        if doc_texts:
                            # 添加到向量库
                            vectordb.add_documents(doc_texts)
                            status.write(f"✅ 文件 {file_path.name} 已添加 {len(doc_texts)} 个片段")
                        else:
                            status.write(f"⚠️ 文件 {file_path.name} 没有提取到有效内容")
                    except Exception as e:
                        status.write(f"❌ 处理文件 {file_path.name} 失败: {str(e)}")
                else:
                    status.write(f"⚠️ 跳过不支持的文件格式: {file_path.name}")
            
            # 保存到会话状态
            st.session_state.vectordb = vectordb
            
            # 更新状态
            status.update(label="向量库更新完成", state="complete")
            return vectordb

        except Exception as e:
            status.update(label=f"处理文档时出错: {str(e)}", state="error")
            raise e
            
    except Exception as e:
        print(f"更新向量库失败: {str(e)}")
        st.error(f"更新向量库失败: {str(e)}")
        return None


def rebuild_vectordb_for_files(documents: list, embedding_model=None) -> ChromaDB:
    """为选中的文档创建临时向量数据库（使用临时目录）"""
    try:
        # 如果没有提供 embedding_model，则创建新的
        if embedding_model is None:
            embedding_model = get_embedding_model()
            
        if not embedding_model:
            st.error("初始化嵌入模型失败")
            return None
            
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        # 使用临时目录创建持久化客户端
        client = chromadb.PersistentClient(
            path=temp_dir,
            settings=CHROMA_SETTINGS
        )
        
        # 确保集合名称唯一，避免冲突
        collection_name = f"temp_collection_{int(time.time())}"
        collection = client.get_or_create_collection(collection_name)
        
        # 使用 LangChain 的 ChromaDB 包装器
        vectordb = ChromaDB(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_model
        )
        
        # 添加文档
        vectordb.add_documents(documents)
        
        return vectordb
        
    except Exception as e:
        st.error(f"创建临时向量库失败: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


def rebuild_entire_vectordb():
    """完全重建向量库"""
    try:
        # 获取所有知识库文件
        files = get_knowledge_base_files()
        if not files:
            st.warning("知识库中没有文件")
            return False
            
        # 删除旧的向量库
        vector_db_path = Path(VECTOR_DB_PATH)
        if vector_db_path.exists():
            shutil.rmtree(str(vector_db_path))
            
        # 创建新的向量库目录
        vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # 获取嵌入模型
        embedding_model = get_embedding_model()
        if not embedding_model:
            st.error("初始化嵌入模型失败")
            return False
        
        # 加载文档
        documents = load_documents(files)
        if not documents:
            st.warning("没有找到有效文档")
            return False
        
        # 使用新版本 API 创建向量库
        client = chromadb.PersistentClient(
            path=str(vector_db_path),
            settings=CHROMA_SETTINGS
        )
        
        # 创建集合
        collection = client.get_or_create_collection("knowledge_base")
        
        # 使用 LangChain 的 ChromaDB 包装器
        vectordb = ChromaDB(
            client=client,
            collection_name="knowledge_base",
            embedding_function=embedding_model
        )
        
        # 添加文档
        vectordb.add_documents(documents)
        
        # 保存到会话状态
        st.session_state.vectordb = vectordb
        
        st.success(f"成功重建向量库，包含 {len(documents)} 个文档")
        return True
        
    except Exception as e:
        st.error(f"重建向量库失败: {str(e)}")
        return False
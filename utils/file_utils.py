from utils.imports import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
import os
import streamlit as st
from config.config import KNOWLEDGE_BASE_PATH
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def normalize_path(file_path):
    """
    规范化文件路径
    
    Args:
        file_path (str): 原始文件路径
    
    Returns:
        str: 规范化后的文件路径
    """
    return str(Path(file_path).resolve())

def get_file_loader(file_path):
    """
    根据文件类型获取相应的加载器
    
    Args:
        file_path (str): 文件路径
    
    Returns:
        BaseLoader: 文档加载器实例
    """
    # 忽略 .DS_Store 文件
    if file_path.endswith('.DS_Store'):
        return None
        
    # 规范化文件路径
    normalized_path = normalize_path(file_path)
    
    if not os.path.exists(normalized_path):
        st.error(f"文件不存在: {file_path}")
        return None
        
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.pdf':
            return PyPDFLoader(normalized_path)
        elif file_extension in ['.docx', '.doc']:
            return Docx2txtLoader(normalized_path)
        elif file_extension in ['.txt', '.md']:
            return TextLoader(normalized_path)
        else:
            st.warning(f"跳过不支持的文件类型: {file_extension}")
            return None
    except Exception as e:
        st.error(f"创建加载器失败 {file_path}: {str(e)}")
        return None

def load_documents(file_paths: list) -> list:
    """加载文档"""
    documents = []
    for file_path in file_paths:
        try:
            # 确保文件存在
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                continue
                
            ext = os.path.splitext(file_path)[1].lower()
            
            # 根据文件类型选择加载器
            loader = None
            if ext == '.pdf':
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                except Exception as e:
                    logger.warning(f"PDF 加载警告 {file_path}: {str(e)}")
                    continue
            elif ext in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif ext == '.md':
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                print(f"不支持的文件类型: {ext}")
                continue

            if loader is None:
                print(f"无法创建加载器: {file_path}")
                continue

            # 加载文档
            try:
                docs = loader.load()
            except Exception as e:
                print(f"加载文件内容失败 {file_path}: {str(e)}")
                continue

            # 检查加载结果
            if docs is None:
                print(f"警告: {file_path} 没有返回有效内容")
                continue
                
            if not docs:  # 空列表检查
                print(f"警告: {file_path} 内容为空")
                continue

            # 处理文档元数据
            for doc in docs:
                if doc is None:  # 检查单个文档是否为 None
                    continue
                    
                try:
                    doc.metadata['source_file'] = os.path.basename(file_path)
                    doc.metadata['file_type'] = ext
                    documents.append(doc)  # 单个添加而不是 extend
                except AttributeError as e:
                    print(f"处理文档元数据失败 {file_path}: {str(e)}")
                    continue
                
        except Exception as e:
            print(f"处理文件 {file_path} 时发生错误: {str(e)}")
            continue
            
    return documents

def get_knowledge_base_files():
    """获取知识库中的所有文件"""
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        return []
    
    files = []
    for root, _, filenames in os.walk(KNOWLEDGE_BASE_PATH):
        for filename in filenames:
            if filename.startswith('.'):  # 跳过隐藏文件
                continue
            files.append(os.path.join(root, filename))
    return files

def is_valid_file(file_path):
    """
    检查文件是否有效
    
    Args:
        file_path (str): 文件路径
    
    Returns:
        bool: 文件是否有效
    """
    try:
        normalized_path = normalize_path(file_path)
        return os.path.exists(normalized_path) and os.path.isfile(normalized_path)
    except Exception:
        return False 
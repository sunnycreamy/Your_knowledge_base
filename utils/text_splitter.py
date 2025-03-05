from utils.imports import RecursiveCharacterTextSplitter
import os
import streamlit as st
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, TEXT_SEPARATORS

def get_text_splitter(file_path=None):
    """根据文件类型智能选择分块大小"""
    
    # 根据文件扩展名判断文件类型
    if file_path:
        ext = os.path.splitext(file_path)[1].lower()
        
        # 根据文件类型调整配置
        if ext == '.pdf':
            config = {
                "chunk_size": 500,  # PDF使用较小的分块
                "chunk_overlap": 50,
                "separators": ["\n\n", "\n", "。", ".", " ", ""]
            }
        elif ext in ['.txt', '.md']:
            config = {
                "chunk_size": 800,  # 文本文件使用中等分块
                "chunk_overlap": 100,
                "separators": ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
            }
        else:
            # 默认配置
            config = {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "separators": TEXT_SEPARATORS
            }
    else:
        config = {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "separators": TEXT_SEPARATORS
        }
    
    return RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=config["separators"]
    )

def _is_academic_document(file_path):
    """检查是否是学术/技术文档"""
    try:
        # 读取文件前几行
        academic_keywords = [
            "abstract", "introduction", "methodology", "conclusion",
            "references", "摘要", "引言", "方法", "结论", "参考文献"
        ]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(5000)  # 只读取前5000字符
            content = content.lower()
            
            # 检查学术关键词
            keyword_count = sum(1 for keyword in academic_keywords if keyword in content)
            return keyword_count >= 2  # 如果包含2个以上关键词，认为是学术文档
            
    except Exception:
        return False

def _is_long_document(file_path):
    """检查是否是长文本文档"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 如果文本长度超过10000字符，认为是长文本
            return len(content) > 10000
    except Exception:
        return False 
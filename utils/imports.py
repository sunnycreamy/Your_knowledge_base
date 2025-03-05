"""集中管理所有外部导入"""
try:
    # LangChain 基础组件
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # 模型和向量库
    from langchain_community.llms import Ollama
    from langchain_chroma import Chroma
    from langchain_ollama import ChatOllama
    from langchain_ollama import OllamaEmbeddings
    
    # 文档加载器
    from langchain_community.document_loaders import (
        PyPDFLoader,
        Docx2txtLoader,
        TextLoader,
        UnstructuredMarkdownLoader
    )
    
    # 智谱AI
    try:
        from zhipuai import ZhipuAI as ChatZhipuAI
    except ImportError:
        try:
            from zhipuai import ChatZhipuAI
        except ImportError:
            ChatZhipuAI = None
            
    # LangChain Chat 模型支持
    from langchain.schema import HumanMessage, AIMessage
    from langchain_community.chat_models import ChatOllama

    # API 客户端导入
    try:
        import requests
        from typing import Dict, List, Optional, Union
        import os
        from datetime import datetime, timedelta
        import json
        from dotenv import load_dotenv
        from pathlib import Path
    except ImportError as e:
        print(f"API 客户端导入错误: {e}")
        

    # PDF处理
    from PyPDF2 import PdfReader
    
    # Streamlit
    import streamlit as st
    
    # LangChain Schema
    from langchain.schema import (
        Document,
        HumanMessage,
        AIMessage
    )
    
    # SQLite 兼容性处理
    try:
        __import__('pysqlite3')
        import sys
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        pass

except ImportError as e:
    print(f"基础导入错误: {e}")
    print("请运行: pip install -r requirements.txt") 
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from zhipuai_embedding import ZhipuAIEmbeddings
# from langchain.vectorstores.chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# import io
# import fitz  # PyMuPDF
from langchain.schema import Document
from PyPDF2 import PdfReader
from zhipuai_llm import ZhipuAILLM

load_dotenv()  # 读取 .env 文件
zhipuai_api_key = os.getenv('ZHIPUAI_API_KEY')

if not zhipuai_api_key:
    st.error("未找到 ZHIPUAI_API_KEY。请确保在 .env 文件中设置了正确的 API key。")
    st.stop()

def generate_response(input_text, zhipuai_api_key):
    llm = ZhipuAILLM(
                    model= "glm-4-flash",
                     temperature=0.7, 
                     api_key=zhipuai_api_key)
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    #st.info(output)
    return output

# def get_vectordb_disk():
#     # 定义 Embeddings
#     embedding = ZhipuAIEmbeddings()
#     # 向量数据库持久化路径
#     persist_directory = 'data_base/vector_db/chroma'
#     # 加载数据库
#     vectordb = Chroma(
#         persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
#         embedding_function=embedding
#     )
        
#     return vectordb

# def get_vectordb_memory(split_docs):
#     # 定义 Embeddings
#     embedding = ZhipuAIEmbeddings()
#     # 加载数据库
#     vectordb = Chroma.from_documents(
#         documents=split_docs,
#         embedding=embedding
#     )
#     return vectordb

# # 获取vectordb
# def get_vectordb(uploaded_files):
#     embedding = ZhipuAIEmbeddings()
#     persist_directory = 'data_base/vector_db/chroma'
#     if uploaded_files:
#         documents = []
#         for uploaded_file in uploaded_files:
#             if uploaded_file is not None:
#                 # PDF保存文件到本地,并通过文件路径加载文档
#                 # path = os.path.join("data_base/knowledge_db", uploaded_file.name)
#                 # with open(path, "wb") as f:
#                 #     f.write(uploaded_file.getbuffer())
#                 # documents = load_pdf(path)


#                 # # 从缓存中读取 PDF 文件内容 方法一
#                 # pdf_bytes = uploaded_file.read()
#                 # pdf_stream = io.BytesIO(pdf_bytes)
#                 # pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")

#                     # # 提取文本内容
#                 # text = ""
#                 # for page_num in range(len(pdf_document)):
#                 #     page = pdf_document.load_page(page_num)
#                 #     text += page.get_text()

#                 # 从缓存中读取 PDF 文件内容 方法二
#                 pdf_document = PdfReader(uploaded_file)
#                     # 提取文本内容
#                 text = ""
#                 for page in pdf_document.pages:
#                     text += page.extract_text()

#                 # 创建 langchain 文档对象
#                 document = Document(page_content=text)
#                 documents.append(document)
#                 st.sidebar.success(f"{uploaded_file.name} has been successfully uploaded.")
#             else:
#                 st.sidebar.error(f"{uploaded_file.name} is not a valid file.")

#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=500, chunk_overlap=50)

#         split_docs = text_splitter.split_documents(documents)

#         vectordb = Chroma.from_documents(
#                                             documents=split_docs,
#                                             embedding=embedding)
#     else:
#         vectordb = Chroma(
#                             persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
#                             embedding_function=embedding)
#     return vectordb

def get_vectordb(uploaded_files):
    # 加载环境变量
    load_dotenv()
    
    # 获取 API key
    api_key = os.getenv('ZHIPUAI_API_KEY')
    if not api_key:
        st.error("请设置智谱 AI API key")
        return None
        
    embedding = ZhipuAIEmbeddings(api_key=api_key)
    persist_directory = 'data_base/vector_db/chroma'
    
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            try:
                # 打印文件信息
                st.write(f"正在处理文件: {uploaded_file.name}")
                
                # 直接从上传的文件对象读取，不保存到本地
                pdf_document = PdfReader(uploaded_file)
                text = ""
                for page in pdf_document.pages:
                    text += page.extract_text()
                
                # 打印提取的文本长度
                st.write(f"提取的文本长度: {len(text)}")
                
                if text.strip():
                    document = Document(
                        page_content=text,
                        metadata={"source": uploaded_file.name}
                    )
                    documents.append(document)
                    st.sidebar.success(f"{uploaded_file.name} 已成功处理。")
                else:
                    st.sidebar.warning(f"{uploaded_file.name} 是空文件。")
                
            except Exception as e:
                st.sidebar.error(f"处理 {uploaded_file.name} 时出错: {str(e)}")
                continue
        
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 增加块大小
            chunk_overlap=200,  # 增加重叠部分
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]  # 优化分割符
        )
        split_docs = text_splitter.split_documents(documents)
            
        # 创建临时向量数据库（不持久化）
        vectordb = Chroma.from_documents(
            documents=split_docs,
            embedding=embedding
        )
        return vectordb
    else:
        # 如果没有上传文件，加载已存在的向量数据库
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )

#带有历史记录的问答链
def get_chat_qa_chain(input_text, zhipuai_api_key, vectordb):
    llm = ZhipuAILLM(model= "glm-4-flash", temperature = 0,api_key=zhipuai_api_key)
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    retriever=vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    result = qa.invoke({"question": input_text})
    return result['answer']

#不带历史记录的问答链
def get_qa_chain(question: str, zhipuai_api_key: str, vectordb, query_mode="全局查询", selected_files=None):
    try:
        llm = ZhipuAILLM(model="glm-4", temperature=0.7, api_key=zhipuai_api_key)
        
        # 创建检索器并获取相关文档
        retriever = vectordb.as_retriever(
            search_kwargs={
                "k": 6,  # 增加检索文档数
                "fetch_k": 10,  # 预取更多文档
                "score_threshold": 0.5  # 设置相关性阈值
            }
        )
        docs = retriever.get_relevant_documents(question)
        
        # 打印检索到的文档内容（调试用）
        st.write(f"找到 {len(docs)} 个相关文档片段")
        for i, doc in enumerate(docs):
            st.write(f"\n文档 {i+1}:")
            st.write(f"内容: {doc.page_content[:200]}...")  # 只显示前200个字符
            st.write(f"来源: {doc.metadata.get('source', '未知')}")
        
        # 添加思考步骤
        template = """你是一个专业的文档分析助手。请按照以下步骤回答问题：

1. 仔细阅读上下文信息
2. 分析关键信息点
3. 组织逻辑框架
4. 提供详细答案

上下文信息：
{context}

问题：{question}

思考步骤：
1. 主要信息：[列出关键信息点]
2. 相关细节：[补充重要细节]
3. 逻辑关系：[分析信息之间的联系]
4. 完整回答：[基于以上分析给出全面答案]

回答："""
        
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        # 创建问答链
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        
        # 获取答案
        result = qa_chain.invoke({"query": question})
        
        # 添加来源信息
        sources = set()
        if hasattr(result, 'source_documents'):
            for doc in result['source_documents']:
                if 'source' in doc.metadata:
                    sources.add(doc.metadata['source'])
        
        answer = result["result"]
        if sources:
            answer += f"\n\n参考来源：{', '.join(sources)}"
        
        return answer
        
    except Exception as e:
        st.error(f"问答处理出错: {str(e)}")
        return "抱歉，处理您的问题时出现错误。"


# #加载PDF和Markdown文件
# def load_pdf(file_path):
#     return PyMuPDFLoader(file_path).load()


# Streamlit 应用程序界面
def main():
    st.title('ggbond_knowledge_base')
    
    # API密钥输入
    zhipuai_api_key = st.sidebar.text_input('Zhipu API Key', type='password')
    
    # 文件上传功能
    uploaded_files = st.sidebar.file_uploader("上传PDF文件", type=["pdf"], accept_multiple_files=True)
    
    # 系统维护工具放在侧边栏的折叠面板中
    with st.sidebar.expander("系统维护", expanded=False):
        if st.button("📋 检查文件状态", key="check_status"):
            st.write("### 文件状态检查")
            for category in os.listdir("data_base/knowledge_db"):
                category_path = os.path.join("data_base/knowledge_db", category)
                if os.path.isdir(category_path):
                    st.write(f"\n#### {category} 类别：")
                    for file in os.listdir(category_path):
                        if file.endswith('.pdf'):
                            file_path = os.path.join(category_path, file)
                            success, message = process_file(file_path, category)
                            if success:
                                st.success(f"✅ {file} 处理正常")
                            else:
                                st.error(f"❌ {file} 处理失败: {message}")
        
        if st.button("🔄 重建向量数据库", key="rebuild_db"):
            with st.spinner("正在重建向量数据库..."):
                if os.path.exists("data_base/vector_db/chroma"):
                    import shutil
                    shutil.rmtree("data_base/vector_db/chroma")
                vectordb = process_knowledge_base()
                st.success("向量数据库重建完成！")
    
    # 添加文件选择器
    query_mode, selected_files = file_selector()
    
    # 处理上传的文件并获取vectordb
    if uploaded_files:
        vectordb = get_vectordb(uploaded_files)
    else:
        # 尝试加载现有向量数据库
        try:
            embedding = ZhipuAIEmbeddings()
            vectordb = Chroma(
                persist_directory='data_base/vector_db/chroma',
                embedding_function=embedding
            )
        except Exception as e:
            st.error("加载向量数据库失败，请重新处理知识库。")
            vectordb = None
    
    # 选择问答方式
    selected_method = st.sidebar.selectbox(
        "选择问答方式",
        ["qa_chain", "chat_qa_chain", "chat"]
    )
    
    # 初始化聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])
    
    # 用户输入
    if prompt := st.chat_input("请输入您的问题"):
        st.session_state.messages.append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if selected_method == "qa_chain":
                answer = get_qa_chain(prompt, zhipuai_api_key, vectordb, query_mode, selected_files)
            elif selected_method == "chat_qa_chain":
                answer = get_chat_qa_chain(prompt, zhipuai_api_key, vectordb)
            else:
                answer = generate_response(prompt, zhipuai_api_key)
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "text": answer})

def file_selector():
    st.sidebar.markdown("### 选择要查询的文件")
    knowledge_dir = "data_base/knowledge_db"
    
    # 查询模式选择
    query_mode = st.sidebar.radio(
        "选择查询模式",
        ["全局查询", "分类查询"],
        help="全局查询将搜索所有文件，分类查询可以选择特定类别"
    )
    
    selected_files = []
    if query_mode == "分类查询":
        # 获取所有分类目录
        categories = [d for d in os.listdir(knowledge_dir) 
                     if os.path.isdir(os.path.join(knowledge_dir, d))]
        
        # 选择分类
        selected_categories = st.sidebar.multiselect(
            "选择要查询的分类",
            categories,
            help="选择一个或多个分类进行查询"
        )
        
        # 显示所选分类下的文件
        if selected_categories:
            for category in selected_categories:
                category_path = os.path.join(knowledge_dir, category)
                if os.path.exists(category_path):
                    files = [f for f in os.listdir(category_path) if f.endswith('.pdf')]
                    if files:
                        st.sidebar.markdown(f"#### {category}类文件：")
                        category_files = st.sidebar.multiselect(
                            f"选择{category}类文件",
                            files,
                            key=f"files_{category}"
                        )
                        selected_files.extend([os.path.join(category, f) for f in category_files])
    
    return query_mode, selected_files

def file_manager():
    st.sidebar.markdown("### 文件管理")
    
    # 显示已存储的文件
    knowledge_dir = "data_base/knowledge_db"
    if os.path.exists(knowledge_dir):
        files = os.listdir(knowledge_dir)
        if files:
            st.sidebar.markdown("#### 已存储的文件：")
            for file in files:
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    st.write(file)
                with col2:
                    if st.button("删除", key=f"delete_{file}"):
                        file_path = os.path.join(knowledge_dir, file)
                        try:
                            os.remove(file_path)
                            st.success(f"已删除 {file}")
                            # 删除后需要重建向量数据库
                            if os.path.exists("data_base/vector_db/chroma"):
                                import shutil
                                shutil.rmtree("data_base/vector_db/chroma")
                            st.rerun()
                        except Exception as e:
                            st.error(f"删除失败：{str(e)}")

def check_password():
    """返回`True` 如果用户输入正确的密码"""
    def password_entered():
        """检查是否输入了正确的密码"""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # 不要在session state中保存密码
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # 显示输入框
        st.text_input(
            "请输入密码", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # 密码错误
        st.text_input(
            "请输入密码", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 密码错误")
        return False
    else:
        # 密码正确
        return True

def create_category_structure():
    """创建分类目录结构"""
    base_dir = "data_base/knowledge_db"
    categories = {
        "历史": "history",
        "社会": "society",
        "文学": "literature",
        "科技": "technology",
        "人生周报":"life_weekly",
        # 可以添加更多分类
    }
    
    # 创建分类目录
    for category in categories.values():
        os.makedirs(os.path.join(base_dir, category), exist_ok=True)
    
    return categories

def enhanced_file_selector():
    """增强版文件选择器"""
    st.sidebar.markdown("### 知识库查询设置")
    
    # 获取所有分类
    categories = create_category_structure()
    base_dir = "data_base/knowledge_db"
    
    # 查询模式选择
    query_mode = st.sidebar.radio(
        "选择查询模式",
        ["全局查询", "分类查询"],
        help="全局查询将搜索所有文件，分类查询可以选择特定类别"
    )
    
    selected_files = []
    if query_mode == "分类查询":
        # 选择分类
        selected_categories = st.sidebar.multiselect(
            "选择要查询的分类",
            list(categories.keys()),
            help="选择一个或多个分类进行查询"
        )
        
        # 显示所选分类下的文件
        if selected_categories:
            for category in selected_categories:
                category_path = os.path.join(base_dir, categories[category])
                if os.path.exists(category_path):
                    files = os.listdir(category_path)
                    if files:
                        st.sidebar.markdown(f"#### {category}类文件：")
                        category_files = st.sidebar.multiselect(
                            f"选择{category}类文件",
                            files,
                            key=f"files_{category}"
                        )
                        selected_files.extend([os.path.join(categories[category], f) for f in category_files])
    
    return query_mode, selected_files

def process_file(file_path, category):
    """处理单个文件并返回处理状态"""
    try:
        # 读取PDF
        reader = PdfReader(file_path)
        text = ""
        total_pages = len(reader.pages)
        
        if total_pages == 0:
            return False, "PDF文件为空"
            
        # 读取所有页面
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                
        # 检查提取的文本
        if not text.strip():
            return False, "无法提取文本内容"
            
        # 返回成功状态和文本长度信息
        return True, f"成功提取 {len(text)} 字符的文本"
        
    except Exception as e:
        return False, f"处理出错: {str(e)}"

def process_knowledge_base():
    """处理知识库目录中的所有PDF文件"""
    knowledge_dir = "data_base/knowledge_db"
    documents = []
    
    # 确保目录存在
    os.makedirs(knowledge_dir, exist_ok=True)
    
    # 获取所有子目录（分类目录）
    categories = [d for d in os.listdir(knowledge_dir) 
                 if os.path.isdir(os.path.join(knowledge_dir, d))]
    
    if not categories:
        st.warning("知识库目录中没有分类目录。")
        return None
    
    # 显示处理进度
    progress_text = st.empty()
    progress_bar = st.progress(0)
    total_files = 0
    processed_files = 0
    
    # 计算总文件数
    for category in categories:
        category_path = os.path.join(knowledge_dir, category)
        pdf_files = [f for f in os.listdir(category_path) if f.endswith('.pdf')]
        total_files += len(pdf_files)
    
    # 处理每个分类下的文件
    for category in categories:
        category_path = os.path.join(knowledge_dir, category)
        pdf_files = [f for f in os.listdir(category_path) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            try:
                file_path = os.path.join(category_path, pdf_file)
                progress_text.text(f"正在处理: {category}/{pdf_file}")
                
                # 使用 PyMuPDF 分批读取
                reader = PdfReader(file_path)
                total_pages = len(reader.pages)
                
                # 分批处理页面
                batch_size = 10  # 每批处理10页
                for page_start in range(0, total_pages, batch_size):
                    batch_text = ""
                    page_end = min(page_start + batch_size, total_pages)
                    
                    for page_num in range(page_start, page_end):
                        try:
                            page = reader.pages[page_num]
                            batch_text += page.extract_text() + "\n"
                        except Exception as e:
                            st.warning(f"{pdf_file} 第 {page_num + 1} 页处理出错: {str(e)}")
                            continue
                    
                    if batch_text.strip():
                        documents.append(Document(
                            page_content=batch_text,
                            metadata={
                                "source": f"{category}/{pdf_file}",
                                "category": category,
                                "page_range": f"{page_start+1}-{page_end}"
                            }
                        ))
                
                processed_files += 1
                progress = processed_files / total_files
                progress_bar.progress(progress)
                
            except Exception as e:
                st.error(f"处理 {pdf_file} 时出错: {str(e)}")
                continue
    
    if not documents:
        st.error("没有成功处理任何文档。")
        return None
    
    # 创建向量数据库
    try:
        # 确保 API key 存在
        api_key = os.getenv('ZHIPUAI_API_KEY')
        if not api_key:
            st.error("未找到 ZHIPUAI_API_KEY，请检查 .env 文件")
            return None
            
        # 创建 embedding 实例
        embedding = ZhipuAIEmbeddings(api_key=api_key)
        
        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        split_docs = text_splitter.split_documents(documents)
        
        if split_docs:
            persist_directory = 'data_base/vector_db/chroma'
            
            # 确保目录存在
            os.makedirs(persist_directory, exist_ok=True)
            
            # 创建向量数据库
            vectordb = Chroma.from_documents(
                documents=split_docs,
                embedding=embedding,
                persist_directory=persist_directory
            )
            
            # 持久化保存
            vectordb.persist()
            
            # 显示成功信息
            st.success(f"成功处理 {len(split_docs)} 个文档片段！")
            return vectordb
        else:
            st.error("文档分割后为空。")
            return None
            
    except Exception as e:
        st.error(f"创建向量数据库时出错: {str(e)}")
        return None

if __name__ == "__main__":
    main()

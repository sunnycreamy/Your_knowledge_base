from utils.imports import (
    st,
    PdfReader,
    Document
)
from utils.file_utils import get_knowledge_base_files, load_documents
from utils.model_utils import get_embedding_model
from utils.text_splitter import get_text_splitter
from utils.vectordb_utils import rebuild_entire_vectordb, add_documents_to_vectordb
import os
import logging
from pathlib import Path
import requests
from config.config import OLLAMA_URL, APP_TITLE, APP_ICON, VECTOR_DB_PATH,AVAILABLE_MODELS
# from langchain.schema import Document
from langchain_chroma import Chroma as ChromaDB
import chromadb
from services.google_drive_service import GoogleDriveService


# 设置日志记录器
logger = logging.getLogger(__name__)

# 初始化 Google Drive 服务
drive_service = GoogleDriveService()

# def get_ollama_models():
#     """获取Ollama已安装的模型列表"""
#     try:
#         response = requests.get(f"{OLLAMA_URL}/api/tags")
#         if response.status_code == 200:
#             models = response.json().get('models', [])
#             return [model['name'] for model in models]
#         return []
#     except Exception as e:
#         st.error(f"获取Ollama模型列表失败: {str(e)}")
#         return []

def check_cloud_sync_status():
    """检查云同步状态"""
    try:
        # 尝试验证 Google Drive 访问权限
        drive_service.authenticate()
        access_ok, folder_name = drive_service.verify_folder_access()
        return access_ok
    except Exception as e:
        st.error(f"检查云同步状态失败: {str(e)}")
        return False

def download_from_cloud():
    """从云端下载文件"""
    try:
        # 认证 Google Drive
        drive_service.authenticate()
        
        # 同步文件
        return drive_service.sync_drive_files()
    except Exception as e:
        st.error(f"从云端下载失败: {str(e)}")
        return False

def setup_sidebar():
    """设置侧边栏"""
    with st.sidebar:
        st.title("设置")
        
        # 模型选择区域
        st.markdown("### 模型设置")
        selected_model = st.selectbox(
            "选择模型",
            options=AVAILABLE_MODELS,
            key="selected_model"
        )
        
        # 初始化 api_key 变量
        api_key = None
        
        # API密钥输入
        if selected_model == "智谱GLM4":
            api_key = st.text_input("API Key", type="password", key="api_key")
        
        st.markdown("---")

         # 直接在这里添加Google Drive同步功能
        st.markdown("### 功能选择")
        drive_sync = st.checkbox("Google Drive 同步", key="drive_sync_checkbox")
        
        # 创建一个占位符用于显示Google Drive同步内容
        drive_sync_content = st.empty()
        
        # 将占位符存储在会话状态中，以便main.py可以访问
        st.session_state.drive_sync_content = drive_sync_content
        st.session_state.drive_sync_checked = drive_sync
        
        st.markdown("---")
        
        
        # 文件管理区域
        st.markdown("### 文件管理")
        
        # 新建分类
        new_category = st.text_input("新建分类", placeholder="输入分类名称")
        if st.button("创建分类", type="primary", key="create_category"):
            if new_category:
                category_path = Path("data_base/knowledge_db") / new_category
                if not category_path.exists():
                    category_path.mkdir(parents=True)
                    st.success(f"已创建分类: {new_category}")
                    st.rerun()
                else:
                    st.warning("该分类已存在")
        
        # 选择分类
        base_path = Path("data_base/knowledge_db")
        categories = [d.name for d in base_path.iterdir() 
                     if d.is_dir() and not d.name.startswith('.')]
        
        if categories:
            selected_category = st.selectbox(
                "选择分类",
                options=sorted(categories),
                key="category_select"
            )
            
            if selected_category:
                # 文件上传
                uploaded_files = st.file_uploader(
                    "上传文件",
                    accept_multiple_files=True,
                    type=['pdf', 'docx', 'txt', 'md'],
                    key=f"uploader_{selected_category}"
                )
                
                if uploaded_files:
                    handle_file_upload(uploaded_files, selected_category)
                
                # 显示当前文件
                show_category_files(selected_category)
        
        st.markdown("---")

        st.markdown("### 向量库管理")
        
        if st.button("增量添加文档到向量库"):
            with st.spinner("正在添加文档到向量库..."):
                try:
                    # 获取所有文档
                    files = get_knowledge_base_files()
                    if not files:
                        st.warning("没有找到任何文档文件")
                    else:
                        # 增量添加到向量库
                        vectordb = add_documents_to_vectordb(files)
                        if vectordb:
                            st.success(f"成功添加 {len(files)} 个文档到向量库！")
                        else:
                            st.error("添加文档失败，请检查日志或联系管理员")
                except Exception as e:
                    st.error(f"""
                    添加文档时出错: {str(e)}
                    如果问题持续存在，请联系管理员进行向量库维护
                    """)
        
        # 添加帮助信息
        st.markdown("""
            **向量库使用说明：**
            1. 系统会自动处理文档并更新向量库
            2. 如更新出现问题，点击"增量添加文档"按钮可将新文档的向量添加到知识库
            
            **遇到问题？**
            - 如果遇到向量库异常或需要重建向量库
            - 请联系系统管理员进行处理
            - 管理员邮箱：xxx@xxx.com
            """)
        
        # # Google Drive同步选项
        # st.title("功能选择")
        # if st.checkbox("Google Drive 同步"):
        #     show_drive_sync()

        # 添加关于信息
        st.markdown("### 关于")
        st.markdown("本应用基于 LangChain 和 Streamlit 构建")
        st.markdown("版本: 1.0.0")
        
        return selected_model, api_key

def handle_file_upload(uploaded_files, category):
    """处理文件上传"""
    category_path = Path("data_base/knowledge_db") / category
    uploaded_file_paths = []
    
    # 使用单个状态组件
    status_container = st.status("处理上传文件...")
    
    try:
        # 第一阶段：保存文件
        total_files = len(uploaded_files)
        for idx, file in enumerate(uploaded_files, 1):
            try:
                status_container.write(f"上传文件 ({idx}/{total_files}): {file.name}")
                save_path = category_path / file.name
                
                if save_path.exists():
                    status_container.write(f"⚠️ 文件已存在: {file.name}")
                    continue
                    
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())
                uploaded_file_paths.append(str(save_path))
                status_container.write(f"✅ 已上传: {file.name}")
            except Exception as e:
                status_container.write(f"❌ 上传失败 {file.name}: {str(e)}")
    
        # 第二阶段：更新向量库（如果有上传的文件）
        if uploaded_file_paths:
            status_container.update(label="正在更新向量库...")
            
            try:
                # 获取嵌入模型
                embedding_model = get_embedding_model()
                if not embedding_model:
                    status_container.write("❌ 初始化嵌入模型失败")
                    return
                
                # 加载向量库 - 使用新版本的 ChromaDB API
                vector_db_path = Path(VECTOR_DB_PATH)
                vector_db_path.mkdir(parents=True, exist_ok=True)
                
                # 使用 PersistentClient 而不是旧的接口
                client = chromadb.PersistentClient(path=str(vector_db_path))
                collection = client.get_or_create_collection("knowledge_base")
                
                # 使用 LangChain 的 ChromaDB 包装器
                vectordb = ChromaDB(
                    client=client,
                    collection_name="knowledge_base",
                    embedding_function=embedding_model
                )
                
                # 处理每个文件
                for file_path in uploaded_file_paths:
                    file_path = Path(file_path)
                    status_container.write(f"处理文件: {file_path.name}")
                    
                    try:
                        # 获取文本分割器
                        text_splitter = get_text_splitter(str(file_path))
                        
                        # 加载和分割文档
                        doc_texts = text_splitter.split_documents(load_documents([str(file_path)]))
                        
                        if doc_texts:
                            # 添加到向量库
                            vectordb.add_documents(doc_texts)
                            status_container.write(f"✅ 已添加 {len(doc_texts)} 个片段")
                    except Exception as e:
                        status_container.write(f"❌ 处理文件 {file_path.name} 失败: {str(e)}")
                
                # 新版本 ChromaDB 不需要显式调用 persist
                # 数据会自动保存
                status_container.write("✅ 向量库更新完成")
                st.session_state.files_updated = True
                
            except Exception as e:
                status_container.write(f"❌ 更新向量库失败: {str(e)}")
                import traceback
                status_container.write(traceback.format_exc())
    
    finally:
        # 更新状态
        if st.session_state.get('files_updated', False):
            status_container.update(label="文件处理完成", state="complete")
        else:
            status_container.update(label="文件处理完成，但有错误发生", state="error")
    
    # 如果成功更新了文件，刷新页面
    if st.session_state.get('files_updated', False):
        st.success("文件处理完成，正在刷新页面...")
        st.session_state.files_updated = False
        st.rerun()

def show_category_files(category):
    """显示分类下的文件"""
    category_path = Path("data_base/knowledge_db") / category
    
    # 获取文件列表并排除隐藏文件
    try:
        files = [f.name for f in category_path.iterdir() 
                if f.is_file() and not f.name.startswith('.')]
    except Exception as e:
        st.error(f"读取文件列表失败: {str(e)}")
        return
    
    if files:
        with st.expander("查看当前文件", expanded=False):
            for file in sorted(files):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"📄 {file}")
                with col2:
                    delete_key = f"delete_{category}_{file}"
                    if st.button("删除", key=delete_key):
                        try:
                            file_path = category_path / file
                            # 检查文件是否存在
                            if not file_path.exists():
                                st.error(f"文件不存在: {file}")
                                return
                                
                            # 删除文件
                            file_path.unlink()
                            
                            # ✅ 先显示成功消息
                            st.success(f"已删除: {file}")
                            
                            # ✅ 成功后再标记状态
                            st.session_state.file_deleted = True
                            st.session_state.deleted_file = file  # 可选：记录被删除的文件名
                            
                        except Exception as e:
                            logger.error(f"删除文件失败: {str(e)}")
                            st.error(f"删除失败: {str(e)}")
    else:
        st.info(f"'{category}' 分类中暂无文件")

def get_available_categories():
    """获取可用的文件分类"""
    base_path = Path("data_base/knowledge_db")
    if not base_path.exists():
        return []
    
    categories = []
    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            categories.append(item.name)
    return sorted(categories)

def get_category_files(category):
    """获取指定分类下的PDF文件"""
    category_path = Path("data_base/knowledge_db") / category
    if not category_path.exists():
        return []
    
    files = []
    for item in category_path.iterdir():
        if item.is_file() and item.suffix.lower() == '.pdf':
            files.append(item)
    return sorted(files, key=lambda x: x.name.lower())

def file_selector():
    """
    文件选择器组件
    
    Returns:
        list: 选中的文件路径列表
    """
    # 获取可用分类
    categories = get_available_categories()
    if not categories:
        st.warning("知识库中没有找到任何分类文件夹")
        return []
    
    # 分类选择
    selected_categories = st.multiselect(
        "选择文件分类",
        options=categories,
        default=categories,
        help="选择要查询的知识分类"
    )
    
    selected_files = []
    
    # 为每个选中的分类创建一个子expander
    for category in selected_categories:
        with st.expander(f"📁 {category}", expanded=True):
            category_files = get_category_files(category)
            if not category_files:
                st.info(f"{category} 分类下没有PDF文件")
                continue
            
            # 显示文件选择器
            file_names = [f.name for f in category_files]
            selected = st.multiselect(
                "选择文件",
                options=file_names,
                default=file_names,
                key=f"files_{category}"  # 为每个分类使用唯一的key
            )
            
            # 将选中的文件添加到列表中
            for file_name in selected:
                file_path = str(Path("data_base/knowledge_db") / category / file_name)
                selected_files.append(file_path)
    
    if selected_files:
        st.success(f"已选择 {len(selected_files)} 个文件")
    
    return selected_files

def show_file_upload():
    """
    文件上传组件
    
    Returns:
        list: 上传的文件列表
    """
    return st.file_uploader(
        "上传文件到知识库",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt', 'md']
    )

def show_system_status(vectordb_status=False, drive_sync_status=False):
    """
    显示系统状态
    
    Args:
        vectordb_status (bool): 向量数据库状态
        drive_sync_status (bool): 云同步状态
    """
    col1, col2 = st.columns(2)
    with col1:
        status_color = "🟢" if vectordb_status else "🔴"
        st.markdown(f"{status_color} 向量数据库状态")
    
    with col2:
        status_color = "🟢" if drive_sync_status else "🔴"
        st.markdown(f"{status_color} 云同步状态")

def display_chat_message(role: str, content: str):
    """显示聊天消息"""
    if role == "user":
        st.markdown(f"🧑 **问题：** {content}")
    else:
        st.markdown(f"🤖 **回答：** {content}")
       


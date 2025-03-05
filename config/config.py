import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Google Drive配置
CREDENTIALS_PATH = os.path.join(BASE_DIR, 'client_secret.json')
TOKEN_PATH = os.path.join(BASE_DIR, 'token.pickle')

# Google Drive文件夹ID 
DRIVE_FOLDER_ID = os.getenv('DRIVE_FOLDER_ID')
if not DRIVE_FOLDER_ID:
    raise ValueError("请在环境变量中设置 DRIVE_FOLDER_ID")

# 知识库路径
KNOWLEDGE_BASE_PATH = Path("data_base/knowledge_db")

# API配置
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")  
ZHIPUAI_API_KEY = os.getenv('ZHIPUAI_API_KEY')


# 文档处理配置
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TEXT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

# 应用配置
APP_TITLE = "📚Your Personal Knowledge Base"
APP_ICON = "📚"
LAYOUT = "wide"  
INITIAL_SIDEBAR_STATE = "expanded"

# 模型配置
AVAILABLE_MODELS = [
    "智谱GLM4",
    "deepseek-r1:latest",
    "llama3.2:latest"
] 
DEFAULT_MODEL = "智谱GLM4"
TEMPERATURE = 0.7

# Embedding模型配置
EMBEDDING_MODEL = "bge-m3"  # 固定使用 bge-m3

# 确保必要的目录存在
KNOWLEDGE_BASE_PATH.mkdir(parents=True, exist_ok=True)
VECTOR_DB_PATH = Path(BASE_DIR) /"data_base" / "vector_db" / "chroma_db"
VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

# 数据库配置
DATABASE_URL = "sqlite:///./google_drive_sync.db"

# Google Drive文件夹ID - 只从环境变量获取
DRIVE_FOLDER_ID = os.getenv('DRIVE_FOLDER_ID')
if not DRIVE_FOLDER_ID:
    raise ValueError("请在环境变量中设置 DRIVE_FOLDER_ID") 
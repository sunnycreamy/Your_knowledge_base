import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import shutil
from utils.vectordb_utils import rebuild_vectordb_for_files
from utils.file_utils import get_knowledge_base_files
from config.config import VECTOR_DB_PATH

def rebuild_vectorstore():
    """重建向量库"""
    try:
        # 1. 获取所有知识库文件
        files = get_knowledge_base_files()
        if not files:
            print("没有找到任何文档文件")
            return
        
        # 2. 删除现有的向量库
        vector_db_path = Path(VECTOR_DB_PATH)
        if vector_db_path.exists():
            print(f"删除现有向量库: {vector_db_path}")
            shutil.rmtree(vector_db_path)
        
        # 3. 重建向量库
        print(f"开始重建向量库，处理 {len(files)} 个文件...")
        vectordb = rebuild_vectordb_for_files(files)
        
        if vectordb:
            print("向量库重建成功！")
        else:
            print("向量库重建失败！")
            
    except Exception as e:
        print(f"重建向量库时出错: {str(e)}")

if __name__ == "__main__":
    print("开始重建向量库...")
    rebuild_vectorstore()



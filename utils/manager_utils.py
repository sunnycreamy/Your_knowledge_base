import streamlit as st
from pathlib import Path
import os
import shutil

def ensure_knowledge_base_structure():
    """确保知识库基础结构存在"""
    base_path = Path("data_base/knowledge_db")
    base_path.mkdir(parents=True, exist_ok=True)
    
    # 确保基础分类存在
    default_categories = [
        'Autobiography',
        'history',
        'life_weekly',
        'literature',
        'society',
        'technology'
    ]
    
    # 创建默认分类
    for category in default_categories:
        category_path = base_path / category
        category_path.mkdir(exist_ok=True)
        
    return True

def get_file_count(category):
    """获取分类下的文件数量"""
    category_path = Path("data_base/knowledge_db") / category
    if not category_path.exists():
        return 0
    return len([f for f in category_path.iterdir() if f.is_file() and not f.name.startswith('.')])

def delete_category(category):
    """删除分类及其内容"""
    try:
        category_path = Path("data_base/knowledge_db") / category
        if category_path.exists():
            shutil.rmtree(category_path)
            return True
    except Exception as e:
        st.error(f"删除分类失败: {str(e)}")
        return False

def delete_file(category, filename):
    """删除指定文件"""
    try:
        file_path = Path("data_base/knowledge_db") / category / filename
        if file_path.exists():
            os.remove(file_path)
            return True
    except Exception as e:
        st.error(f"删除文件失败: {str(e)}")
        return False

def rename_category(old_name, new_name):
    """重命名分类"""
    try:
        old_path = Path("data_base/knowledge_db") / old_name
        new_path = Path("data_base/knowledge_db") / new_name
        if old_path.exists() and not new_path.exists():
            old_path.rename(new_path)
            return True
    except Exception as e:
        st.error(f"重命名分类失败: {str(e)}")
        return False

def get_file_info(category, filename):
    """获取文件信息"""
    file_path = Path("data_base/knowledge_db") / category / filename
    if file_path.exists():
        stats = file_path.stat()
        return {
            'size': stats.st_size,
            'modified': stats.st_mtime,
            'created': stats.st_ctime
        }
    return None 
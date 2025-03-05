import streamlit as st
from pathlib import Path
import os
from config.config import KNOWLEDGE_BASE_PATH
from utils.vectordb_utils import rebuild_vectordb_for_files


def show_file_manager_dialog():
    """显示文件管理器对话框"""
    # 确保状态变量已初始化
    initialize_file_manager()
    
    try:
        base_path = Path("data_base/knowledge_db")
        if not base_path.exists():
            base_path.mkdir(parents=True)
        
        # 显示现有分类
        st.subheader("知识库分类")
        categories = [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        for category in sorted(categories, key=lambda x: x.name.lower()):
            with st.expander(f"📁 {category.name}", expanded=True):
                # 显示分类中的文件
                files = [f for f in category.iterdir() if f.is_file() and not f.name.startswith('.')]
                
                if not files:
                    st.info("该分类下暂无文件")
                else:
                    for file in sorted(files, key=lambda x: x.name.lower()):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.text(f"📄 {file.name}")
                        with col2:
                            if st.button("删除", key=f"del_{file.name}"):
                                try:
                                    file.unlink()
                                    st.success(f"已删除: {file.name}")
                                    st.session_state.need_rerun = True  # 设置重新运行标记
                                except Exception as e:
                                    st.error(f"删除失败: {str(e)}")
                
                # 分类操作按钮
                if st.button("删除分类", key=f"del_category_{category.name}"):
                    try:
                        if any(category.iterdir()):
                            st.error("分类不为空，无法删除")
                        else:
                            category.rmdir()
                            st.success(f"已删除分类: {category.name}")
                            st.session_state.need_rerun = True  # 设置重新运行标记
                    except Exception as e:
                        st.error(f"删除分类失败: {str(e)}")
        
        # 新建分类
        st.subheader("新建分类")
        new_category = st.text_input("输入分类名称")
        if st.button("创建", type="primary"):
            if new_category:
                try:
                    new_path = base_path / new_category
                    if new_path.exists():
                        st.error("该分类已存在")
                    else:
                        new_path.mkdir(parents=True)
                        st.success(f"已创建分类: {new_category}")
                        st.session_state.need_rerun = True  # 设置重新运行标记
                except Exception as e:
                    st.error(f"创建分类失败: {str(e)}")
            else:
                st.warning("请输入分类名称")
        
        # 在所有操作完成后，检查是否需要重新运行
        if st.session_state.need_rerun:
            st.session_state.need_rerun = False  # 重置标记
            st.rerun()
            
    except Exception as e:
        st.error(f"文件管理器出错: {str(e)}")

def create_file_manager():
    """创建文件管理器"""
    if st.button("打开文件管理器", type="primary"):
        st.session_state.show_file_manager = True
    
    if st.session_state.show_file_manager:
        show_file_manager_dialog()

def initialize_file_manager():
    """初始化文件管理器的状态变量"""
    if "need_rerun" not in st.session_state:
        st.session_state.need_rerun = False

    # 显示文件管理器
    if st.session_state.get('show_file_manager', False):
        show_file_manager_dialog() 
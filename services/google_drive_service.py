import warnings
import logging
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import os.path
import pickle
from config.config import CREDENTIALS_PATH, TOKEN_PATH, DRIVE_FOLDER_ID
import streamlit as st
from datetime import datetime
from models.file_model import DriveFile
from services.database import get_db
from pathlib import Path

logger = logging.getLogger("knowledge_base")
# 抑制警告信息
warnings.filterwarnings('ignore', message='file_cache is only supported with oauth2client<4.0.0')
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)


class GoogleDriveService:
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    
    def __init__(self):
        self.token_path = TOKEN_PATH
        self.credentials_path = CREDENTIALS_PATH
        self.creds = None
        self.local_base_path = Path("data_base/knowledge_db")
        
        # 确保基础目录存在
        self.local_base_path.mkdir(parents=True, exist_ok=True)
        # logger.info(f"确保本地目录存在: {self.local_base_path}")

    def authenticate(self):
        # 尝试从token.pickle加载凭据
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                self.creds = pickle.load(token)
        
        # 如果没有有效凭据，进行认证流程
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.SCOPES)
                self.creds = flow.run_local_server(port=0)
            
            # 保存凭据
            with open(self.token_path, 'wb') as token:
                pickle.dump(self.creds, token)
        
        return self.creds
    
    def list_files(self, mime_types=None):
        """列出指定Google Drive文件夹中的文件"""
        service = build('drive', 'v3', credentials=self.creds)
        
        # 构建查询条件
        query = f"'{DRIVE_FOLDER_ID}' in parents and trashed = false"
        if mime_types:
            mime_types_query = [
                "application/pdf",
                "text/plain",
                "application/vnd.google-apps.document"  # Google Docs
            ]
            query += " and (" + " or ".join([f"mimeType='{mime_type}'" for mime_type in mime_types_query]) + ")"
        
        try:
            results = service.files().list(
                q=query,
                pageSize=100,
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                supportsAllDrives=True,  # 添加支持共享驱动器
                includeItemsFromAllDrives=True  # 包含所有可访问的驱动器
            ).execute()
            
            files = results.get('files', [])
            
            # 添加调试信息
            logger.info("调试信息：")
            logger.info(f"查询条件: {query}")
            logger.info(f"找到文件数量: {len(files)}")
            for file in files:
                logger.write(f"文件名: {file['name']}, MIME类型: {file['mimeType']}")
            
            return files
        except Exception as e:
            st.error(f"获取文件列表失败: {str(e)}")
            return []

    def verify_folder_access(self):
        """验证是否可以访问指定的文件夹"""
        service = build('drive', 'v3', credentials=self.creds)
        try:
            folder = service.files().get(fileId=DRIVE_FOLDER_ID).execute()
            return True, folder['name']
        except Exception as e:
            return False, str(e)

    def sync_files(self, service, folder_id, current_path):
        """同步文件并返回新文件列表"""
        query = f"'{folder_id}' in parents and trashed = false"
        results = service.files().list(
            q=query,
            fields="files(id, name, mimeType, modifiedTime)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        
        files = results.get('files', [])
        new_files = []
        
        with get_db() as db:
            for file in files:
                # 检查是否是文件夹
                if file['mimeType'] == 'application/vnd.google-apps.folder':
                    # 递归处理子文件夹
                    new_path = current_path + '/' + file['name']
                    new_files.extend(self.sync_files(service, file['id'], new_path))
                    continue
                
                # 检查是否是支持的文件类型
                if file['mimeType'] not in ['application/pdf', 'text/plain']:
                    continue
                
                # 检查文件是否已存在
                existing_file = db.query(DriveFile).filter_by(
                    drive_file_id=file['id']
                ).first()
                
                if not existing_file:
                    # 新文件，添加到数据库
                    file_path = current_path + '/' + file['name']
                    new_file = DriveFile(
                        drive_file_id=file['id'],
                        file_name=file_path,
                        source='google_drive',
                        last_sync_time=datetime.utcnow()
                    )
                    db.add(new_file)
                    new_files.append(file)
                    
        return new_files

    def get_folder_id(self, service, parent_id, folder_name):
        """获取指定文件夹的ID"""
        query = f"'{parent_id}' in parents and name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)',
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        files = results.get('files', [])
        return files[0]['id'] if files else None

    def download_file(self, service, file_id, local_path):
        """下载文件到指定路径"""
        try:
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            fh.seek(0)
            with open(local_path, 'wb') as f:
                f.write(fh.read())
            return True
        except Exception as e:
            st.error(f"下载文件失败: {str(e)}")
            return False

    def sync_folder_content(self, service, folder_id, local_folder_path):
        """同步文件夹内容，避免重复同步"""
        try:
            local_folder_path.mkdir(parents=True, exist_ok=True)
            
            # 获取文件夹中的所有内容
            query = f"'{folder_id}' in parents and trashed=false"
            results = service.files().list(
                q=query,
                fields="files(id, name, mimeType)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            
            # 获取本地已有的文件列表
            existing_files = set(f.name for f in local_folder_path.glob('*'))
            
            for item in results.get('files', []):
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    new_local_path = local_folder_path / item['name']
                    self.sync_folder_content(service, item['id'], new_local_path)
                elif item['mimeType'] == 'application/pdf':
                    local_file_path = local_folder_path / item['name']
                    if item['name'] not in existing_files:
                        # 只在下载新文件时显示信息
                        logger.info(f"新增文件: {item['name']}")
                        self.download_file(service, item['id'], str(local_file_path))
            
        except Exception as e:
            st.error(f"同步失败: {str(e)}")

    def sync_drive_files(self):
        """同步Drive文件到本地"""
        service = build('drive', 'v3', credentials=self.creds)
        try:
            # 确保本地基础路径存在
            self.local_base_path.mkdir(parents=True, exist_ok=True)
            
            with st.spinner("正在同步文件..."):
                # 获取Drive中的所有文件夹
                results = service.files().list(
                    q=f"'{DRIVE_FOLDER_ID}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
                    fields="files(id, name)",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True
                ).execute()
                
                # 将Drive文件夹信息存储在字典中
                drive_folders = {
                    folder['name']: folder['id'] 
                    for folder in results.get('files', [])
                }
                
                if not drive_folders:
                    st.warning("在Google Drive中没有找到任何文件夹")
                    return True
                    
                logger.info(f"在Drive中找到以下文件夹: {', '.join(drive_folders.keys())}")
                
                # 同步所有文件夹
                for folder_name, folder_id in drive_folders.items():
                    local_folder_path = self.local_base_path / folder_name
                    self.sync_folder_content(service, folder_id, local_folder_path)
                
                st.success("同步完成！")
                return True
                
        except Exception as e:
            st.error(f"同步失败: {str(e)}")
            return False 
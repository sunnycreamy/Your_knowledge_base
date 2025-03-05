from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text
from services.database import Base

class DriveFile(Base):
    __tablename__ = 'drive_files'
    
    file_id = Column(String, primary_key=True)
    drive_file_id = Column(String, unique=True)
    file_name = Column(String)
    content = Column(Text)
    embeddings = Column(Text)  # 存储为JSON字符串
    source = Column(String, default='google_drive')
    last_sync_time = Column(DateTime, default=datetime.utcnow) 
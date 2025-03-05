import logging
import streamlit as st
from datetime import datetime
from pathlib import Path

# 创建日志目录
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# 配置日志格式
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"

# 创建日志文件名（使用当前日期）
log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"

# 配置日志处理器
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    datefmt=date_format,
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# 创建logger实例
logger = logging.getLogger("knowledge_base")

# 添加Streamlit输出处理
class StreamlitHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            if record.levelno >= logging.ERROR:
                st.error(msg)
            elif record.levelno >= logging.WARNING:
                st.warning(msg)
            elif record.levelno >= logging.INFO:
                logger.info(msg)
            else:
                st.debug(msg)
        except Exception:
            self.handleError(record)

# 添加Streamlit处理器
streamlit_handler = StreamlitHandler()
streamlit_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(streamlit_handler)

# 导出logger实例
__all__ = ['logger'] 
import os
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv
import requests
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging

# 加载环境变量
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

logger = logging.getLogger(__name__)

class APIHandler:
    def __init__(self):
        """初始化API处理器"""
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")
        
        if self.google_api_key:
            logger.info(f"Loaded Google API Key: {self.google_api_key[:5]}...")
        if self.google_cse_id:
            logger.info(f"Loaded Google CSE ID: {self.google_cse_id[:5]}...")

    def search_web(self, query: str) -> Dict:
        """同步搜索网页数据"""
        if not self.google_api_key or not self.google_cse_id:
            return {'google': {'error': 'API密钥未配置', 'source': 'Google Custom Search'}}

        try:
            # 优化搜索查询
            enhanced_query = f"{query} latest news site:(.edu OR .gov OR .org) -site:youtube.com -site:gov.cn -site:edu.cn -site:org.cn"
            
            # 构建请求参数
            params = {
                'q': enhanced_query,
                'key': self.google_api_key,
                'cx': self.google_cse_id,
                'num': 5,
                'sort': 'date', 
                'safe': 'active'
            }
            
            # 使用 requests 发送同步请求
            response = requests.get('https://www.googleapis.com/customsearch/v1', params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if "items" in data:
                    formatted_results = []
                    for item in data["items"]:
                        result = {
                            'title': item.get("title", "无标题"),
                            'link': item.get("link", ""),
                            'snippet': item.get("snippet", "无描述"),
                        }
                        result = {k: v for k, v in result.items() if v}
                        formatted_results.append(result)
                    
                    return {
                        'google': {
                            'data': formatted_results,
                            'source': 'Google Custom Search',
                            'total_results': data.get("searchInformation", {}).get("totalResults", "0")
                        }
                    }
                else:
                    return {
                        'google': {
                            'error': '未找到相关信息',
                            'source': 'Google Custom Search'
                        }
                    }
            else:
                return {
                    'google': {
                        'error': f'API请求失败: {response.status_code}',
                        'source': 'Google Custom Search'
                    }
                }
                    
        except requests.RequestException as e:
            logger.error(f'网络请求错误: {str(e)}')
            return {
                'google': {
                    'error': f'网络请求错误: {str(e)}',
                    'source': 'Google Custom Search'
                }
            }
        except json.JSONDecodeError as e:
            logger.error(f'JSON解析错误: {str(e)}')
            return {
                'google': {
                    'error': f'JSON解析错误: {str(e)}',
                    'source': 'Google Custom Search'
                }
            }
        except Exception as e:
            logger.error(f'未知错误: {str(e)}')
            return {
                'google': {
                    'error': f'未知错误: {str(e)}',
                    'source': 'Google Custom Search'
                }
            }

    def search_web_sync(self, query: str) -> Dict:
        """同步版本的网页搜索"""
        return self.search_web(query)

    
    def process_query(self, query: str) -> Dict[str, Any]:
        """同步版本的 API 查询处理，内部调用异步方法"""
        try:
            return self.search_web(query)
        except Exception as e:
            return self.handle_error(e)

    def handle_error(self, error: Exception) -> Dict[str, str]:
        """统一的错误处理方法"""
        return {
            "status": "error",
            "message": str(error),
            "error_type": error.__class__.__name__
        } 
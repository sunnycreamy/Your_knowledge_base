# 智能文档问答系统

基于智谱AI的PDF文档问答系统，支持分类管理和智能问答。

## 功能特点

- 📚 支持PDF文件管理和分类
- 🔍 全局/分类查询模式
- 💡 智能问答功能
- ✅ 文件状态检查
- 🔄 向量数据库管理
- 📤 文件临时上传功能

## 目录结构

your-project-name/
├── m_stereamlit_zhipu.py # 主程序
├── zhipuai_embedding.py # 向量化模块
├── zhipuai_llm.py # LLM接口
├── requirements.txt # 依赖列表
├── .env # 环境变量
├── README.md # 项目说明
└── data_base/ # 数据目录
├── knowledge_db/ # PDF文件存储
└── vector_db/ # 向量数据库
```

## 快速开始

### 1. 环境配置
```bash
# 创建新环境
conda create -n your-env-name python=3.9

# 激活环境
conda activate your-env-name

# 安装依赖包
pip install -r requirements.txt
```

### 2. 配置API密钥
复制环境变量示例文件：
```bash
cp .env.example .env
```
编辑 .env 文件，填入你的 API Key：
```
ZHIPUAI_API_KEY=你的智谱API密钥
```

### 3. 创建必要目录
```bash
mkdir -p data_base/knowledge_db
mkdir -p data_base/vector_db/chroma
```

### 4. 运行应用
```bash
streamlit run m_stereamlit_zhipu.py
```

## 使用指南

### PDF文件管理
1. 创建分类目录：
```bash
mkdir -p data_base/knowledge_db/你的分类名称
```

2. 添加PDF文件：
- 将PDF文件放入对应分类目录
- 或使用界面的临时上传功能

### 知识库管理
1. 检查文件状态：使用侧边栏的"系统维护"
2. 重建知识库：首次使用或添加新文件后

### 问答功能
1. 选择查询模式：
   - 全局查询：搜索所有文档
   - 分类查询：指定分类和文件
2. 选择问答方式：
   - qa_chain：适合文档问答
   - chat_qa_chain：支持上下文对话
   - chat：通用对话模式

## 常见问题

1. **找不到文档内容？**
   - 检查文件是否正确放置
   - 重建向量数据库
   - 确认文件编码格式

2. **回答质量不理想？**
   - 尝试更具体的问题
   - 使用分类查询模式
   - 选择合适的问答方式

## 维护命令

```bash
# 查看环境
conda info --envs

# 查看已安装包
pip list

# 更新依赖
pip install -r requirements.txt --upgrade
```

## 注意事项

1. 确保API密钥配置正确
2. PDF文件需放在正确的分类目录
3. 大文件处理可能需要较长时间
4. 定期备份重要文档
5. 临时上传的文件不会永久保存

## 许可证

MIT License
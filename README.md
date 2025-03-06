#knowledge_base
-这是一个基于Python的智能文档处理系统，集成了Google Drive同步、向量数据库存储和自然语言处理功能，帮助用户高效管理和检索文档信息。
## 功能特点
-Google Drive同步：自动同步指定Google Drive文件夹中的文档
-智能文档处理：支持多种文档格式（PDF、Word、文本等）的解析和处理
-向量数据库存储：使用Chroma DB进行高效的语义检索
-自然语言查询：通过自然语言提问获取文档中的信息
-日志系统：完善的日志记录，便于调试和问题追踪

## 安装指南
前提条件
Python 3.8+
网络连接（用于API调用和Google Drive访问）
Google账户（用于Google Drive API授权）

安装步骤
1. 克隆项目

2. 安装依赖
bash
pip install -r requirements.txt

3. 配置环境变量
编辑.env文件
cp .env.example .env
bash
```
ZHIPUAI_API_KEY=你的智谱API密钥
OLLAMA_URL=http://localhost:11434
```

3. 运行应用
```bash
streamlit run main.py
```

使用指南

模型选择
- 智谱GLM4：需要配置API密钥
- Ollama模型：需要本地运行Ollama服务

对话模式
1. 知识库对话
   - 上传文档到知识库
   - 基于知识库内容进行问答
2. 文档对话
   - 选择特定文档进行对话
3. 自由对话
   - 无限制的AI对话模式
   - 网络连接

文档管理
- 支持新建分类
- 文档上传和管理
- 自动向量化处理
- 手动增量更新数据库

注意事项

1. 首次使用需完成环境配置
2. 确保模型服务正常运行
3. 大文件处理可能需要较长时间
4. 定期维护知识库

## 初始化知识库

首次使用需要创建必要的目录结构：

1. 创建知识库目录
```bash
mkdir -p data_base/knowledge_db
mkdir -p data_base/vector_db
```

2. 添加文档
- 将你的文档文件放入 `data_base/knowledge_db` 目录
- 支持的格式：PDF、Word、TXT 等

3. 构建向量库
```bash
python scripts/init_vector_db.py
```

注意：
- 首次构建向量库可能需要一些时间，取决于文档数量和大小
- 确保已正确配置模型 API（智谱GLM4或Ollama）
- 向量库构建完成后，系统即可进行文档问答

## 目录结构说明
```
data_base/
├── knowledge_db/           # 存放原始文档
│   ├── Autobiography/     # 自传类文档
│   ├── history/          # 历史类文档
│   ├── life_weekly/      # 生活周刊
│   ├── literature/       # 文学作品
│   ├── society/         # 社会类文档
│   └── technology/      # 技术文档
└── vector_db/           # 存放向量化后的数据
```

可以根据个人需求创建不同的文档分类目录。建议的分类方式：
1. 使用英文目录名，避免中文路径可能造成的编码问题




import os

# 创建目录结构
directories = [
    "data_base",
    "data_base/knowledge_db",
    "data_base/vector_db",
    "data_base/vector_db/chroma"
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"创建目录: {directory}")

def check_directories():
    directories = [
        "data_base",
        "data_base/knowledge_db",
        "data_base/vector_db",
        "data_base/vector_db/chroma"
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ {directory} 已创建")
        else:
            print(f"❌ {directory} 未创建")

check_directories() 
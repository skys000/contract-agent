# 劳动合同智能合规审查系统

基于 LangGraph 多智能体循环反思架构与 FAISS 向量知识库检索增强（RAG）的劳动合同智能审查系统。

## 功能特性
1. **多格式合同解析**：支持 `.docx`、`.pdf` 等格式的劳动合同文本自动提取与结构化清洗。
2. **本地法规 RAG 检索**：离线构建本地 FAISS 向量索引，智能匹配《劳动法》、《劳动合同法》及相关司法解释。
3. **LangGraph 协同会审**：通过“初审智能体（Auditor）”与“反思智能体（Critic）”多轮循环判定，消除幻觉。
4. **大屏看板与报告导出**：统计风险项数据，以打字机流式效果呈现审查意见，支持一键导出 Markdown 报告。

## 安装与运行

### 1. 准备环境
确保安装了 Python 3.10+。
```powershell
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\Activate.ps1
```

### 2. 安装依赖
```powershell
pip install -r requirements.txt
```

### 3. 环境变量配置
复制并修改根目录下的 `.env` 文件，填入您的大模型 API 凭证。

### 4. 构建本地向量数据库
将法律法规的原始 `.txt` 文本放入 `data/laws/` 下，运行构建向量索引。

### 5. 启动系统
```powershell
streamlit run app.py
```

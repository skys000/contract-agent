# 劳动合同智能合规审查系统

基于 LangGraph 多智能体循环反思架构与 FAISS 向量知识库检索增强（RAG）的劳动合同智能审查系统。

## 功能特性
1. **多格式合同解析**：支持 `.docx`、`.pdf` 等格式的劳动合同文本自动提取与结构化清洗。
2. **本地法规 RAG 检索**：离线构建本地 FAISS 向量索引，智能匹配《劳动法》、《劳动合同法》及相关司法解释。
3. **LangGraph 协同会审**：通过“初审智能体（Auditor）”与“反思智能体（Critic）”多轮循环判定，消除幻觉。
4. **大屏看板与报告导出**：统计风险项数据，以打字机流式效果呈现审查意见，支持一键导出 Markdown 报告。
5. **可选 MinerU 增强解析**：支持将 MinerU 安装到独立虚拟环境中，并可按需启用 NVIDIA GPU / CUDA 加速。

## 项目结构

```text
contract-agent/
├── app.py                  # Streamlit 前端入口：审计工作区、法律咨询、看板、法条文库
├── src/
│   ├── agent.py            # LangGraph Auditor/Critic 双智能体审查工作流
│   ├── retriever.py        # 法规 RAG 构建、FAISS 查询、多路召回与重排
│   ├── parser.py           # 合同解析、MinerU CLI 增强解析、脱敏与元数据提取
│   └── database.py         # SQLite 审计记录与运营看板统计
├── data/
│   ├── laws/               # 本地劳动法律法规文件
│   ├── faiss_index/        # 本地 FAISS 向量索引
│   └── temp/               # 上传与解析过程临时目录
├── .env.example            # 环境变量示例
└── requirements.txt        # 项目主虚拟环境依赖
```

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

### 2.1 可选：安装 MinerU 本地增强解析器
系统默认依赖保持轻量，可直接使用基础解析器完成 `.docx` 和 `.pdf` 合同解析。

由于 MinerU 依赖较重，且可能升级 `numpy`、`openai`、`pypdf`、`python-docx` 等核心包，推荐将 MinerU 安装到项目主虚拟环境之外的独立环境中。以下命令均在项目根目录执行：
```powershell
# 在项目目录外创建独立 MinerU 环境
python -m venv ..\mineru-venv

# 安装 uv
..\mineru-venv\Scripts\python.exe -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple
..\mineru-venv\Scripts\python.exe -m pip install uv -i https://mirrors.aliyun.com/pypi/simple

# 安装 MinerU
..\mineru-venv\Scripts\uv.exe pip install -U "mineru[all]" --python ..\mineru-venv\Scripts\python.exe -i https://mirrors.aliyun.com/pypi/simple
```

国内网络建议配置模型源：
```powershell
$env:MINERU_MODEL_SOURCE = "modelscope"
```

在项目 `.env` 中启用 MinerU：
```env
DOCUMENT_PARSER=mineru_cli
MINERU_CLI_COMMAND=..\mineru-venv\Scripts\mineru.exe
MINERU_MODEL_SOURCE=modelscope
MINERU_BACKEND=pipeline
MINERU_METHOD=auto
MINERU_LANG=ch
MINERU_FORMULA=false
MINERU_TABLE=true
MINERU_TIMEOUT_SECONDS=600
```

MinerU CLI 验证命令：
```powershell
..\mineru-venv\Scripts\mineru.exe --help
```

手动测试解析：
```powershell
$env:MINERU_MODEL_SOURCE = "modelscope"
..\mineru-venv\Scripts\mineru.exe -p data\temp\sample.pdf -o data\mineru_output -b pipeline -m auto -l ch -f false -t true
```

可选：启用 NVIDIA GPU / CUDA 加速。先确认本机显卡和独立 MinerU 环境中的 PyTorch 状态：
```powershell
nvidia-smi
..\mineru-venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

如果输出仍为 CPU 版 PyTorch，例如 `torch.cuda.is_available()` 为 `False`，可在独立 MinerU 环境中安装与显卡驱动兼容的 CUDA 版 PyTorch。以下示例适用于已确认支持 CUDA 12.8 的 Windows/NVIDIA 环境：
```powershell
..\mineru-venv\Scripts\python.exe -m pip install --force-reinstall "..\torch-2.8.0+cu128-cp310-cp310-win_amd64.whl" "..\torchvision-0.23.0+cu128-cp310-cp310-win_amd64.whl"
```

也可直接从 PyTorch 官方源安装：
```powershell
..\mineru-venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

本项目在 RTX 4060 Laptop GPU 上验证：同一份 12 页 PDF，CPU 版 `pipeline` 约 74.81 秒；安装 `torch 2.8.0+cu128` 后，缓存后二次运行约 63.64 秒，OCR 阶段吞吐明显提升。

说明：
- MinerU 为可选增强依赖，不安装也不影响系统基础运行。
- 推荐使用独立 MinerU 虚拟环境，不建议直接安装进项目主 `venv`。
- 未检测到 MinerU CLI、解析失败或超时时，系统会自动回退到基础解析器。
- 本方案使用本地 MinerU CLI，不使用 `langchain-mineru` API Loader，不会把原始合同上传到 MinerU 云端 API。
- Windows 环境建议使用 Python 3.10-3.12。
- GPU / CUDA 加速仅作为可选优化，不写入项目主依赖；如 CUDA 版 PyTorch 与 MinerU 依赖冲突，可删除并重建独立 `..\mineru-venv`。

### 3. 环境变量配置
复制并修改根目录下的 `.env` 文件，填入您的大模型 API 凭证。

### 4. 构建本地向量数据库
将法律法规的原始 `.txt` 文本放入 `data/laws/` 下，运行构建向量索引。

### 5. 启动系统
```powershell
streamlit run app.py
```

### 6. 编译检查
```powershell
.\venv\Scripts\python.exe -m py_compile app.py src\parser.py src\retriever.py src\agent.py src\database.py
```

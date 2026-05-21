# -*- coding: utf-8 -*-
"""
模块名: src/parser.py
作用: 提供对 .docx 和 .pdf 格式劳动合同文件的底层读取与段落空白行清洗提取功能。
"""

import os
import re
import json
import shutil
import subprocess
import tempfile
from docx import Document
import pypdf

# 保存最近一次解析链路的前端提示语，由 app.py 在上传解析后读取并展示给用户
LAST_PARSER_MESSAGE = ""

def _parser_log(message: str) -> None:
    """
    统一输出解析链路日志，便于在 Streamlit 控制台确认当前使用 legacy、MinerU、CPU 或 GPU。
    """
    # 使用 flush=True 保证 Streamlit 控制台能及时看到当前解析器、后端和 GPU 状态
    print(f"[Parser] {message}", flush=True)

def _resolve_command_path(command: str) -> str:
    """
    将 MinerU CLI 命令解析为可读路径，用于日志展示与运行环境定位。
    """
    # 优先使用系统 PATH 或可执行文件路径解析命令
    resolved = shutil.which(command)
    if resolved:
        return resolved
    # 如果 PATH 中找不到，则转成绝对路径用于日志展示，方便排查配置问题
    return os.path.abspath(command)

def _get_mineru_runtime_status(cli_command: str) -> str:
    """
    通过 MinerU 所在虚拟环境的 Python 检测 torch / CUDA / GPU 状态。

    该检测只用于控制台展示，不参与业务分支判断，避免 GPU 检测失败影响合同解析回退机制。
    """
    # 先定位 mineru.exe，再推导同一虚拟环境下的 python.exe
    cli_path = _resolve_command_path(cli_command)
    python_path = os.path.join(os.path.dirname(cli_path), "python.exe")
    if not os.path.exists(python_path):
        return f"未找到 MinerU 环境 Python: {python_path}"
    # 在 MinerU 独立环境中执行一段极小 Python 脚本，读取 torch 与 CUDA 状态
    script = (
        "import json\n"
        "try:\n"
        "    import torch\n"
        "    print(json.dumps({"
        "'torch': torch.__version__, "
        "'cuda': torch.version.cuda, "
        "'cuda_available': torch.cuda.is_available(), "
        "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda'"
        "}, ensure_ascii=False))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'error': str(e)}, ensure_ascii=False))\n"
    )
    try:
        # 用子进程隔离检测逻辑，避免将 MinerU 环境依赖导入项目主 venv
        result = subprocess.run(
            [python_path, "-c", script],
            check=False,
            timeout=15,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        # MinerU 环境脚本输出 JSON，取最后一行解析，避免第三方库警告干扰前置输出
        status = json.loads((result.stdout or "{}").strip().splitlines()[-1])
        if status.get("error"):
            return f"torch 检测失败: {status['error']}"
        # 拼接为单行控制台日志，便于用户判断当前是否启用了 CUDA/GPU
        return (
            f"torch={status.get('torch')}, "
            f"cuda={status.get('cuda')}, "
            f"cuda_available={status.get('cuda_available')}, "
            f"device={status.get('device')}"
        )
    except Exception as e:
        # 检测失败不影响合同解析，只把异常文本返回给日志
        return f"torch 检测失败: {e}"

def parse_docx(file_path: str) -> str:
    """
    提取并清洗 Word (.docx) 文档中的全部段落文本
    :param file_path: Word 文件的绝对或相对路径
    :return: 清洗合并后的合同文本字符串，以换行符分隔
    """
    # 初始化 Word 文档解析对象
    doc = Document(file_path)
    full_text_list = []
    
    # 遍历文档中的每一个自然段
    for paragraph in doc.paragraphs:
        # 去除段落前后的空白字符
        clean_text = paragraph.text.strip()
        # 仅保留非空段落，过滤掉文档中的冗余空行
        if clean_text:
            full_text_list.append(clean_text)
            
    # 用换行符连接所有有内容的段落并返回
    return "\n".join(full_text_list)

def parse_pdf(file_path: str) -> str:
    """
    读取并过滤 PDF (.pdf) 文档中的文本数据，保持基本的物理分段
    :param file_path: PDF 文件的绝对或相对路径
    :return: 经过清洗、去空格处理后的 PDF 文本字符串
    """
    full_text_list = []
    
    # 以二进制只读模式打开 PDF 文件
    with open(file_path, "rb") as pdf_file:
        # 初始化 PDF 阅读器
        reader = pypdf.PdfReader(pdf_file)
        
        # 遍历 PDF 的每一页
        for page in reader.pages:
            # 提取当前页的原始文本
            extracted_text = page.extract_text()
            if extracted_text:
                # 对提取出的多行文本按换行拆分，进行精细化去空格清洗
                lines = [line.strip() for line in extracted_text.split("\n") if line.strip()]
                # 将本页清洗后的文本行用换行符重新连接，并加入总列表
                full_text_list.append("\n".join(lines))
                
    # 用换行符拼接所有页面提取的内容并返回
    return "\n".join(full_text_list)

def parse_with_legacy(file_path: str) -> str:
    """
    使用项目原有轻量解析器提取合同文本
    :param file_path: 输入文件的绝对或相对路径
    :return: 解析清洗后的文本内容
    :raises ValueError: 当上传不支持的格式或不存在文件时抛出异常
    """
    # 先校验文件是否存在，避免后续解析库抛出不友好的底层错误
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到指定的合同文件: {file_path}")
        
    # 根据扩展名选择项目内置轻量解析方式
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == ".docx":
        # Word 文档直接通过 python-docx 提取自然段文本
        return parse_docx(file_path)
    elif file_extension == ".pdf":
        # PDF 文档通过 pypdf 逐页抽取文本
        return parse_pdf(file_path)
    elif file_extension == ".txt":
        # 法规库中存在精简 txt 条文，legacy 解析器需要直接支持
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_extension == ".doc":
        # 对旧版 Word 格式进行友情拦截，引导用户转换格式
        raise ValueError("系统暂不支持 .doc 格式，请在 Office 中打开并另存为 .docx 格式后再行上传。")
    else:
        # 对不支持的非法扩展名进行强拦截
        raise ValueError(f"系统不支持的文件格式: {file_extension}。请上传 .docx 或 .pdf 合同文档。")

def _normalize_bool_env(value: str, default: bool) -> bool:
    """
    解析布尔型环境变量，兼容 true/yes/on 等常见写法。
    """
    # 环境变量未配置时返回调用方提供的默认值
    if value is None:
        return default
    # 将常见真值写法统一识别为 True，其余值均视为 False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

def _clean_mineru_markdown(text: str) -> str:
    """
    清理 MinerU 输出 Markdown 中对合同审查无用的图片引用和调试折叠块。
    """
    # 删除图片引用，避免合同审查模型把截图路径当作合同正文
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    # 删除 MinerU 可能输出的折叠调试块，保留纯文本/表格内容
    text = re.sub(r"<details>.*?</details>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 合并过多空行，使审查输入更紧凑
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _read_mineru_content_list(file_path: str) -> str:
    """
    读取 MinerU 的结构化 content_list.json，并按文本、列表、表格等元素拼接为审查可用文本。
    """
    # 读取 MinerU 结构化结果文件，通常是一个由多种内容元素组成的列表
    with open(file_path, "r", encoding="utf-8") as f:
        content_items = json.load(f)
    text_parts = []
    # 若输出结构异常，直接返回空文本，让上层继续尝试其他输出文件或触发回退
    if not isinstance(content_items, list):
        return ""
    for item in content_items:
        # 跳过非字典元素，避免第三方输出异常导致解析中断
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        # 文本、列表、公式和代码块可直接按文本拼接
        if item_type in {"text", "list", "equation", "code"} and item.get("text"):
            text_parts.append(str(item["text"]).strip())
        elif item_type == "table":
            # 表格元素按标题、表体、脚注顺序拼接，尽量保留合同表格信息
            table_parts = []
            for caption in item.get("table_caption", []) or []:
                table_parts.append(str(caption).strip())
            if item.get("table_body"):
                table_parts.append(str(item["table_body"]).strip())
            for footnote in item.get("table_footnote", []) or []:
                table_parts.append(str(footnote).strip())
            if table_parts:
                text_parts.append("\n".join(table_parts))
    # 过滤空片段后合并为审查模型可直接读取的纯文本
    return "\n".join(part for part in text_parts if part).strip()

def _read_mineru_output(output_dir: str) -> str:
    """
    从 MinerU 输出目录中选择可用解析结果。

    优先使用 Markdown；若不存在 Markdown，则回退读取 content_list.json，保证不同 MinerU 输出形态均可兼容。
    """
    markdown_files = []
    content_list_files = []
    # 遍历 MinerU 输出目录，收集 Markdown 和结构化 JSON 两类可读结果
    for root, _, files in os.walk(output_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            lower_name = file_name.lower()
            if lower_name.endswith(".md"):
                markdown_files.append(file_path)
            elif lower_name.endswith("_content_list.json"):
                content_list_files.append(file_path)
    # 多个 Markdown 同时存在时，优先选择体积最大的文件，通常内容最完整
    if markdown_files:
        markdown_files.sort(key=lambda path: os.path.getsize(path), reverse=True)
        with open(markdown_files[0], "r", encoding="utf-8") as f:
            return _clean_mineru_markdown(f.read())
    # 如果没有 Markdown，则读取 content_list.json 作为兼容回退
    if content_list_files:
        content_list_files.sort(key=lambda path: os.path.getsize(path), reverse=True)
        return _read_mineru_content_list(content_list_files[0])
    # 未找到任何可用输出时返回空字符串，由调用方抛出明确错误
    return ""

def parse_with_mineru_cli(file_path: str) -> str:
    """
    调用本地 MinerU CLI 对 PDF/DOCX 合同进行增强解析。

    MinerU 作为可选依赖运行在独立虚拟环境中；本函数只负责命令行调用和输出读取，
    失败时由上层 `extract_contract_text` 自动回退到 legacy 解析器。
    """
    # 调用外部 CLI 前先确认文件存在，避免命令行报错难以定位
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到指定的合同文件: {file_path}")
    # MinerU CLI 仅用于合同上传场景的 docx/pdf，不处理法规库 txt 文件
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension not in [".docx", ".pdf"]:
        raise ValueError("MinerU 增强解析仅用于 .docx 或 .pdf 合同文档。")

    # 计算项目根目录，并把 MinerU 临时输出限制在 data/temp/mineru 下
    project_root = os.path.dirname(os.path.dirname(__file__))
    mineru_root = os.path.join(project_root, "data", "temp", "mineru")
    os.makedirs(mineru_root, exist_ok=True)

    # 从环境变量读取 MinerU CLI 参数，保持增强解析能力可配置、可关闭
    cli_command = os.getenv("MINERU_CLI_COMMAND", "mineru")
    backend = os.getenv("MINERU_BACKEND", "pipeline")
    method = os.getenv("MINERU_METHOD", "auto")
    language = os.getenv("MINERU_LANG", "ch")
    formula = str(_normalize_bool_env(os.getenv("MINERU_FORMULA"), False)).lower()
    table = str(_normalize_bool_env(os.getenv("MINERU_TABLE"), True)).lower()
    timeout = int(os.getenv("MINERU_TIMEOUT_SECONDS", "300"))
    cli_path = _resolve_command_path(cli_command)

    # 复制当前进程环境，并补充 MinerU 本地模型源和服务超时设置
    env = os.environ.copy()
    if os.getenv("MINERU_MODEL_SOURCE"):
        env["MINERU_MODEL_SOURCE"] = os.getenv("MINERU_MODEL_SOURCE")
    env["MINERU_TASK_RESULT_TIMEOUT_SECONDS"] = str(timeout)
    env["MINERU_LOCAL_API_STARTUP_TIMEOUT_SECONDS"] = str(timeout)
    _parser_log(
        "使用 MinerU CLI 增强解析: "
        f"cli={cli_path}, backend={backend}, method={method}, lang={language}, "
        f"formula={formula}, table={table}, model_source={env.get('MINERU_MODEL_SOURCE', 'default')}, "
        f"timeout={timeout}s"
    )
    _parser_log(f"MinerU 运行环境: {_get_mineru_runtime_status(cli_command)}")

    # 每次解析使用独立临时输出目录，避免不同上传文件的 MinerU 输出互相污染
    with tempfile.TemporaryDirectory(dir=mineru_root) as output_dir:
        # 按 MinerU CLI 参数格式组装命令，不通过 shell 拼接，降低路径空格和转义风险
        command = [
            cli_command,
            "-p", file_path,
            "-o", output_dir,
            "-b", backend,
            "-m", method,
            "-l", language,
            "-f", formula,
            "-t", table
        ]
        # 同步执行 MinerU CLI；stdout/stderr 捕获后由异常机制交给上层 fallback 处理
        subprocess.run(
            command,
            check=True,
            timeout=timeout,
            env=env,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        # 读取 MinerU 生成的 Markdown 或结构化 JSON 内容
        text = _read_mineru_output(output_dir)
        if not text:
            raise ValueError("MinerU 未生成可用的 Markdown 或 content_list.json 文本。")
        # 返回清洗后的合同文本，供后续脱敏、元数据提取和智能体审查使用
        return text

def get_last_parser_message() -> str:
    """
    返回最近一次解析流程给前端展示的状态消息。
    """
    return LAST_PARSER_MESSAGE

def extract_contract_text(file_path: str, prefer_mineru: bool = False) -> str:
    """
    合同文本解析的中心分流函数，依据文件扩展名分流处理并进行格式拦截
    :param file_path: 输入文件的绝对或相对路径
    :param prefer_mineru: 是否优先使用本地 MinerU CLI 增强解析
    :return: 解析清洗后的合同原文内容
    :raises ValueError: 当上传不支持的格式或不存在文件时抛出异常
    """
    global LAST_PARSER_MESSAGE
    # 每次解析前清空上一次状态，避免前端展示旧解析提示
    LAST_PARSER_MESSAGE = ""
    # 基础文件存在性校验
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到指定的合同文件: {file_path}")
    # 读取扩展名和解析器配置，决定是否尝试 MinerU 增强解析
    file_extension = os.path.splitext(file_path)[1].lower()
    parser_name = os.getenv("DOCUMENT_PARSER", "legacy").strip().lower()
    _parser_log(f"解析请求: file={os.path.basename(file_path)}, ext={file_extension}, prefer_mineru={prefer_mineru}, DOCUMENT_PARSER={parser_name}")
    if prefer_mineru and parser_name == "mineru_cli" and file_extension in [".docx", ".pdf"]:
        cli_command = os.getenv("MINERU_CLI_COMMAND", "mineru")
        # 先检测 CLI 是否可执行，避免每次都进入外部命令失败流程
        cli_path = shutil.which(cli_command)
        if cli_path:
            try:
                # MinerU 成功时直接返回增强解析文本
                text = parse_with_mineru_cli(file_path)
                LAST_PARSER_MESSAGE = "已使用本地 MinerU 增强解析合同内容。"
                _parser_log(LAST_PARSER_MESSAGE)
                return text
            except Exception as e:
                # MinerU 异常时记录提示，并继续回退到基础解析器
                LAST_PARSER_MESSAGE = f"MinerU 增强解析不可用，已自动回退至基础解析器：{e}"
                _parser_log(LAST_PARSER_MESSAGE)
        else:
            # CLI 不存在时不报错中断，而是提示前端并回退 legacy
            LAST_PARSER_MESSAGE = "未检测到 MinerU CLI，已自动使用基础解析器。"
            _parser_log(f"{LAST_PARSER_MESSAGE} cli={_resolve_command_path(cli_command)}")
    else:
        # 未开启 MinerU、非上传入口或文件格式不适用时走基础解析器
        _parser_log("使用基础解析器。")
    # 统一 legacy 回退出口，保证系统基础功能不依赖 MinerU
    return parse_with_legacy(file_path)

def desensitize_text(text: str) -> str:
    """
    根据 SRS 5.3 节的隐私安全规定，对文本中的敏感信息进行本地正则脱敏替换
    :param text: 原始合同文本
    :return: 脱敏后的文本，敏感数据已被替换为占位符
    """
    # 1. 匹配 18 位身份证号码
    id_pattern = r'\d{17}[\dXx]'
    text = re.sub(id_pattern, "[USER_ID]", text)
    
    # 2. 匹配 11 位手机号码（限制前后不为数字，防止错切其他编号）
    phone_pattern = r'(?<!\d)1[3-9]\d{9}(?!\d)'
    text = re.sub(phone_pattern, "[USER_PHONE]", text)
    
    return text

def extract_metadata(text: str) -> dict:
    """
    利用启发式正则表达式规则从合同文本中提取关键元数据（甲方、乙方等）
    :param text: 合同全文文本
    :return: 包含元数据字典
    """
    metadata = {
        "party_a": "未知用人单位",
        "party_b": "未知劳动者",
        "duration": "未知期限",
        "salary": "未明确约定"
    }
    
    # 1. 匹配 甲方 (用人单位)
    party_a_patterns = [
        r'(?:甲方|用人单位)(?:\s*[\(（](?:用人单位|甲方)[\)）])?\s*[:：\s]*(.*?公司|.*?集团|.*?厂|.*?店|.*?医院|.*?学校|.*?局)',
        r'(?:甲方|用人单位)(?:\s*[\(（](?:用人单位|甲方)[\)）])?\s*[:：\s]*([^\n，。；\s]+)'
    ]
    for pattern in party_a_patterns:
        match = re.search(pattern, text)
        if match:
            val = match.group(1).strip()
            if len(val) >= 4 and not any(kw in val for kw in ["乙方", "劳动者", "工作内容", "合同期限"]):
                metadata["party_a"] = val
                break
                
    # 2. 匹配 乙方 (劳动者)
    party_b_patterns = [
        r'(?:乙方|劳动者)(?:\s*[\(（](?:劳动者|乙方)[\)）])?\s*[:：\s]*([\u4e00-\u9fa5]{2,4})(?:\s|[,，。；\n]|$)',
        r'(?:乙方|劳动者)\s*[:：\s]*([\u4e00-\u9fa5]{2,4})(?:\s|[,，。；\n]|$)'
    ]
    blacklist_b = ["在三年", "在合同", "在工作", "在试用", "劳动合", "用人单", "工作内", "乙方在", "劳动者", "被聘用"]
    for pattern in party_b_patterns:
        for match in re.finditer(pattern, text):
            val = match.group(1).strip()
            if val and not any(bk in val for bk in blacklist_b):
                metadata["party_b"] = val
                break
        if metadata["party_b"] != "未知劳动者":
            break
                
    # 3. 匹配合同期限
    duration_patterns = [
        r'合同期(?:限)?(?:为|是|自)\s*([^\n，。；]+)',
        r'期限(?:为|是|自)\s*([^\n，。；]+)'
    ]
    for pattern in duration_patterns:
        match = re.search(pattern, text)
        if match:
            metadata["duration"] = match.group(1).strip()
            break
            
    # 4. 匹配薪资额度
    salary_patterns = [
        r'(?:工资|薪资|报酬|底薪)[^，。；\n]*?(?:为|是)[^\d，。；\n]*?(\d+)\s*(?:元|元/月)',
        r'(?:基本工资|基本薪酬)[为是]\s*(\d+)\s*(?:元|元/月)'
    ]
    for pattern in salary_patterns:
        matches = re.findall(pattern, text)
        if matches:
            # 去重并保持先后顺序
            unique_matches = list(dict.fromkeys(matches))
            metadata["salary"] = " / ".join(unique_matches) + " 元/月"
            break
            
    return metadata


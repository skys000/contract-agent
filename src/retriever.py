# -*- coding: utf-8 -*-
"""
模块名: src/retriever.py
作用: 扫描 data/laws 目录，利用 parser.py 提取法条文本并利用 FAISS 构建本地 RAG 向量知识库。
"""

import os
import re
import sys
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

# 确保能正常导入同一目录下的 parser 模块
sys.path.append(os.path.dirname(__file__))
from parser import extract_contract_text

ARTICLE_PATTERN = r"第[一二三四五六七八九十百零〇]+条"

def _get_embedding_model_name() -> str:
    model_name = os.getenv("EMBEDDING_MODEL_NAME")
    if not model_name:
        raise ValueError("未配置向量模型名称，请在 .env 中设置 EMBEDDING_MODEL_NAME。")
    return model_name

def _has_risk_context(text: str, keywords: List[str], safe_markers: List[str]) -> bool:
    compact_text = re.sub(r"\s+", "", text)
    if not any(keyword in compact_text for keyword in keywords):
        return False
    return not any(marker in compact_text for marker in safe_markers)

def _build_search_queries(contract_text: str) -> List[str]:
    normalized_text = re.sub(r"\n\s*", "\n", contract_text)
    clauses = [
        clause.strip()
        for clause in re.split(r"(?=\n?第[一二三四五六七八九十百零〇\d]+条)", normalized_text)
        if len(clause.strip()) > 20
    ]
    if not clauses:
        clauses = [p.strip() for p in contract_text.split('\n') if len(p.strip()) > 20]
    priority_keywords = [
        "试用期", "社会保险", "社保", "加班", "工作时间", "休息休假", "违约金", "服务期",
        "培训", "竞业", "保密", "解除", "终止", "工资", "薪资", "劳动报酬", "工作地点",
        "工作内容", "劳动保护", "劳动条件", "知识产权", "专利", "著作权", "软件著作权",
        "职务成果", "职务发明", "职务作品", "源代码", "代码", "算法", "开源", "个人作品"
    ]
    priority_clauses = [clause for clause in clauses if any(keyword in clause for keyword in priority_keywords)]
    fallback_clauses = [clause for clause in clauses if clause not in priority_clauses]
    topic_queries = []
    compact_text = re.sub(r"\s+", "", contract_text)
    if "试用期" in compact_text:
        topic_queries.extend([
            "试用期 不得超过 一个月 二个月 六个月 劳动合同期限",
            "试用期 工资 不得低于 百分之八十 最低工资标准"
        ])
    if any(keyword in compact_text for keyword in ["社会保险", "社保", "五险"]):
        topic_queries.extend([
            "劳动合同 必备条款 社会保险",
            "用人单位 劳动者 必须依法参加社会保险 缴纳社会保险费",
            "《中华人民共和国社会保险法》第五十八条 自用工之日起三十日 社会保险登记"
        ])
    if any(keyword in compact_text for keyword in ["工作时间", "休息休假", "加班", "工时"]):
        topic_queries.extend([
            "每日工作时间不超过八小时 平均每周工作时间 工时制度",
            "用人单位应当保证劳动者每周至少休息一日",
            "延长工作时间 加班工资 支付工资报酬"
        ])
    if any(keyword in compact_text for keyword in ["年休假", "带薪年休假", "未休年休假"]):
        topic_queries.extend([
            "职工连续工作一年以上 享受带薪年休假",
            "单位根据生产工作的具体情况 职工本人意愿 统筹安排年休假",
            "不能安排年休假 应当按照日工资收入支付年休假工资报酬"
        ])
    if any(keyword in compact_text for keyword in ["劳动报酬", "工资", "薪资", "底薪"]):
        topic_queries.append("工资 货币形式 按月支付 不得克扣 无故拖欠")
    if _has_risk_context(contract_text, ["违约金", "服务期", "培训费"], ["不约定由乙方承担违法违约金", "依法发生专项培训服务期"]):
        topic_queries.extend([
            "专项培训费用 专业技术培训 服务期 违约金",
            "不得与劳动者约定由劳动者承担违约金"
        ])
    if any(keyword in compact_text for keyword in ["商业秘密", "保密"]):
        topic_queries.append("商业秘密 知识产权 保密事项")
    if any(keyword in compact_text for keyword in ["竞业", "竞业限制"]):
        topic_queries.extend([
            "竞业限制 商业秘密 知识产权 保密事项 按月给予经济补偿",
            "竞业限制 人员 范围 地域 期限 不得超过二年",
            "竞业限制 未约定经济补偿 十二个月平均工资 30% 最低工资标准"
        ])
    if any(keyword in compact_text for keyword in ["知识产权", "专利", "著作权", "软件著作权", "职务成果", "职务发明", "职务作品", "源代码", "代码", "算法", "开源", "个人作品"]):
        topic_queries.extend([
            "《中华人民共和国专利法》第六条 职务发明创造 主要利用本单位物质技术条件 专利申请权",
            "《中华人民共和国著作权法》第十八条 职务作品 计算机软件 主要利用法人组织物质技术条件",
            "《计算机软件保护条例》第十三条 软件著作权 本职工作 开发目标 物质技术条件"
        ])
    if any(keyword in compact_text for keyword in ["工伤", "职业伤害", "意外事故", "工伤保险"]):
        topic_queries.extend([
            "工伤保险 用人单位 缴纳工伤保险费 职工不缴纳",
            "职工有下列情形之一的应当认定为工伤 工作时间工作场所",
            "工伤保险待遇 治疗工伤的医疗费用 停工留薪期"
        ])
    if any(keyword in compact_text for keyword in ["职业病", "职业危害", "职业健康", "职业禁忌", "防护用品", "劳动防护"]):
        topic_queries.extend([
            "职业病危害 用人单位 职业健康检查 费用由用人单位承担",
            "职业病危害 防护设施 个人使用的职业病防护用品 劳动者保护权利",
            "不得安排孕期哺乳期女职工从事接触职业病危害作业"
        ])
    if any(keyword in compact_text for keyword in ["女职工", "女员工", "怀孕", "产期", "哺乳期", "产假", "三期", "孕期"]):
        topic_queries.extend([
            "女职工在孕期产期哺乳期 用人单位不得解除劳动合同",
            "女职工劳动保护 产假 生育享受",
            "不得安排女职工 怀孕 哺乳 劳动强度 夜班"
        ])
    if any(keyword in compact_text for keyword in ["工作内容", "工作地点"]):
        topic_queries.append("劳动合同应当具备 工作内容 工作地点")
    if any(keyword in compact_text for keyword in ["劳动保护", "劳动条件"]):
        topic_queries.append("劳动保护 劳动条件 职业危害防护")
    if _has_risk_context(contract_text, ["解除", "终止"], ["依照中华人民共和国劳动合同法", "依照《中华人民共和国劳动合同法》", "不得违法解除", "依法应支付经济补偿"]):
        topic_queries.append("解除劳动合同 终止劳动合同 经济补偿")
    if any(keyword in compact_text for keyword in ["劳动争议", "仲裁", "调解", "诉讼", "争议处理"]):
        dispute_topic_queries = [
            "劳动争议 因确认劳动关系 劳动合同 劳动报酬 工伤医疗费 经济补偿",
            "发生劳动争议 当事人可以申请调解 仲裁 提起诉讼",
            "劳动争议申请仲裁的时效期间为一年"
        ]
        topic_queries = dispute_topic_queries + topic_queries
    queries = topic_queries + priority_clauses + fallback_clauses
    return list(dict.fromkeys(queries[:24] or [contract_text]))

def _is_substantive_law_chunk(text: str) -> bool:
    compact_text = re.sub(r"\s+", "", text)
    if not re.search(ARTICLE_PATTERN, compact_text):
        return False
    if "目录" in compact_text and len(re.findall(ARTICLE_PATTERN, compact_text)) <= 2:
        return False
    if "第一章总则" in compact_text and len(compact_text) < 260:
        return False
    return True

def _law_chunk_keyword_score(query_text: str, law_text: str) -> int:
    compact_query = re.sub(r"\s+", "", query_text)
    compact_law = re.sub(r"\s+", "", law_text)
    keyword_groups = [
        ["试用期", "不得超过", "一个月", "二个月", "六个月", "劳动合同期限"],
        ["试用期", "工资", "百分之八十", "最低工资标准"],
        ["社会保险", "社保", "缴纳社会保险"],
        ["工作时间", "休息休假", "延长工作时间", "加班"],
        ["工资", "劳动报酬", "报酬", "货币形式"],
        ["违约金", "服务期", "专项培训"],
        ["竞业限制", "商业秘密", "保密事项", "经济补偿"],
        ["竞业限制", "高级管理人员", "高级技术人员", "不得超过二年"],
        ["竞业限制", "十二个月平均工资", "30%", "最低工资标准"],
        ["知识产权", "职务成果", "职务发明", "职务作品"],
        ["职务发明创造", "物质技术条件", "专利申请权"],
        ["职务作品", "计算机软件", "著作权"],
        ["软件著作权", "本职工作", "开发目标", "物质技术条件"],
        ["工作内容", "工作地点", "必备条款"],
        ["劳动保护", "劳动条件", "职业危害"],
        ["解除劳动合同", "终止劳动合同", "经济补偿"],
        ["社会保险", "自用工之日起三十日", "社会保险登记"],
        ["社会保险", "按时足额缴纳社会保险费", "按月"],
        ["工伤", "工伤保险", "认定为工伤", "工伤保险待遇"],
        ["女职工", "孕期", "产期", "哺乳期", "不得解除"],
        ["产假", "生育", "女职工劳动保护"],
        ["加班工资", "延长工作时间", "支付工资报酬", "百分之一百五十"],
        ["每日工作时间", "每周工作时间", "不超过八小时", "四十四小时"],
        ["年休假", "带薪年休假", "未休年休假", "工资报酬"],
        ["职业病危害", "职业健康检查", "防护用品", "职业禁忌"],
        ["劳动争议", "调解", "仲裁", "诉讼", "仲裁时效"]
    ]
    score = 0
    for group in keyword_groups:
        query_hit = any(keyword in compact_query for keyword in group)
        law_hit = any(keyword in compact_law for keyword in group)
        if query_hit and law_hit:
            score += 3
    all_keywords = [keyword for group in keyword_groups for keyword in group]
    exact_overlap = [keyword for keyword in all_keywords if keyword in compact_query and keyword in compact_law]
    return score + min(len(exact_overlap), 5)

def _law_chunk_penalty(query_text: str, law_text: str) -> float:
    compact_query = re.sub(r"\s+", "", query_text)
    compact_law = re.sub(r"\s+", "", law_text)
    penalty = 0.0
    generic_patterns = [
        "适用本法", "国家提倡", "社会保险水平应当与社会经济发展水平",
        "劳动合同分为固定期限", "集体合同签订后", "非全日制用工",
        "劳务派遣单位", "依法建立和完善规章制度",
        "国家工作人员在社会保险管理、监督工作中滥用职权",
        "国务院人事部门、国务院劳动保障部门依据职权"
    ]
    if any(pattern in compact_law for pattern in generic_patterns):
        penalty += 0.45
    catch_all_patterns = [
        "劳动合同应当具备以下条款",
        "用人单位的名称、住所和法定代表人或者主要负责人",
        "社会保险制度坚持广覆盖、保基本、多层次、可持续的方针",
        "中华人民共和国境内的用人单位和个人依法缴纳社会保险费"
    ]
    if any(pattern in compact_law for pattern in catch_all_patterns):
        penalty += 0.6
    female_worker_markers = ["女职工", "女员工", "怀孕", "产期", "哺乳期", "孕期", "产假", "三期"]
    if "女职工" in compact_law and not any(marker in compact_query for marker in female_worker_markers):
        penalty += 0.55
    if "未成年" in compact_law and "未成年" not in compact_query:
        penalty += 0.55
    if "工会" in compact_law and "工会" not in compact_query:
        penalty += 0.35
    social_violation_markers = ["不缴", "未缴", "无故不缴", "未依法缴纳", "社保豁免"]
    if any(pattern in compact_law for pattern in ["无故不缴纳社会保险费", "未依法为劳动者缴纳社会保险费"]):
        if not any(marker in compact_query for marker in social_violation_markers):
            penalty += 0.65
    termination_violation_markers = ["违法解除", "单方面解除", "随时解除", "不支付经济补偿", "辞退", "开除"]
    if any(pattern in compact_law for pattern in ["可以解除劳动合同", "给予经济补偿", "支付经济补偿"]):
        if not any(marker in compact_query for marker in termination_violation_markers):
            penalty += 0.35
    return penalty

def _law_source_adjustment(query_text: str, doc: LangchainDocument) -> float:
    compact_query = re.sub(r"\s+", "", query_text)
    compact_law = re.sub(r"\s+", "", doc.page_content)
    source = doc.metadata.get("source", "")
    adjustment = 0.0
    if any(keyword in compact_query for keyword in ["劳动争议", "仲裁", "调解", "诉讼", "争议处理"]):
        if "劳动争议调解仲裁法" in source or "最高法劳动争议司法解释" in source:
            adjustment -= 0.65
    if any(keyword in compact_query for keyword in ["职业病", "职业危害", "职业健康", "职业禁忌", "防护用品", "劳动防护"]):
        if "职业病防治法" in source:
            adjustment -= 0.55
    if any(keyword in compact_query for keyword in ["年休假", "带薪年休假", "未休年休假"]):
        if "职工带薪年休假条例" in source:
            adjustment -= 0.55
    if any(keyword in compact_query for keyword in ["知识产权", "专利", "著作权", "软件著作权", "职务成果", "职务发明", "职务作品", "源代码", "代码", "算法"]):
        if "知识产权法" in source:
            adjustment -= 0.55
    if any(keyword in compact_query for keyword in ["加班", "工作时间", "工时", "大小周"]):
        if "社会保险法" in source and not any(marker in compact_query for marker in ["社会保险", "社保", "工伤", "工伤保险"]):
            adjustment += 0.65
        if "劳动法" in source and any(marker in compact_law for marker in ["每日工作时间", "延长工作时间", "支付高于劳动者正常工作时间工资"]):
            adjustment -= 0.35
    return adjustment

def _extract_article_from_law_text(text: str, article: str) -> str:
    pattern = rf"(?m)^{re.escape(article)}[　 \t][\s\S]*?(?=^{ARTICLE_PATTERN}[　 \t]|\Z)"
    match = re.search(pattern, text)
    if not match:
        return ""
    return match.group(0).strip()

def _canonicalize_retrieved_doc(doc: LangchainDocument, db_dir: str) -> List[Tuple[str, str]]:
    source = doc.metadata.get("source", "未知法条来源")
    articles = list(dict.fromkeys(re.findall(rf"(?m)^({ARTICLE_PATTERN})[　 \t]", doc.page_content)))
    if not articles:
        return [(source, doc.page_content)]
    laws_dir = os.path.join(os.path.dirname(db_dir), "laws")
    file_path = os.path.join(laws_dir, source)
    if not os.path.isfile(file_path):
        return [(source, doc.page_content)]
    law_text = extract_contract_text(file_path)
    canonical_articles = []
    for article in articles:
        article_text = _extract_article_from_law_text(law_text, article)
        if article_text:
            canonical_articles.append((source, article_text))
    return canonical_articles or [(source, doc.page_content)]

def _split_law_text_by_article(text: str) -> List[str]:
    matches = list(re.finditer(rf"(?m)^{ARTICLE_PATTERN}[　 \t]", text))
    if not matches:
        return []
    articles = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        article_text = text[start:end].strip()
        if _is_substantive_law_chunk(article_text):
            articles.append(article_text)
    return articles

def _load_law_article_documents(db_dir: str) -> List[LangchainDocument]:
    laws_dir = os.path.join(os.path.dirname(db_dir), "laws")
    law_documents = []
    if not os.path.isdir(laws_dir):
        return law_documents
    for file_name in os.listdir(laws_dir):
        file_path = os.path.join(laws_dir, file_name)
        if not os.path.isfile(file_path) or file_name.startswith("."):
            continue
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext not in [".docx", ".pdf", ".txt"]:
            continue
        text_content = extract_contract_text(file_path)
        article_texts = _split_law_text_by_article(text_content)
        for article_text in article_texts:
            law_documents.append(
                LangchainDocument(
                    page_content=article_text,
                    metadata={"source": file_name}
                )
            )
    return law_documents

def _is_relevant_law_chunk(query_text: str, doc: LangchainDocument) -> bool:
    source = doc.metadata.get("source", "")
    compact_query = re.sub(r"\s+", "", query_text)
    compact_text = re.sub(r"\s+", "", doc.page_content)
    dispute_keywords = ["仲裁", "调解", "诉讼", "受理", "劳动争议"]
    law_name_markers = [
        "《",
        "劳动合同法", "劳动法", "社会保险法", "工伤保险条例", "女职工劳动保护特别规定",
        "职工带薪年休假条例", "专利法", "著作权法", "计算机软件保护条例",
        "劳动争议调解仲裁法", "最高人民法院关于审理劳动争议案件适用法律问题的解释"
    ]
    target_articles = re.findall(ARTICLE_PATTERN, compact_query) if any(marker in compact_query for marker in law_name_markers) else []
    if "劳动合同法" in compact_query and "劳动合同法" not in source:
        return False
    if "劳动法" in compact_query and "劳动合同法" not in compact_query and "劳动法" not in source:
        return False
    social_source_markers = ["社会保险", "社保", "工伤", "工伤保险", "生育保险", "养老保险", "医疗保险", "失业保险", "保险费", "社保登记"]
    if "社会保险法" in source and not any(marker in compact_query for marker in social_source_markers):
        return False
    source_filters = [
        ("社会保险法", "社会保险法"),
        ("工伤保险条例", "工伤保险条例"),
        ("女职工劳动保护特别规定", "女职工劳动保护特别规定"),
        ("职工带薪年休假条例", "职工带薪年休假条例"),
        ("专利法", "知识产权法"),
        ("著作权法", "知识产权法"),
        ("计算机软件保护条例", "知识产权法"),
    ]
    for query_law_name, source_law_name in source_filters:
        if query_law_name in compact_query and source_law_name not in source:
            return False
    if target_articles and not any(article in compact_text for article in target_articles):
        return False
    if "劳动争议调解仲裁法" in source and not any(keyword in compact_query for keyword in dispute_keywords):
        return False
    if "最高人民法院关于审理劳动争议案件适用法律问题的解释" in source or "最高法劳动争议司法解释" in source:
        if "为正确审理劳动争议案件" in compact_text:
            return False
        if "竞业" in compact_query:
            return "竞业限制" in compact_text
        if not any(keyword in compact_query for keyword in dispute_keywords):
            return False
    return True

def build_law_vector_db(laws_dir: str, save_dir: str) -> None:
    """
    扫描并解析 laws_dir 目录下的所有 Word/PDF 法律文本，构建本地 FAISS 向量数据库
    :param laws_dir: 存放法律法规原始文件的目录 (data/laws/)
    :param save_dir: FAISS 向量库的本地持久化输出目录 (data/faiss_index/)
    """
    raw_documents: List[LangchainDocument] = []
    
    if not os.path.exists(laws_dir):
        raise FileNotFoundError(f"法律法规源目录不存在: {laws_dir}")
        
    print(f"[RAG] 开始扫描法条目录: {laws_dir}...")
    
    # 1. 遍历 laws_dir 目录下的所有文件并解析
    for file_name in os.listdir(laws_dir):
        file_path = os.path.join(laws_dir, file_name)
        # 仅处理文件，跳过目录或隐藏文件
        if not os.path.isfile(file_path) or file_name.startswith("."):
            continue
            
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext in [".docx", ".pdf", ".txt"]:
            print(f"[RAG] 正在解析法条文件: {file_name}...")
            try:
                # 调用 parser.py 中的高鲁棒分流提取逻辑
                text_content = extract_contract_text(file_path)
                
                # 如果是空白文件则跳过
                if not text_content.strip():
                    continue
                    
                article_texts = _split_law_text_by_article(text_content)
                if article_texts:
                    for article_text in article_texts:
                        raw_documents.append(
                            LangchainDocument(
                                page_content=article_text,
                                metadata={"source": file_name}
                            )
                        )
                else:
                    raw_documents.append(
                        LangchainDocument(
                            page_content=text_content,
                            metadata={"source": file_name}
                        )
                    )
            except Exception as e:
                print(f"[RAG 警告] 解析文件 {file_name} 时出错: {e}，已跳过。")
                
    if not raw_documents:
        raise ValueError(f"未能在目录 {laws_dir} 中找到或成功解析任何有效的法律条文文件。")
        
    # 2. 文本语义切分
    # 优先按“第X条”切成完整法条。只有超长法条才继续用递归切分器兜底。
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", "。", "；", " ", ""]
    )
    split_docs = []
    for doc in raw_documents:
        if len(doc.page_content) <= 900:
            split_docs.append(doc)
        else:
            split_docs.extend(text_splitter.split_documents([doc]))
    print(f"[RAG] 法规文本按法条粒度切分完毕，共生成 {len(split_docs)} 个语义文本块。")
    
    # 3. 初始化 OpenAI 兼容协议的 Embedding 客户端
    embedding_model_name = _get_embedding_model_name()
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("API_KEY"),
        openai_api_base=os.getenv("BASE_URL"),
        model=embedding_model_name,
        chunk_size=32
    )
    
    # 4. 构建 FAISS 本地索引
    print(f"[RAG] 正在调用 {embedding_model_name} 接口计算法条向量，请稍候...")
    db = FAISS.from_documents(split_docs, embeddings)
    
    # 5. 持久化索引到本地磁盘，以供系统主程序以毫秒级时延高速检索
    os.makedirs(save_dir, exist_ok=True)
    db.save_local(save_dir)
    print(f"[RAG] 向量数据库构建并持久化成功！已保存至: {save_dir}")

def query_laws(query: str, db_dir: str, top_k: int = 5) -> str:
    """
    加载本地 FAISS 索引并检索与 Query 最相关的 Top-K 法条文本。
    采用多路段落检索 (Multi-query Retrieval) 策略以解决长文本查询向量稀释问题。
    :param query: 合同待审条款文本
    :param db_dir: 本地 FAISS 向量库索引所在目录 (data/faiss_index/)
    :param top_k: 召回的相似条文最大数量，默认为 5
    :return: 拼接后的格式化参考法条文本
    """
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("API_KEY"),
        openai_api_base=os.getenv("BASE_URL"),
        model=_get_embedding_model_name(),
        chunk_size=32
    )
    
    # 校验本地索引是否存在
    if not os.path.exists(os.path.join(db_dir, "index.faiss")):
        raise FileNotFoundError(f"本地向量数据库未构建，未在 {db_dir} 下找到 index.faiss 文件。")
        
    # 安全加载本地向量库
    db = FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)
    
    # 按合同条款切分，将具体合同条款作为 RAG 查询输入，符合 SRS 中“条款向量化检索 Top-K 法条”的设计
    search_queries = _build_search_queries(query)
        
    # 批量计算段落向量，防止循环中频繁调用 API 导致高延迟
    query_vectors = embeddings.embed_documents(search_queries)
    
    unique_docs = {}
    query_hits = []
    law_article_documents = _load_law_article_documents(db_dir)
    
    # 对每一个实质性段落进行独立检索（每段召回 Top 2 法条）
    for vec in query_vectors:
        current_hits = []
        current_query = search_queries[len(query_hits)]
        results = db.similarity_search_with_score_by_vector(vec, k=max(80, top_k * 10))
        for rank, (doc, score) in enumerate(results):
            if not _is_substantive_law_chunk(doc.page_content):
                continue
            if not _is_relevant_law_chunk(current_query, doc):
                continue
            rerank_score = score - (_law_chunk_keyword_score(current_query, doc.page_content) * 0.18) + _law_chunk_penalty(current_query, doc.page_content) + _law_source_adjustment(current_query, doc)
            # 使用内容本身作为键进行去重
            key = re.sub(r"\s+", "", doc.page_content)
            if key not in unique_docs or rerank_score < unique_docs[key][0]:
                unique_docs[key] = (rerank_score, doc)
            current_hits.append((rank, rerank_score, key))
        lexical_hits = [
            doc for doc in law_article_documents
            if _law_chunk_keyword_score(current_query, doc.page_content) >= 3
            and _is_relevant_law_chunk(current_query, doc)
        ]
        for lexical_rank, doc in enumerate(sorted(lexical_hits, key=lambda item: -_law_chunk_keyword_score(current_query, item.page_content))[:8]):
            lexical_score = 0.35 - (_law_chunk_keyword_score(current_query, doc.page_content) * 0.12) + _law_chunk_penalty(current_query, doc.page_content) + _law_source_adjustment(current_query, doc)
            key = re.sub(r"\s+", "", doc.page_content)
            if key not in unique_docs or lexical_score < unique_docs[key][0]:
                unique_docs[key] = (lexical_score, doc)
            current_hits.append((lexical_rank, lexical_score, key))
        query_hits.append(sorted(current_hits, key=lambda item: (item[1], item[0])))
                
    # 提取前 top_k 个独立去重法条
    selected_keys = []
    max_hits = max((len(hits) for hits in query_hits), default=0)
    for hit_index in range(max_hits):
        for hits in query_hits:
            if hit_index >= len(hits):
                continue
            key = hits[hit_index][2]
            if key not in selected_keys:
                selected_keys.append(key)
            if len(selected_keys) >= top_k:
                break
        if len(selected_keys) >= top_k:
            break
    if len(selected_keys) < top_k:
        for key, _ in sorted(unique_docs.items(), key=lambda item: item[1][0]):
            if key not in selected_keys:
                selected_keys.append(key)
            if len(selected_keys) >= top_k:
                break
    final_docs = [unique_docs[key][1] for key in selected_keys[:top_k]]

    canonical_refs = []
    seen_refs = set()
    for doc in final_docs:
        for source, article_text in _canonicalize_retrieved_doc(doc, db_dir):
            ref_key = (source, re.sub(r"\s+", "", article_text))
            if ref_key in seen_refs:
                continue
            seen_refs.add(ref_key)
            canonical_refs.append((source, article_text))
            if len(canonical_refs) >= top_k:
                break
        if len(canonical_refs) >= top_k:
            break

    if not canonical_refs:
        canonical_refs = [
            (doc.metadata.get("source", "未知法条来源"), doc.page_content)
            for doc in final_docs
        ]

    # 拼接法条及来源元数据
    formatted_references = []
    for i, (source, article_text) in enumerate(canonical_refs[:top_k], 1):
        ref_text = f"【参考依据 {i}】(来源: {source})\n{article_text}"
        formatted_references.append(ref_text)
        
    return "\n\n".join(formatted_references)

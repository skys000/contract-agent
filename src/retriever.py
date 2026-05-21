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
    """
    读取向量模型名称，确保 RAG 构建和查询阶段使用同一套 embedding 配置。
    """
    # 从环境变量读取 embedding 模型，避免在源码中写死供应商或模型名
    model_name = os.getenv("EMBEDDING_MODEL_NAME")
    if not model_name:
        # embedding 模型是构建和查询 FAISS 的必要条件，缺失时直接给出明确配置错误
        raise ValueError("未配置向量模型名称，请在 .env 中设置 EMBEDDING_MODEL_NAME。")
    return model_name

def _has_risk_context(text: str, keywords: List[str], safe_markers: List[str]) -> bool:
    """
    判断合同中是否存在某类风险关键词，同时排除明确合规的安全表述。
    """
    # 去除空白后做关键词判断，兼容 PDF/Word 解析后出现的换行和空格差异
    compact_text = re.sub(r"\s+", "", text)
    # 如果没有命中风险关键词，则该主题不需要额外补充检索 query
    if not any(keyword in compact_text for keyword in keywords):
        return False
    # 命中安全表述时不触发风险主题扩展，避免把合规描述误当成违法风险
    return not any(marker in compact_text for marker in safe_markers)

def _build_search_queries(contract_text: str) -> List[str]:
    """
    将整份合同扩展为多条 RAG 检索 query。

    该函数先按合同条款拆分，再针对试用期、社保、工时、工资、竞业限制、
    知识产权、女职工保护等劳动合同高频风险点补充主题 query，避免长合同整体向量稀释。
    """
    # 统一换行格式，减少不同解析器输出格式对条款切分的影响
    normalized_text = re.sub(r"\n\s*", "\n", contract_text)
    # 优先按“第X条”切分合同条款，使 RAG 查询更接近具体风险事实
    clauses = [
        clause.strip()
        for clause in re.split(r"(?=\n?第[一二三四五六七八九十百零〇\d]+条)", normalized_text)
        if len(clause.strip()) > 20
    ]
    # 如果合同没有明显条号，则退化按自然行切分
    if not clauses:
        clauses = [p.strip() for p in contract_text.split('\n') if len(p.strip()) > 20]
    # 这些关键词代表劳动合同高频审查点，命中后优先参与检索
    priority_keywords = [
        "试用期", "社会保险", "社保", "加班", "工作时间", "休息休假", "违约金", "服务期",
        "培训", "竞业", "保密", "解除", "终止", "工资", "薪资", "劳动报酬", "工作地点",
        "工作内容", "劳动保护", "劳动条件", "知识产权", "专利", "著作权", "软件著作权",
        "职务成果", "职务发明", "职务作品", "源代码", "代码", "算法", "开源", "个人作品"
    ]
    # 将高风险条款排在前面，提升有限 query 数量下的召回质量
    priority_clauses = [clause for clause in clauses if any(keyword in clause for keyword in priority_keywords)]
    fallback_clauses = [clause for clause in clauses if clause not in priority_clauses]
    topic_queries = []
    # compact_text 用于全局场景识别，避免被换行空格干扰
    compact_text = re.sub(r"\s+", "", contract_text)
    if "试用期" in compact_text:
        # 试用期通常需要同时召回期限上限和试用期工资比例
        topic_queries.extend([
            "试用期 不得超过 一个月 二个月 六个月 劳动合同期限",
            "试用期 工资 不得低于 百分之八十 最低工资标准"
        ])
    if any(keyword in compact_text for keyword in ["社会保险", "社保", "五险"]):
        # 社保风险需要同时覆盖劳动合同必备条款和社会保险登记义务
        topic_queries.extend([
            "劳动合同 必备条款 社会保险",
            "用人单位 劳动者 必须依法参加社会保险 缴纳社会保险费",
            "《中华人民共和国社会保险法》第五十八条 自用工之日起三十日 社会保险登记"
        ])
    if any(keyword in compact_text for keyword in ["工作时间", "休息休假", "加班", "工时"]):
        # 工时/加班场景优先召回标准工时、休息日和加班工资依据
        topic_queries.extend([
            "每日工作时间不超过八小时 平均每周工作时间 工时制度",
            "用人单位应当保证劳动者每周至少休息一日",
            "延长工作时间 加班工资 支付工资报酬"
        ])
    if any(keyword in compact_text for keyword in ["年休假", "带薪年休假", "未休年休假"]):
        # 年休假属于专项法规，单独补 query 可降低被通用劳动法条淹没的概率
        topic_queries.extend([
            "职工连续工作一年以上 享受带薪年休假",
            "单位根据生产工作的具体情况 职工本人意愿 统筹安排年休假",
            "不能安排年休假 应当按照日工资收入支付年休假工资报酬"
        ])
    if any(keyword in compact_text for keyword in ["劳动报酬", "工资", "薪资", "底薪"]):
        # 工资支付是劳动合同核心义务，补充按月货币支付主题 query
        topic_queries.append("工资 货币形式 按月支付 不得克扣 无故拖欠")
    if _has_risk_context(contract_text, ["违约金", "服务期", "培训费"], ["不约定由乙方承担违法违约金", "依法发生专项培训服务期"]):
        # 服务期/违约金需要区分专项培训合法违约金与普通离职违法违约金
        topic_queries.extend([
            "专项培训费用 专业技术培训 服务期 违约金",
            "不得与劳动者约定由劳动者承担违约金"
        ])
    if any(keyword in compact_text for keyword in ["商业秘密", "保密"]):
        # 保密条款通常与商业秘密、知识产权或竞业限制相邻
        topic_queries.append("商业秘密 知识产权 保密事项")
    if any(keyword in compact_text for keyword in ["竞业", "竞业限制"]):
        # 竞业限制重点召回人员范围、期限和经济补偿规则
        topic_queries.extend([
            "竞业限制 商业秘密 知识产权 保密事项 按月给予经济补偿",
            "竞业限制 人员 范围 地域 期限 不得超过二年",
            "竞业限制 未约定经济补偿 十二个月平均工资 30% 最低工资标准"
        ])
    if any(keyword in compact_text for keyword in ["知识产权", "专利", "著作权", "软件著作权", "职务成果", "职务发明", "职务作品", "源代码", "代码", "算法", "开源", "个人作品"]):
        # 研发/科技岗位的知识产权风险需要召回职务发明、职务作品、软件著作权边界
        topic_queries.extend([
            "《中华人民共和国专利法》第六条 职务发明创造 主要利用本单位物质技术条件 专利申请权",
            "《中华人民共和国著作权法》第十八条 职务作品 计算机软件 主要利用法人组织物质技术条件",
            "《计算机软件保护条例》第十三条 软件著作权 本职工作 开发目标 物质技术条件"
        ])
    if any(keyword in compact_text for keyword in ["工伤", "职业伤害", "意外事故", "工伤保险"]):
        # 工伤场景补充认定、缴费和待遇三类依据
        topic_queries.extend([
            "工伤保险 用人单位 缴纳工伤保险费 职工不缴纳",
            "职工有下列情形之一的应当认定为工伤 工作时间工作场所",
            "工伤保险待遇 治疗工伤的医疗费用 停工留薪期"
        ])
    if any(keyword in compact_text for keyword in ["职业病", "职业危害", "职业健康", "职业禁忌", "防护用品", "劳动防护"]):
        # 职业病/劳动保护场景需要召回防护用品和职业健康检查费用规则
        topic_queries.extend([
            "职业病危害 用人单位 职业健康检查 费用由用人单位承担",
            "职业病危害 防护设施 个人使用的职业病防护用品 劳动者保护权利",
            "不得安排孕期哺乳期女职工从事接触职业病危害作业"
        ])
    if any(keyword in compact_text for keyword in ["女职工", "女员工", "怀孕", "产期", "哺乳期", "产假", "三期", "孕期"]):
        # 女职工“三期”保护属于专项高风险场景，单独补充 query
        topic_queries.extend([
            "女职工在孕期产期哺乳期 用人单位不得解除劳动合同",
            "女职工劳动保护 产假 生育享受",
            "不得安排女职工 怀孕 哺乳 劳动强度 夜班"
        ])
    if any(keyword in compact_text for keyword in ["工作内容", "工作地点"]):
        # 工作内容和地点属于劳动合同必备条款
        topic_queries.append("劳动合同应当具备 工作内容 工作地点")
    if any(keyword in compact_text for keyword in ["劳动保护", "劳动条件"]):
        # 劳动保护/劳动条件不足时需要召回职业危害防护相关依据
        topic_queries.append("劳动保护 劳动条件 职业危害防护")
    if _has_risk_context(contract_text, ["解除", "终止"], ["依照中华人民共和国劳动合同法", "依照《中华人民共和国劳动合同法》", "不得违法解除", "依法应支付经济补偿"]):
        # 解除终止条款只在存在潜在风险上下文时补 query，避免合规引用触发无关召回
        topic_queries.append("解除劳动合同 终止劳动合同 经济补偿")
    if any(keyword in compact_text for keyword in ["劳动争议", "仲裁", "调解", "诉讼", "争议处理"]):
        # 劳动争议程序性条款优先放在最前面，保证咨询/争议场景召回稳定
        dispute_topic_queries = [
            "劳动争议 因确认劳动关系 劳动合同 劳动报酬 工伤医疗费 经济补偿",
            "发生劳动争议 当事人可以申请调解 仲裁 提起诉讼",
            "劳动争议申请仲裁的时效期间为一年"
        ]
        topic_queries = dispute_topic_queries + topic_queries
    # 主题 query 优先，其次高风险条款，最后普通条款；用 dict 去重并限制数量控制 API 成本
    queries = topic_queries + priority_clauses + fallback_clauses
    return list(dict.fromkeys(queries[:24] or [contract_text]))

def _is_substantive_law_chunk(text: str) -> bool:
    """
    过滤目录、总则标题等非实质条文，减少 RAG 引用噪声。
    """
    # 去掉空白后判断，避免 PDF/Word 解析差异影响条号识别
    compact_text = re.sub(r"\s+", "", text)
    # 没有“第X条”结构的片段通常不是可引用的正式法条
    if not re.search(ARTICLE_PATTERN, compact_text):
        return False
    # 目录页中的条号只是索引，不应作为实体法依据召回
    if "目录" in compact_text and len(re.findall(ARTICLE_PATTERN, compact_text)) <= 2:
        return False
    # 过短的“第一章 总则”标题类内容缺少具体裁判规则，应过滤
    if "第一章总则" in compact_text and len(compact_text) < 260:
        return False
    return True

def _law_chunk_keyword_score(query_text: str, law_text: str) -> int:
    """
    根据劳动合同场景关键词计算 query 与候选法条的词面相关性加分。
    """
    # 同时压缩 query 与法条文本，确保关键词匹配不受空格、换行影响
    compact_query = re.sub(r"\s+", "", query_text)
    compact_law = re.sub(r"\s+", "", law_text)
    # 每组关键词对应一个劳动合同审查主题，query 和法条同时命中即加权
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
        # query 命中主题关键词，说明当前检索确实关注该劳动法场景
        query_hit = any(keyword in compact_query for keyword in group)
        # 法条命中同一组关键词，说明候选依据与当前场景方向一致
        law_hit = any(keyword in compact_law for keyword in group)
        if query_hit and law_hit:
            score += 3
    # 额外计算逐词重合度，作为精细加分，但限制上限防止长文本刷分
    all_keywords = [keyword for group in keyword_groups for keyword in group]
    exact_overlap = [keyword for keyword in all_keywords if keyword in compact_query and keyword in compact_law]
    return score + min(len(exact_overlap), 5)

def _law_chunk_penalty(query_text: str, law_text: str) -> float:
    """
    对容易误召回的泛化条款或无关法规片段施加惩罚分。
    """
    # 压缩文本后统一匹配惩罚规则
    compact_query = re.sub(r"\s+", "", query_text)
    compact_law = re.sub(r"\s+", "", law_text)
    penalty = 0.0
    # 泛化条款通常不能直接支撑具体劳动合同风险结论，因此施加基础惩罚
    generic_patterns = [
        "适用本法", "国家提倡", "社会保险水平应当与社会经济发展水平",
        "劳动合同分为固定期限", "集体合同签订后", "非全日制用工",
        "劳务派遣单位", "依法建立和完善规章制度",
        "国家工作人员在社会保险管理、监督工作中滥用职权",
        "国务院人事部门、国务院劳动保障部门依据职权"
    ]
    if any(pattern in compact_law for pattern in generic_patterns):
        penalty += 0.45
    # catch-all 条款容易被很多 query 命中，但往往只说明必备事项，需进一步降权
    catch_all_patterns = [
        "劳动合同应当具备以下条款",
        "用人单位的名称、住所和法定代表人或者主要负责人",
        "社会保险制度坚持广覆盖、保基本、多层次、可持续的方针",
        "中华人民共和国境内的用人单位和个人依法缴纳社会保险费"
    ]
    if any(pattern in compact_law for pattern in catch_all_patterns):
        penalty += 0.6
    # 女职工、未成年、工会等专项条款只有在 query 明确相关时才应优先出现
    female_worker_markers = ["女职工", "女员工", "怀孕", "产期", "哺乳期", "孕期", "产假", "三期"]
    if "女职工" in compact_law and not any(marker in compact_query for marker in female_worker_markers):
        penalty += 0.55
    if "未成年" in compact_law and "未成年" not in compact_query:
        penalty += 0.55
    if "工会" in compact_law and "工会" not in compact_query:
        penalty += 0.35
    # 社保行政处罚类条文只在明确“不缴/未缴”场景下优先召回
    social_violation_markers = ["不缴", "未缴", "无故不缴", "未依法缴纳", "社保豁免"]
    if any(pattern in compact_law for pattern in ["无故不缴纳社会保险费", "未依法为劳动者缴纳社会保险费"]):
        if not any(marker in compact_query for marker in social_violation_markers):
            penalty += 0.65
    # 解除/经济补偿条文在非解除争议场景下容易误入，适度降权
    termination_violation_markers = ["违法解除", "单方面解除", "随时解除", "不支付经济补偿", "辞退", "开除"]
    if any(pattern in compact_law for pattern in ["可以解除劳动合同", "给予经济补偿", "支付经济补偿"]):
        if not any(marker in compact_query for marker in termination_violation_markers):
            penalty += 0.35
    return penalty

def _law_source_adjustment(query_text: str, doc: LangchainDocument) -> float:
    """
    按 query 主题和法规来源做轻量级来源重排，提升专项法规在对应场景下的优先级。
    """
    # 根据 query、法条正文和来源文件名综合判断是否需要来源级别加减权
    compact_query = re.sub(r"\s+", "", query_text)
    compact_law = re.sub(r"\s+", "", doc.page_content)
    source = doc.metadata.get("source", "")
    adjustment = 0.0
    # 返回负数代表更优先，正数代表降权；后续会加到 rerank_score 中
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
        # 加班/工时问题不应优先引用社保法
        if "社会保险法" in source and not any(marker in compact_query for marker in ["社会保险", "社保", "工伤", "工伤保险"]):
            adjustment += 0.65
        # 劳动法中的工时与加班工资条款在该场景下应适度优先
        if "劳动法" in source and any(marker in compact_law for marker in ["每日工作时间", "延长工作时间", "支付高于劳动者正常工作时间工资"]):
            adjustment -= 0.35
    return adjustment

def _extract_article_from_law_text(text: str, article: str) -> str:
    """
    从完整法规文本中提取指定“第X条”的全文，用于将向量命中片段回填为完整法条。
    """
    # 使用条号作为起点，截取到下一条或文件末尾，尽量保持完整法条文本
    pattern = rf"(?m)^{re.escape(article)}[　 \t][\s\S]*?(?=^{ARTICLE_PATTERN}[　 \t]|\Z)"
    match = re.search(pattern, text)
    if not match:
        # 如果本地原文格式不匹配，则交给调用方保留向量命中的原片段
        return ""
    return match.group(0).strip()

def _canonicalize_retrieved_doc(doc: LangchainDocument, db_dir: str) -> List[Tuple[str, str]]:
    """
    将向量库召回的候选片段规范化为本地法规文件中的完整法条文本。
    """
    # 取出向量库元数据中的来源文件名，用于回到 data/laws 查找原始法规文件
    source = doc.metadata.get("source", "未知法条来源")
    # 从召回片段中识别所有法条编号，并用 dict 保持去重后的原顺序
    articles = list(dict.fromkeys(re.findall(rf"(?m)^({ARTICLE_PATTERN})[　 \t]", doc.page_content)))
    if not articles:
        # 无法识别条号时，直接返回召回片段本身
        return [(source, doc.page_content)]
    # 根据 FAISS 目录反推 laws 目录位置
    laws_dir = os.path.join(os.path.dirname(db_dir), "laws")
    file_path = os.path.join(laws_dir, source)
    if not os.path.isfile(file_path):
        # 找不到原始法规文件时保留向量库内容，保证系统仍可返回依据
        return [(source, doc.page_content)]
    # 重新解析原始法规文件，以便从全文中截取完整法条
    law_text = extract_contract_text(file_path)
    canonical_articles = []
    for article in articles:
        article_text = _extract_article_from_law_text(law_text, article)
        if article_text:
            canonical_articles.append((source, article_text))
    # 如果完整法条回填失败，则回退到向量召回片段
    return canonical_articles or [(source, doc.page_content)]

def _split_law_text_by_article(text: str) -> List[str]:
    """
    按“第X条”切分法规文本，优先保持完整法条粒度，便于后续精确引用。
    """
    # 定位所有以“第X条”开头的法条起点
    matches = list(re.finditer(rf"(?m)^{ARTICLE_PATTERN}[　 \t]", text))
    if not matches:
        return []
    articles = []
    for index, match in enumerate(matches):
        # 当前条从本条起点截取到下一条起点；最后一条截取到全文末尾
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        article_text = text[start:end].strip()
        # 过滤目录、总则标题等非实体条文
        if _is_substantive_law_chunk(article_text):
            articles.append(article_text)
    return articles

def _load_law_article_documents(db_dir: str) -> List[LangchainDocument]:
    """
    从本地 laws 目录加载完整法条文档，作为向量召回之外的词面补充召回池。
    """
    # 根据 FAISS 索引目录定位同级 laws 目录
    laws_dir = os.path.join(os.path.dirname(db_dir), "laws")
    law_documents = []
    if not os.path.isdir(laws_dir):
        return law_documents
    for file_name in os.listdir(laws_dir):
        file_path = os.path.join(laws_dir, file_name)
        # 跳过目录、隐藏文件和非文件项
        if not os.path.isfile(file_path) or file_name.startswith("."):
            continue
        file_ext = os.path.splitext(file_name)[1].lower()
        # 仅加载当前解析器支持的法规格式
        if file_ext not in [".docx", ".pdf", ".txt"]:
            continue
        # 法规文件使用 legacy 解析，避免 MinerU 增强解析影响法库构建速度
        text_content = extract_contract_text(file_path)
        article_texts = _split_law_text_by_article(text_content)
        for article_text in article_texts:
            # 每条法条作为一个 LangChain Document，metadata 保存来源文件名
            law_documents.append(
                LangchainDocument(
                    page_content=article_text,
                    metadata={"source": file_name}
                )
            )
    return law_documents

def _is_relevant_law_chunk(query_text: str, doc: LangchainDocument) -> bool:
    """
    判断候选法条是否与当前 query 场景相关。

    该函数通过法名、条号、来源文件和场景关键词过滤明显无关条文，
    避免把社会保险法总则、女职工条款、劳动争议条款等误召回到无关合同风险中。
    """
    # 统一压缩 query 和候选法条，便于关键词与条号匹配
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
    # 只有 query 明确带法名/书名号时，才用条号做强过滤，避免普通合同条号误判为法律条号
    target_articles = re.findall(ARTICLE_PATTERN, compact_query) if any(marker in compact_query for marker in law_name_markers) else []
    # 明确指定劳动合同法时，过滤其他来源
    if "劳动合同法" in compact_query and "劳动合同法" not in source:
        return False
    # 明确指定劳动法且不是劳动合同法时，过滤其他来源
    if "劳动法" in compact_query and "劳动合同法" not in compact_query and "劳动法" not in source:
        return False
    social_source_markers = ["社会保险", "社保", "工伤", "工伤保险", "生育保险", "养老保险", "医疗保险", "失业保险", "保险费", "社保登记"]
    # 社会保险法只有在 query 关注社保/工伤等主题时才参与
    if "社会保险法" in source and not any(marker in compact_query for marker in social_source_markers):
        return False
    # 专项法规的显式法名查询必须对应到正确来源文件，避免跨法误召回
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
    # query 明确指定条号时，候选法条必须包含对应条号
    if target_articles and not any(article in compact_text for article in target_articles):
        return False
    # 劳动争议法条只在仲裁/调解/诉讼等争议场景下参与
    if "劳动争议调解仲裁法" in source and not any(keyword in compact_query for keyword in dispute_keywords):
        return False
    if "最高人民法院关于审理劳动争议案件适用法律问题的解释" in source or "最高法劳动争议司法解释" in source:
        # 司法解释开篇目的性条款不是具体裁判规则，直接过滤
        if "为正确审理劳动争议案件" in compact_text:
            return False
        # 竞业限制查询可使用司法解释中的竞业补偿条款
        if "竞业" in compact_query:
            return "竞业限制" in compact_text
        # 其他场景没有劳动争议关键词时不召回司法解释
        if not any(keyword in compact_query for keyword in dispute_keywords):
            return False
    return True

def build_law_vector_db(laws_dir: str, save_dir: str) -> None:
    """
    扫描并解析 laws_dir 目录下的所有 Word/PDF 法律文本，构建本地 FAISS 向量数据库
    :param laws_dir: 存放法律法规原始文件的目录 (data/laws/)
    :param save_dir: FAISS 向量库的本地持久化输出目录 (data/faiss_index/)

    构建阶段尽量按完整法条切块，只有超长条文才继续递归拆分，确保最终报告引用时更接近真实法条原文。
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
                    
                # 优先按完整法条切分，保证后续审查报告引用的依据可读、可核验
                article_texts = _split_law_text_by_article(text_content)
                if article_texts:
                    for article_text in article_texts:
                        # 每条法条独立入库，并保留来源文件名
                        raw_documents.append(
                            LangchainDocument(
                                page_content=article_text,
                                metadata={"source": file_name}
                            )
                        )
                else:
                    # 无法按条号切分时，将整篇文本作为一个文档兜底
                    raw_documents.append(
                        LangchainDocument(
                            page_content=text_content,
                            metadata={"source": file_name}
                        )
                    )
            except Exception as e:
                print(f"[RAG 警告] 解析文件 {file_name} 时出错: {e}，已跳过。")
                
    if not raw_documents:
        # 未解析到任何法规时直接中断，避免构建空 FAISS 索引
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
        # 普通法条长度合适时保持原样，不再切碎
        if len(doc.page_content) <= 900:
            split_docs.append(doc)
        else:
            # 超长条文才递归切分，兼顾向量检索长度限制和法条完整性
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

    查询阶段综合向量相似度、关键词加分、无关条款惩罚和法规来源调整，最后再回填完整法条。
    """
    # 初始化与构建阶段一致的 embedding 客户端，保证查询向量空间一致
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
    
    # unique_docs 用压缩后的法条内容去重，query_hits 记录每条 query 的候选排序
    unique_docs = {}
    query_hits = []
    # 加载完整法条文本池，用于弥补向量召回对精确条号/关键词的遗漏
    law_article_documents = _load_law_article_documents(db_dir)
    
    # 对每一个实质性段落进行独立检索（每段召回 Top 2 法条）
    for vec in query_vectors:
        current_hits = []
        current_query = search_queries[len(query_hits)]
        # 初始召回放宽到较大 k，再通过本地规则过滤和重排
        results = db.similarity_search_with_score_by_vector(vec, k=max(80, top_k * 10))
        for rank, (doc, score) in enumerate(results):
            # 先过滤非实质法条，如目录、标题、总则概述
            if not _is_substantive_law_chunk(doc.page_content):
                continue
            # 再按当前 query 的主题、来源和条号做相关性过滤
            if not _is_relevant_law_chunk(current_query, doc):
                continue
            # 综合向量距离、关键词加分、噪声惩罚和来源调整得到最终排序分
            rerank_score = score - (_law_chunk_keyword_score(current_query, doc.page_content) * 0.18) + _law_chunk_penalty(current_query, doc.page_content) + _law_source_adjustment(current_query, doc)
            # 使用内容本身作为键进行去重
            key = re.sub(r"\s+", "", doc.page_content)
            if key not in unique_docs or rerank_score < unique_docs[key][0]:
                unique_docs[key] = (rerank_score, doc)
            current_hits.append((rank, rerank_score, key))
        # 词面补充召回：对明确关键词/条号场景，直接从完整法条池补候选
        lexical_hits = [
            doc for doc in law_article_documents
            if _law_chunk_keyword_score(current_query, doc.page_content) >= 3
            and _is_relevant_law_chunk(current_query, doc)
        ]
        for lexical_rank, doc in enumerate(sorted(lexical_hits, key=lambda item: -_law_chunk_keyword_score(current_query, item.page_content))[:8]):
            # 词面命中的基础分较低，表示其在精确主题上可与向量候选竞争
            lexical_score = 0.35 - (_law_chunk_keyword_score(current_query, doc.page_content) * 0.12) + _law_chunk_penalty(current_query, doc.page_content) + _law_source_adjustment(current_query, doc)
            key = re.sub(r"\s+", "", doc.page_content)
            if key not in unique_docs or lexical_score < unique_docs[key][0]:
                unique_docs[key] = (lexical_score, doc)
            current_hits.append((lexical_rank, lexical_score, key))
        # 每个 query 的候选按重排分排序，供后续轮询式合并
        query_hits.append(sorted(current_hits, key=lambda item: (item[1], item[0])))
                
    # 提取前 top_k 个独立去重法条
    selected_keys = []
    max_hits = max((len(hits) for hits in query_hits), default=0)
    # 轮询各 query 的第 1、第 2、第 3 名，避免单个长条款垄断所有召回位
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
    # 如果轮询后数量不足，则按全局最佳分数补齐
    if len(selected_keys) < top_k:
        for key, _ in sorted(unique_docs.items(), key=lambda item: item[1][0]):
            if key not in selected_keys:
                selected_keys.append(key)
            if len(selected_keys) >= top_k:
                break
    # 根据选中的去重键取回最终候选 Document
    final_docs = [unique_docs[key][1] for key in selected_keys[:top_k]]

    canonical_refs = []
    seen_refs = set()
    for doc in final_docs:
        # 将向量片段回填为本地法规中的完整法条
        for source, article_text in _canonicalize_retrieved_doc(doc, db_dir):
            ref_key = (source, re.sub(r"\s+", "", article_text))
            if ref_key in seen_refs:
                continue
            # 同一来源同一法条只输出一次，避免报告参考依据重复
            seen_refs.add(ref_key)
            canonical_refs.append((source, article_text))
            if len(canonical_refs) >= top_k:
                break
        if len(canonical_refs) >= top_k:
            break

    if not canonical_refs:
        # 极端情况下无法回填完整法条，则退回原始向量召回片段
        canonical_refs = [
            (doc.metadata.get("source", "未知法条来源"), doc.page_content)
            for doc in final_docs
        ]

    # 拼接法条及来源元数据
    formatted_references = []
    for i, (source, article_text) in enumerate(canonical_refs[:top_k], 1):
        # 统一输出格式，便于 Auditor prompt 读取来源和条文正文
        ref_text = f"【参考依据 {i}】(来源: {source})\n{article_text}"
        formatted_references.append(ref_text)
        
    return "\n\n".join(formatted_references)

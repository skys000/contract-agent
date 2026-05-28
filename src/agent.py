# -*- coding: utf-8 -*-
"""
模块名: src/agent.py
作用: 基于 LangGraph 的 Auditor-Critic 双智能体协同审查状态机逻辑。
      LLM 调用通过 OpenAI 兼容协议的大模型 API 驱动，且法条匹配通过本地 RAG 向量检索进行。
"""

import os
import sys
import re
from typing import TypedDict, Dict, Any
from openai import OpenAI
from langgraph.graph import StateGraph, END

# 将当前目录加入查找路径，确保正确引入 retriever 模块
sys.path.append(os.path.dirname(__file__))
from retriever import query_laws
from parser import extract_contract_text

MAX_REFLECTION_ROUNDS = 5
ARTICLE_NUM_PATTERN = r"[一二三四五六七八九十百千万零〇两\d]+"

def _get_llm_client() -> OpenAI:
    """
    创建 OpenAI 兼容协议的大模型客户端，统一读取新旧两套环境变量。
    """
    # 优先读取通用 LLM_* 配置，兼容历史 DEEPSEEK_* 配置
    api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
    if not api_key or not base_url:
        # 大模型是智能体审查的必要依赖，缺失时给出明确配置指引
        raise ValueError("未配置大模型 API，请在 .env 中设置 LLM_API_KEY/LLM_BASE_URL 或兼容的 DEEPSEEK_API_KEY/DEEPSEEK_BASE_URL。")
    # 使用 OpenAI SDK 的兼容协议接口连接 DeepSeek 或其他兼容服务
    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )

def _get_chat_model_name() -> str:
    """
    获取聊天模型名称，兼容 `CHAT_MODEL_NAME` 与旧版 `DEEPSEEK_MODEL_NAME`。
    """
    # 优先使用通用模型变量，未设置时回退到旧版 DeepSeek 变量
    model_name = os.getenv("CHAT_MODEL_NAME") or os.getenv("DEEPSEEK_MODEL_NAME")
    if not model_name:
        # 模型名缺失时无法发起 chat.completions 请求
        raise ValueError("未配置聊天模型名称，请在 .env 中设置 CHAT_MODEL_NAME 或兼容的 DEEPSEEK_MODEL_NAME。")
    return model_name

def _clean_report_text(text: str) -> str:
    """
    清理模型输出中可能出现的寒暄、身份声明或重复引导语，保留正式报告正文。
    """
    # 去除首尾空白，先获得基础干净文本
    cleaned = text.strip()
    # 移除模型常见的“好的/当然/作为专家”等开场话术
    cleaned = re.sub(r"^(好的|当然|以下是|作为专业的中国劳动法务审查专家智能体|我将依据|我已仔细阅读)[^\n]*\n+", "", cleaned)
    # 移除重复说明“根据参考依据报告如下”等非正式报告正文
    cleaned = re.sub(r"(?m)^根据您提供的【?参考的劳动法律法规条款依据】?.*?报告如下：\s*$", "", cleaned)
    # 再兜底清理单行“好的，...”式寒暄
    cleaned = re.sub(r"(?m)^好的，.*$", "", cleaned)
    return cleaned.strip()

def _clean_feedback_text(text: str) -> str:
    """
    清理 Critic 反馈中的非结构化开场白，便于路由器稳定识别通过/未通过状态。
    """
    # 复用报告清洗逻辑，先去掉通用寒暄
    cleaned = _clean_report_text(text)
    # Critic 有时会输出额外的“我的审计结论如下”，这里统一删除
    cleaned = re.sub(r"(?m)^我的审计结论如下：\s*$", "", cleaned)
    # 删除过程性描述，保留可被路由器识别的审核结论
    cleaned = re.sub(r"(?m)^正在进行.*$", "", cleaned)
    return cleaned.strip()

def _normalize_law_name(name: str) -> str:
    """
    将法规文件名、模型引用名统一归一化为系统内部可比较的法名。
    """
    # 优先匹配系统已知法规关键词，避免文件名后缀、年份和全称差异影响比较
    for keyword in [
        "关于企业实行不定时工作制和综合计算工时工作制的审批办法",
        "最高人民法院关于审理劳动争议案件适用法律问题的解释",
        "女职工劳动保护特别规定",
        "劳动争议调解仲裁法",
        "计算机软件保护条例",
        "工资支付暂行规定",
        "工伤保险条例",
        "社会保险法",
        "著作权法",
        "专利法",
        "劳动合同法",
        "劳动法",
    ]:
        if keyword in name:
            return keyword
    # 未命中已知法规时，删除常见国家名前缀、年份后缀、扩展名和空白
    return re.sub(r"中华人民共和国|_\d+|\.docx|\.pdf|\.txt|\s", "", name)

def _extract_law_refs(text: str) -> list[tuple[str, str]]:
    """
    从报告文本中抽取《法律名称》第X条形式的法律引用。
    """
    refs = []
    # 先定位所有《法律名称》出现的位置
    law_matches = list(re.finditer(r"《([^》]+)》", text))
    for match in law_matches:
        # 将模型输出的法律名称归一化，便于与本地法库文件名匹配
        law_name = _normalize_law_name(match.group(1))
        # 只检查法律名称后方近距离文本，避免跨段落误抓条号
        segment = text[match.end():match.end() + 80]
        citation_match = re.match(rf"\s*第{ARTICLE_NUM_PATTERN}条(?:\s*[、,，和及]\s*第?{ARTICLE_NUM_PATTERN}条)*", segment)
        if not citation_match:
            continue
        # 支持“第五十八条、第六十条”这类合并引用，将每个条号拆开
        articles = re.findall(rf"第?({ARTICLE_NUM_PATTERN})条", citation_match.group(0))
        for article in articles:
            ref = (law_name, article)
            if ref not in refs:
                refs.append(ref)
    return refs

def _retrieved_laws_contain_ref(retrieved_laws: str, law_name: str, article: str) -> bool:
    """
    判断 RAG 初始召回内容中是否已经包含指定法名和条号。
    """
    # 将待查法名归一化，和 RAG 来源文本中的法名保持同一比较口径
    normalized_law = _normalize_law_name(law_name)
    # 先定位条号，再检查条号附近是否出现对应法名
    for match in re.finditer(rf"第{article}条", retrieved_laws):
        context = retrieved_laws[max(0, match.start() - 160):match.end() + 160]
        if normalized_law in context:
            return True
    return False

def _split_risk_blocks(raw_audit: str) -> list[tuple[str, str, str]]:
    """
    将 Auditor 草稿按高/中/低风险标题切分为独立风险块，供质量门和补充法条扫描使用。
    """
    # 风险项标题统一使用【高风险】/【中风险】/【低风险】，这里按标题定位每个块的起点
    matches = list(re.finditer(r"(?m)^#{1,6}\s*【(高风险|中风险|低风险)】[^\n]*", raw_audit))
    blocks = []
    for idx, match in enumerate(matches):
        # 当前风险块默认截止到下一个风险标题或全文末尾
        next_risk_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_audit)
        # 若风险块内部提前出现新的一级/二级/三级章节标题，则在章节标题前截断
        next_section = re.search(r"(?m)^#{1,3}\s+", raw_audit[match.end():next_risk_start])
        end = match.end() + next_section.start() if next_section else next_risk_start
        blocks.append((match.group(1), match.group(0), raw_audit[match.start():end]))
    return blocks

def _find_missing_supplemental_refs(raw_audit: str, retrieved_laws: str) -> list[str]:
    """
    找出报告中引用但未出现在 RAG 召回结果、且没有显式标注为补充依据的法条。
    """
    missing_supplements = []
    # 逐个风险块检查，避免跨风险项混淆法条引用
    for _, _, block in _split_risk_blocks(raw_audit):
        # 先找出模型已明确标为“补充法条依据”的引用
        supplemental_refs = set(_extract_explicit_supplemental_refs(block))
        for law_name, article in _extract_law_refs(block):
            if (law_name, article) in supplemental_refs:
                continue
            formatted_ref = f"《{law_name}》第{article}条"
            # 若 RAG 初始依据不含该条，且模型未标补充依据，则记录为缺失标注
            if not _retrieved_laws_contain_ref(retrieved_laws, law_name, article) and formatted_ref not in missing_supplements:
                missing_supplements.append(formatted_ref)
    return missing_supplements

def _extract_explicit_supplemental_refs(text: str) -> list[tuple[str, str]]:
    """
    提取风险块中已由模型显式标注为“补充法条依据”的法律引用。
    """
    refs = []
    for line in text.splitlines():
        # 只处理包含“补充法条依据”的行，减少误抓普通依据
        if "补充法条依据" not in line:
            continue
        marker_index = line.find("补充法条依据")
        # 优先解析标记后方引用；若模型把法条写在标记前，则回看前方 60 个字符
        refs_after_marker = _extract_law_refs(line[marker_index:])
        context_refs = refs_after_marker or _extract_law_refs(line[max(0, marker_index - 60):marker_index])
        for ref in context_refs:
            if ref not in refs:
                refs.append(ref)
    return refs

def _find_supplemental_law_refs(raw_audit: str, retrieved_laws: str) -> list[tuple[str, str]]:
    """
    汇总最终报告需要追加原文的补充法条引用。
    """
    supplemental_refs = []
    for _, _, block in _split_risk_blocks(raw_audit):
        # 显式补充依据和未在 RAG 中出现的引用，都需要在最终报告末尾补全文
        explicitly_supplemental_refs = set(_extract_explicit_supplemental_refs(block))
        for law_name, article in _extract_law_refs(block):
            ref = (law_name, article)
            if ref in explicitly_supplemental_refs or not _retrieved_laws_contain_ref(retrieved_laws, law_name, article):
                if ref not in supplemental_refs:
                    supplemental_refs.append(ref)
    return supplemental_refs

def _extract_article_text_from_law_text(law_text: str, article: str) -> str:
    """
    从完整法规文本中按条号截取单条法条全文。
    """
    # 从目标条号截取到下一条、下一章或文件结尾，保证补充依据是完整条文
    pattern = rf"(?m)^第{re.escape(article)}条[　 \t]?[\s\S]*?(?=^第{ARTICLE_NUM_PATTERN}条[　 \t]?|^第{ARTICLE_NUM_PATTERN}章[　 \t]?|\Z)"
    match = re.search(pattern, law_text)
    return match.group(0).strip() if match else ""

def _lookup_law_article_text(law_name: str, article: str) -> tuple[str, str]:
    """
    在本地 `data/laws` 法规库中按法名和条号精确查找法条全文。
    """
    # 法规原文统一放在项目 data/laws 目录
    laws_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "laws")
    if not os.path.isdir(laws_dir):
        return "", ""
    candidates = []
    for file_name in os.listdir(laws_dir):
        file_path = os.path.join(laws_dir, file_name)
        # 只扫描当前解析器支持的法规文件
        if not os.path.isfile(file_path) or os.path.splitext(file_name)[1].lower() not in [".docx", ".pdf", ".txt"]:
            continue
        # 将文件名归一化后与报告引用法名比较
        normalized_source = _normalize_law_name(file_name)
        if law_name == normalized_source or law_name in normalized_source or normalized_source in law_name:
            candidates.append((file_name, file_path))
        elif law_name in ["专利法", "著作权法", "计算机软件保护条例"] and "知识产权法" in file_name:
            # 知识产权相关法规可能被合并整理到同一个本地文件中
            candidates.append((file_name, file_path))
        elif law_name == "最高人民法院关于审理劳动争议案件适用法律问题的解释" and "最高法劳动争议司法解释" in file_name:
            # 兼容本地文件的短命名
            candidates.append((file_name, file_path))
    for file_name, file_path in candidates:
        try:
            # 解析候选法规全文并按条号提取目标条文
            article_text = _extract_article_text_from_law_text(extract_contract_text(file_path), article)
        except Exception:
            # 单个法规文件解析失败不影响其他候选文件继续尝试
            article_text = ""
        if article_text:
            return file_name, article_text
    return "", ""

def _format_supplemental_refs_section(raw_audit: str, retrieved_laws: str) -> str:
    """
    生成最终报告末尾的“补充法条依据全文”章节。
    """
    # 找出正文中需要在尾部补充全文的法条引用
    supplemental_refs = _find_supplemental_law_refs(raw_audit, retrieved_laws)
    if not supplemental_refs:
        # 没有补充依据时不追加额外章节
        return ""
    # 先写入章节标题和说明，提示这些条文不是初始 RAG 直接提供的依据
    lines = [
        "## 四、 补充法条依据全文",
        "以下为报告正文中引用但未出现在本次 RAG 初始检索结果中的法条，或由 AI 标注为“补充法条依据”的条文。系统已从本地法条库按法名和条号检索并尽量补充原文："
    ]
    for idx, (law_name, article) in enumerate(supplemental_refs, 1):
        # 按法名和条号从本地 data/laws 精确查找原文
        source, article_text = _lookup_law_article_text(law_name, article)
        lines.append(f"### {idx}. 《{law_name}》第{article}条（补充法条依据）")
        if article_text:
            # Markdown 引用块中每一行都加 >，保证多行法条显示整齐
            quoted_article = article_text.replace("\n", "\n> ")
            lines.append(f"- **来源**: {source}")
            lines.append(f"> {quoted_article}")
        else:
            # 查不到时明确提示人工补充，避免误以为系统已经核验全文
            lines.append("- **来源**: 未在本地法条库中精确匹配到该条原文，请人工补充或将对应法规文件加入 `data/laws/` 后重新生成报告。")
    return "\n".join(lines)

def _format_supplemental_refs_context(raw_audit: str, retrieved_laws: str) -> str:
    """
    为 Critic 复核上下文补充 Auditor 已引用但初始 RAG 未命中的本地法条全文。
    """
    supplemental_refs = _find_supplemental_law_refs(raw_audit, retrieved_laws)
    if not supplemental_refs:
        return ""
    lines = [
        "以下法条由系统根据 Auditor 草稿中的引用，从本地法条库按法名和条号精确补充；它们未必属于初始 RAG 命中结果，但可用于复核法条真实性和法律效果。"
    ]
    for idx, (law_name, article) in enumerate(supplemental_refs, 1):
        source, article_text = _lookup_law_article_text(law_name, article)
        if article_text:
            lines.append(f"【补充依据 {idx}】(来源: {source})\n《{law_name}》第{article}条\n{article_text}")
        else:
            lines.append(f"【补充依据 {idx}】《{law_name}》第{article}条\n未在本地法条库中精确匹配到原文，请视为待人工核验引用。")
    return "\n\n".join(lines)

def _extract_declared_risk_counts(raw_audit: str) -> dict[str, int]:
    """
    从报告前言中提取模型声明的高/中/低风险数量，用于和实际风险块数量做一致性校验。
    """
    # 只截取第一个风险项标题之前的整体结论区域，避免误抓正文风险项数量
    preface = raw_audit.split("#### 【", 1)[0]
    # 去掉 Markdown 强调标记并压缩空白，提升正则匹配稳定性
    normalized_preface = re.sub(r"[*_`]", "", preface)
    normalized_preface = re.sub(r"\s+", " ", normalized_preface)
    counts = {}
    for level in ["高风险", "中风险", "低风险"]:
        # 兼容“高风险：2项”和“2项高风险”等常见表述
        match = re.search(rf"{level}(?:项|违规|优化建议)?\s*[:：]?\s*(\d+)\s*项|(\d+)\s*项{level}", normalized_preface)
        if match:
            counts[level] = int(match.group(1) or match.group(2))
    return counts

def _feedback_is_only_supplemental_marking(feedback: str) -> bool:
    """
    判断 Critic 的未通过反馈是否仅要求补充法条标注，避免非阻塞格式问题导致无限返工。
    """
    # 只有 Critic 明确未通过时，才需要进一步判断是否属于可放行问题
    if "【未通过审核】" not in feedback:
        return False
    # 如果反馈不涉及补充法条依据，就不是本函数负责的可放行场景
    if "补充法条依据" not in feedback and "未出现在" not in feedback:
        return False
    # 以下关键词代表实体法律错误或结构性问题，不能仅因“补充依据”字样而放行
    blocking_patterns = [
        r"试用期.{0,12}上限",
        r"风险等级|不得列为|应列为|升为|降为|降级",
        r"工时和加班风险项应引用|完全遗漏实质工时|未引用",
        r"重复计项|风险总数|输出格式|标题格式",
        r"不存在|捏造|明显错误",
        r"遗漏.{0,12}(风险|违法|劳动防护|健康证|体检费|培训费)",
        r"工资|社保|社会保险|加班|工时|工伤|三期|女职工|孕期|产期|哺乳|违约金|解除|防护用品|健康证|体检费|培训费",
    ]
    return not any(re.search(pattern, feedback) for pattern in blocking_patterns)

def _feedback_is_only_nonblocking_optimization(feedback: str) -> bool:
    """
    判断 Critic 反馈是否仅属于低影响优化建议。
    """
    # 只处理带“未通过审核”的反馈；已通过反馈无需转换
    if "【未通过审核】" not in feedback:
        return False
    # 阻塞型关键词表示会影响法律结论、风险等级或风险数量，必须返工
    blocking_patterns = [
        r"试用期.{0,12}上限",
        r"最低工资标准具体金额|未核实的最低工资",
        r"风险数量|风险总数|声明.{0,12}实际",
        r"遗漏.{0,20}(高风险|中风险|核心|强制性义务|工资|社保|社会保险|加班|工时|工伤|三期|女职工|孕期|产期|哺乳|违约金|解除|竞业)",
        r"应列为.{0,12}高风险|调整为.{0,12}高风险|升为.{0,12}高风险|低风险.{0,12}(重新评级|调整|升).*高风险|降为.{0,12}低风险",
        r"不存在|捏造|虚构|编造|方向错误|明显错误|法条引用错误|重复计项|完全重复",
        r"严重违法事项|剥夺劳动者核心权利",
    ]
    # 非阻塞关键词通常只涉及表达、标题、可读性或示范条款优化
    nonblocking_patterns = [
        r"标题|措辞|表述|更精确|更精准|补充说明|计算过程|格式统一|编号|可读性|示范条款|低风险项标题|保持原内容不变|非阻塞",
    ]
    return any(re.search(pattern, feedback) for pattern in nonblocking_patterns) and not any(re.search(pattern, feedback) for pattern in blocking_patterns)

def _feedback_is_approved(feedback: str) -> bool:
    """
    判断 Critic 是否明确给出通过审核结论。
    """
    # 通过标记必须出现在清理后文本开头，且不能同时包含未通过标记
    normalized = feedback.strip()
    return normalized.startswith("【通过审核】") and "【未通过审核】" not in normalized

def _detect_audit_quality_issues(raw_audit: str, retrieved_laws: str, contract_text: str = "") -> list[str]:
    """
    执行确定性质量门检查，覆盖格式、风险计数、非法栏目和少量机器可判定底线规则。

    该函数只做程序确定性校验，不替代 Auditor/Critic 的实体劳动法判断。
    """
    issues = []
    # 将报告切分成风险块，用于计数和格式检查
    risk_blocks = _split_risk_blocks(raw_audit)
    # 提取整体结论中模型自称的风险数量
    declared_counts = _extract_declared_risk_counts(raw_audit)
    # 统计正文中实际出现的高/中/低风险标题数量
    actual_counts = {
        "高风险": sum(1 for level, _, _ in risk_blocks if level == "高风险"),
        "中风险": sum(1 for level, _, _ in risk_blocks if level == "中风险"),
        "低风险": sum(1 for level, _, _ in risk_blocks if level == "低风险"),
    }
    # 检查整体结论数量是否和正文实际条目一致
    mismatched_counts = [f"{level}声明{declared_counts[level]}项、实际{actual_counts[level]}项" for level in declared_counts if declared_counts[level] != actual_counts[level]]
    if mismatched_counts:
        issues.append("整体结论中的风险数量必须与下方具体风险项数量一致：" + "；".join(mismatched_counts))
    # 对“一年期合同试用期上限”这一高频错误做确定性拦截
    if re.search(r"一年.{0,12}试用期.{0,12}上限.{0,12}一个月|一年期限合同试用期上限为一个月", raw_audit):
        issues.append("一年期劳动合同试用期上限应为二个月，不是一个月。")
    # 禁止把实质违法点放到“不计入评级”的摘要栏目中
    if "其他法律风险摘要" in raw_audit or "不在上述风险评级" in raw_audit:
        issues.append("不得将实质性违法点放入不计入评级的其他摘要栏目。")
    for level, title, block in risk_blocks:
        # 每个风险块必须包含依据/分析/修改建议等关键栏目，防止模型只输出标题
        if not re.search(r"\*\*(违规条款|审查依据|问题描述|条款|法律分析|风险分析|依据|违规依据|建议修改后条款|优化建议)", block):
            issues.append(f"{title} 缺少必要的审查依据、法律分析或建议修改内容。")
    return issues

# 1. 定义智能体流转的状态结构 (TypedDict)
class AgentState(TypedDict):
    contract_text: str       # 待审查的劳动合同文本
    retrieved_laws: str      # 通过 RAG 检索出来的相关法定条文参考依据
    raw_audit: str           # Auditor 智能体生成的初审报告草稿
    feedback: str            # Critic 智能体给出的反思、纠错或通过性反馈意见
    final_report: str        # 组装润色后的最终版劳动合同合规审查报告
    loop_count: int          # 反思循环计数器，防范大模型无限纠错死循环

# 2. 节点1: Auditor (初审智能体)
def auditor_node(state: AgentState) -> Dict[str, Any]:
    """
    负责对照参考法条，对合同文本进行初步合规审查，指出高/中/低风险违规点，并提供整改条款。
    """
    # 实例化连接 OpenAI 兼容协议大模型 API 的客户端
    client = _get_llm_client()
    
    # 动态调取 RAG 检索器。如果没有拉取过法条，则执行检索
    retrieved_laws = state.get("retrieved_laws", "")
    if not retrieved_laws:
        # 本地 FAISS 向量库路径
        db_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faiss_index")
        try:
            # 以合同原文作为 Query 检索最相关的劳动法规
            retrieved_laws = query_laws(state["contract_text"], db_dir, top_k=10)
        except Exception as e:
            # 容错降级：RAG 检索失败时，给出基础提示，防止流程中断
            retrieved_laws = f"【RAG 检索失败，已降级】: {e}"
            
    # 初审提示词模版
    prompt = f"""
你是一个专业的中国劳动法务审查专家智能体（Auditor）。请只输出正式审查报告正文，不要输出“好的”“我将”“作为专家”等寒暄或身份声明。

【参考的劳动法律法规条款依据】:
{retrieved_laws}

【待审查的合同条款原文】:
{state['contract_text']}

【历史审查建议与反思反馈 (若有)】:
{state.get('feedback', '暂无。这是第一轮初审，请输出全面细致的审查草稿。')}

【审查要求】:
1. 审查定位：你只审查中国劳动合同合规问题，不做通用商业合同点评。优先依据【参考的劳动法律法规条款依据】逐条核对合同原文；没有明确违法事实的，不得为了凑数量强行判定风险。
2. 风险分级：高风险用于明显违反强制性规定、剥夺劳动者核心权利、可能导致条款无效、行政处罚或重大赔偿的问题；中风险用于明确不利于劳动者且有较明确败诉、补偿或整改风险的问题；低风险仅用于表述优化、证据留痕、程序完善或合同完整性补充。
3. 核心强制义务：社会保险缴纳、加班工资、工资按月足额支付、违法违约金、任意解除、工伤责任转嫁、女职工特殊保护、试用期上限等强制性义务，如合同存在明确违反表述，必须列入正式【高风险】或【中风险】风险项，不得放入“不计入评级”“其他摘要”“补充提示”等非正式栏目。
4. 明确高风险：以现金补贴替代或放弃社会保险；免除加班工资；工资可跨月或跨季度延期支付；要求劳动者承担普通离职违约金；将工伤、疾病或职业伤害责任转嫁给劳动者；约定孕期、产期、哺乳期可解除或降低待遇；赋予甲方无法定事由的任意解除权，原则上判为【高风险】。
5. 试用期：必须严格按《劳动合同法》第十九条判断。三个月以上不满一年不得超过一个月；一年以上不满三年不得超过二个月；三年以上固定期限和无固定期限不得超过六个月。一年期合同试用期上限是二个月；六个月合同可以约定不超过一个月的试用期；三年整固定期限合同约定六个月试用期不得仅因“三年整”判定违法。
6. 工资与最低工资：不得臆测地方最低工资标准，不得写“某地最低工资通常高于/低于某金额”，也不得写“现行/2026年/当地最低工资为XXXX元”等地方最低工资具体金额。除非具体金额明确来自【参考的劳动法律法规条款依据】或合同原文，否则只写“不得低于劳动合同履行地最新公布的最低工资标准”。可以写明根据合同金额直接计算出的结果，例如“转正工资24000元的80%为19200元”。离职当月不支付绩效、奖金、提成等浮动报酬，通常作为中风险或并入工资支付风险；只有明确拒付已提供正常劳动对应的基本工资、最低工资或已确定应发工资时，才列为高风险。
7. 工时与加班：工时和加班风险项应引用《劳动法》第三十六条、第四十一条、第四十四条等对应依据；不得仅引用工资支付条款替代工时限制依据。不得把“超长工时、强制加班、加班费包干/未支付加班费”拆成多个重复高风险，原则上合并为一个工时与加班风险项。
8. 工作地点与单方变更：销售、技术、HR等岗位如因业务需要存在短期出差、客户拜访、项目现场支持，不得直接认定为中风险；只有出现“甲方可单方调整常驻城市/长期驻场且劳动者必须服从”等重大变更授权时，才列为风险。若同时绑定违法解除、降薪、违约金、处罚或“拒绝视为离职/严重违纪”，应合并为一个【高风险】单方变更劳动合同核心条款及违法解除风险项。
9. 重复计项控制：同一违法事实不得重复计入多个风险等级。同一条款同时涉及“拒绝调岗视为离职”“客户要求优先于合同”“项目结束当然终止”的，应围绕共同法益合并；社保替代可说明工伤保险待遇风险，但只有合同明示“工伤自负”“职业伤害责任由劳动者承担”等责任转嫁表述时，才单独列为工伤责任转嫁高风险。
10. 法条引用：每个风险项必须精确引用对应法条序号。优先引用【参考的劳动法律法规条款依据】；如确有必要引用未出现在参考依据但属于本地法条库可补全的真实条文，可在该风险项标明“补充法条依据”，系统会在最终报告末尾自动检索本地法条库并追加全文，不得自行编造法条全文。不得泛泛引用《专利法》《著作权法》《民法典》等无具体条号的法律名称；如不能精确到条号，应改为事实分析或提示需人工补充依据。不得写“法定30%”“最低30%”“不低于平均工资30%”等竞业限制补偿比例，除非该比例和来源明确出现在参考依据或可补全的具体补充法条中；否则只写“在竞业限制期内依法按月支付经济补偿”。写出最低工资、社平工资、缴费基数等具体地方或动态标准时，必须有参考依据或合同原文支持；否则只写“以劳动合同履行地最新公布标准为准”。
11. 修改建议：每个风险项必须给出可直接替换的“建议修改后条款”。建议条款不得引入新的违法点，不得写死动态标准，不得让劳动者放弃法定权利。合同整体合规时，应明确输出“未发现高风险/中风险违规项”，只列对签署、履行或争议解决有实际帮助的必要低风险优化建议，原则上不超过3项。
12. 场景适配：必须以当前合同文本的行业、岗位和工作模式为准，不得套用上一份合同或示例行业。餐饮门店合同重点关注工时排班、后厨/传菜/清洁劳动保护、健康证和体检费用、工资压付、罚款扣款、社保替代、工伤事故；互联网/研发/科技合同重点关注项目责任制、996/值班上线排障、特殊工时审批、加班费包干、培训服务期、竞业限制、知识产权归属、开源/个人作品、绩效淘汰、异地/客户现场驻场和工资扣减。
13. 外包/项目制/客户单位管理：重点审查是否以外包、项目人员、自主择业等名义规避劳动关系或用工主体责任；是否由客户单位直接安排岗位、地点、时间、纪律和考核；是否将客户确认或项目回款作为工资支付前提；是否将客户项目结束、客户不满意作为当然解除或终止条件；是否让劳动者自行承担社保、工伤、税费和商业保险；是否让客户单位要求优先于劳动合同。
14. 销售岗位：重点审查提成结算、回款条件、客户投诉、离职后提成、风险保证金、销售费用、外勤不定时工时、客户拜访和常驻城市调整。提成可约定客观、明确、可核验的结算条件，客户回款可作为合理结算条件之一，但不得无限期拖延、由甲方单方任意认定，或因离职、被辞退、客户投诉、后续退货等不确定事项一概取消已达成结算条件的提成；不得设置风险保证金、押金或从工资提成中单方抵扣坏账、客户投诉、市场费用。
15. 销售岗位风险合并：不得将未完成销售目标、拜访量、回款任务或客户反馈不积极直接等同于旷工或严重违纪。若合同同时将业绩未达标按旷工处理、又将业绩未达标作为严重违纪解除，原则上合并为一个“业绩考核违法替代考勤/严重违纪认定”风险项。销售必要业务费用、差旅、交通、通讯、住宿、客户拜访等履职成本被原则上转嫁给劳动者时，通常列为中风险或低风险优化；只有绑定扣罚、拒付工资、保证金抵扣或违法解除时才升为高风险。
16. 研发/科技岗位专项：培训服务期只有在用人单位提供专项培训费用并进行专业技术培训时才可约定；入职培训、导师带教、内部分享、项目实践通常不构成专项培训。竞业限制应审查人员范围、业务范围、期限、地域、经济补偿和违约金合理性，但不得在无明确依据时写死补偿比例。知识产权条款不得无差别占有劳动者离职后、非职务、未使用单位资源完成的个人作品、开源项目或通用技能成果；引用《专利法》《著作权法》《计算机软件保护条例》时必须精确到具体条号，无法精确引用时应以劳动合同合理性、职务成果边界、是否使用单位资源和是否履行本职工作进行分析，不得泛泛引用无条号法律。涉及职务作品或软件著作权时，优先核对《著作权法》第十八条、《计算机软件保护条例》第十三条；不得将《著作权法》第十六条作为职务作品依据。末位淘汰、绩效排名、代码量、Bug数量、客户/业务评价不得直接作为立即解除且无补偿的依据。
17. 低风险与合并克制：合同已明确约定某项法定内容时，不得仅因表达不够详细就升格为中风险；法定代表人信息空缺但甲方名称住所清楚、休息休假条款重复、保密范围不够细、劳动保护/劳动条件表述笼统等，通常作为低风险优化，除非直接绑定违法解除、扣罚工资、剥夺法定假期或拒绝提供安全条件。整体合规合同的低风险优化建议应保持克制，原则上不超过3项；轻微重复、措辞可更优但不影响履行的内容可在结论中一句带过，不必单独列项。劳动者对违法条款作出的概括确认、同意、承诺、放弃等表述，通常作为前述违法条款无效的补充分析，不单独计为新的高风险；只有该条款本身另行设置独立处罚、违约金、解除或赔偿责任时，才单独列项。
18. 返工落实：如果上一轮 Critic 给出修正指令，必须逐条落实，不得重复原有问题。禁止输出“其他法律风险摘要（不在上述风险评级中）”之类栏目；所有实质性违法点必须纳入风险项总数。

【输出格式】:
- 直接从“### 一、整体合规性结论”开始。
- 不要重复外层标题“劳动合同合规智能审查报告”。
- 不要输出寒暄、身份说明或过程说明。
- 每个具体风险项标题必须使用如下格式之一，方便系统统计：
  - `#### 【高风险】风险项1：...`
  - `#### 【中风险】风险项1：...`
  - `#### 【低风险】优化建议1：...`
- 每个具体风险项标题下必须优先列出一行 `- **合同原文**：...`，其中必须从【待审查的合同条款原文】中逐字复制对应风险条款，方便前端定位和高亮风险条款。
- `合同原文` 严禁使用“……”“...”“省略”“等”等任何省略写法；严禁概括、改写、润色、重组、补词、替换标点或改变原句顺序；严禁把法律分析、风险判断或修改建议写入合同原文。
- 如果同一风险涉及多个连续句子，必须逐字完整保留每个原句及原标点；如果涉及多个不连续位置，应在同一行内按原文出现顺序分别完整摘录为 `片段1：...；片段2：...`，不得为了压缩篇幅把中间内容省略。
- `合同原文` 应只摘录触发该风险所必需的最小完整原句或连续句群，不要整章整节复制；但一旦摘录某个句子，必须完整复制该句，不得截断句首、句中或句尾。
- 整体结论中的风险数量必须与下方具体风险项数量一致。
"""

    # 调用大语言模型进行初审
    response = client.chat.completions.create(
        model=_get_chat_model_name(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1  # 采用低温度系数确保法理审查的严谨性与确定性
    )
    
    # 返回 LangGraph 状态增量：初审草稿、已使用法条、循环轮数
    return {
        "raw_audit": _clean_report_text(response.choices[0].message.content),
        "retrieved_laws": retrieved_laws,
        "loop_count": state.get("loop_count", 0) + 1
    }

# 3. 节点2: Critic (反思纠错智能体)
def critic_node(state: AgentState) -> Dict[str, Any]:
    """
    负责对 Auditor 生成的初审报告草稿进行法理二次审计，防止出现幻觉（如捏造不存在的法条等）。
    """
    # Critic 使用同一套大模型客户端，但提示词定位为复核与质量把关
    client = _get_llm_client()
    # 在 Critic 复核前，先把 Auditor 草稿中引用但初始 RAG 未命中的本地法条补入上下文
    law_context = state["retrieved_laws"]
    supplemental_laws_context = _format_supplemental_refs_context(state["raw_audit"], state["retrieved_laws"])
    if supplemental_laws_context:
        law_context = f"{law_context}\n\n【Auditor 引用的本地补充法条】\n{supplemental_laws_context}"
    
    prompt = f"""
你是一个资深的中国劳动合同法务总监与反思审计智能体（Critic）。请对下属 Auditor 提交的劳动合同审查草稿进行双重反思审计。

你的职责是做劳动合同专项复核，而不是通用合同复核。程序质量闸门只负责格式、数量一致性、补充法条尾注和少数机器可判定底线；劳动法实质判断由你负责。

【待审查合同原文】:
{state['contract_text']}

【Auditor 初步审查草稿】:
{state['raw_audit']}

【参考的法律依据】:
{law_context}

【劳动合同专项复核评分表】:
1. 事实覆盖完整性：是否覆盖劳动合同期限、试用期、工作内容与地点、工时休假、劳动报酬、社会保险、劳动保护、解除终止、违约责任、女职工保护、工伤责任等核心模块；是否遗漏当前场景的高频争议点。
2. 强制性义务识别：是否准确识别放弃或替代社保、免除加班费、工资拖欠或克扣、普通离职违约金、任意解除、工伤责任转嫁、三期女职工解除或降待遇、违法试用期等劳动法强制性问题。
3. 风险等级合理性：高风险用于明显违反强制性规定、剥夺劳动者核心权利或可能导致行政处罚/重大赔偿/条款无效的问题；中风险用于明确不利安排且有较明确败诉或赔偿风险的问题；低风险仅用于表述优化、证据留痕、程序完善等非核心违法问题。不得把工资拖欠、社保替代、免除加班费、违法违约金、任意解除等降为低风险，也不得把单纯表述不完整过度升为高风险。
4. 法条准确性：引用法条必须真实、法名和条号方向正确。真实存在但未出现在参考依据中的法条，如正文未标注“补充法条依据”，不需要仅因此要求返工，系统会在最终报告末尾自动检索本地法条库并追加“补充法条依据全文”。如果法条名称、条号或法律效果明显错误，则应要求返工。不得泛泛引用无具体条号的法律名称；如出现《专利法》《著作权法》《计算机软件保护条例》《民法典》等无条号引用，应要求改为精确条文、事实分析或人工补充依据。
5. 修改建议可执行性：建议条款应能直接替换合同条款，不得引入新的违法点；涉及最低工资、社平工资、地方缴费基数、竞业限制补偿比例等动态事项或裁判规则时，不得编造、假设或写死具体数值。凡出现“法定30%”“最低30%”“不低于平均工资30%”等竞业限制补偿比例，若参考依据或具体补充法条中未明确出现该比例和来源，应要求返工改为“在竞业限制期内依法按月支付经济补偿”。除非具体数值明确来自参考依据、补充法条或合同内可直接计算，否则最低工资、社平工资、地方缴费基数应表述为“以劳动合同履行地最新公布标准为准”。如 Auditor 写出“现行/2026年/当地最低工资为XXXX元”等未由参考依据支持的地方最低工资具体金额，应要求返工改为“劳动合同履行地最新公布的最低工资标准”。
6. 结构与可读性：每个风险项应包含违规条款/审查依据、法律分析或风险分析、依据、建议修改后条款；风险总数应与具体条目一致；不得输出寒暄、身份声明、横向分隔线泛滥或“其他法律风险摘要（不计入评级）”。
7. 合同原文摘录质量：每个风险项的 `- **合同原文**：` 必须逐字来自【待审查合同原文】，不得出现“……”“...”“省略”“等”等省略写法，不得概括、改写、润色、重组、补词、替换标点或改变原句顺序。若同一风险涉及多个连续句子，必须完整保留每个原句；若涉及多个不连续位置，必须按原文顺序分别完整摘录为片段，不得用省略号连接。只要发现合同原文摘录不是逐字完整引用，必须判定【未通过审核】并要求 Auditor 返工。
8. 重复与合并：同一违法事实不得重复计入多个风险等级；工时与加班、业绩考核与违纪解除、单方调岗与拒绝即离职、客户管理介入与外包混同等同一组法益，应优先合并成一个风险项，除非合同存在不同法益的独立违法事实。劳动者概括确认、同意、承诺、放弃前述违法条款的，通常并入前述违法条款分析，不单独计高风险。离职当月不支付绩效、奖金、提成等浮动报酬，通常作为中风险或并入工资支付风险；只有拒付基本工资、最低工资或已确定应发工资时，才支持高风险。
9. 场景适配：必须以当前合同文本的行业、岗位和工作模式为准。餐饮、互联网/研发、销售、外包/项目制合同各有专项高频风险，不得把其他测试合同或示例行业带入当前审查。
10. 外包混同复核：如果 Auditor 同时列出客户/甲方可随时调整工作地点、岗位、时间，客户单位要求优先于劳动合同，乙方拒绝视为主动离职等风险项，应要求合并为一个【高风险】客户管理介入、单方变更劳动合同核心条款及违法解除风险项；若修改建议未明确删除“拒绝视为主动离职/严重违纪”，应要求返工。
11. 销售岗位复核：提成结算条件应客观、明确、可核验；客户回款可以作为合理结算条件之一，但不得无限期拖延、由甲方单方任意认定或因离职、被辞退、客户投诉、后续退货等不确定事项一概取消已达成结算条件的提成。不得设置风险保证金、押金或从工资提成中单方抵扣坏账、客户投诉、市场费用。业绩未达标按旷工处理和业绩未达标直接严重违纪解除应原则上合并。
12. 研发/科技岗位复核：项目责任制、值班上线排障、特殊工时审批、加班费包干、培训服务期、竞业限制、知识产权归属、开源/个人作品、绩效淘汰、异地或客户现场长期驻场、工资扣减均应重点复核。不得遗漏“非职务/未使用单位资源作品全部归单位”“入职培训包装成专项培训服务期”“末位淘汰立即解除”等典型风险；但知识产权条款如无法精确引用条号，应要求改为劳动合同合理性、职务成果边界、是否使用单位资源和是否履行本职工作分析，避免无条号泛引。涉及职务作品或软件著作权时，应核对《著作权法》第十八条、《计算机软件保护条例》第十三条等具体条文；如 Auditor 将《著作权法》第十六条作为职务作品依据，应要求返工。建议条款不得设置对劳动者全部个人成果的无差别报备义务。
13. 合规报告克制性：如果合同整体合规且未发现高风险/中风险，低风险优化建议原则上不超过3项；不得仅因轻微重复、措辞可更优或已由其他条款覆盖的事项机械列项。若低风险超过3项，应复核是否存在可合并或可在结论中一句带过的内容。
14. 试用期建议准确性：六个月劳动合同的试用期上限为一个月，不能把“试用期三个月违法”修正为“根据法律规定不得约定试用期”；两年合同试用期上限为二个月；三年整固定期限合同可约定不超过六个月试用期。

【输出规范】：
- 只有在风险等级、法条引用、输出格式、修改建议均合格时，才可在回复最开头直接输出：“【通过审核】”，无需输出其他修改意见；仅存在补充法条依据漏标时，也视为可通过。
- 如果只存在标题措辞、表述更精确、示范条款可读性、非关键法条补充、计算过程说明、格式统一等不影响法律结论和风险数量的优化建议，应输出“【通过审核】”，可在其后附“非阻塞优化建议”，不得要求返工。
- 如果你发现其中有任何不严谨、法条引用有误、或修改意见不妥之处，请详细写下具体的“修正指令”，以便 Auditor 重新校对。
- 如果未通过，请优先指出劳动合同实质问题，不要只围绕格式措辞；修正指令应能指导 Auditor 直接改写风险项。
- 不要输出“好的”“作为……智能体”“我已收到”等寒暄或身份声明，直接从“【未通过审核】”或“【通过审核】”开始。
"""

    response = client.chat.completions.create(
        model=_get_chat_model_name(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2  # 稍微允许一些发散以促进更深入的潜在风险反思
    )
    
    # 读取 Critic 原始反馈，随后叠加程序确定性质量门
    feedback = response.choices[0].message.content
    # 质量门用于拦截机器可判定问题，如风险数量不一致、明显错误试用期结论等
    deterministic_issues = _detect_audit_quality_issues(state["raw_audit"], state["retrieved_laws"], state["contract_text"])
    if deterministic_issues:
        # 如果程序质量门发现问题，即使模型写了通过，也强制改写为未通过
        feedback_without_approval = re.sub(r"【\s*通过审核\s*】", "", feedback)
        feedback = "【未通过审核】\n" + "\n".join(f"{idx + 1}. {issue}" for idx, issue in enumerate(deterministic_issues)) + "\n" + feedback_without_approval
    elif _feedback_is_only_supplemental_marking(feedback):
        # 只缺“补充法条依据”标注时不返工，最终报告阶段会自动补全文
        feedback = "【通过审核】\n仅存在补充法条依据漏标，最终报告将自动追加补充法条标注说明。"
    elif _feedback_is_only_nonblocking_optimization(feedback):
        # 只涉及标题、措辞、可读性等非阻塞优化时放行，避免无意义循环
        feedback = "【通过审核】\n仅存在不影响法律结论、风险等级和风险数量的非阻塞优化建议，本轮不再返工。"

    # 返回 LangGraph 状态增量：本轮 Critic 反馈
    return {
        "feedback": feedback
    }

# 4. 节点3: ReportGenerator (最终报告润色组装节点)
def report_generator_node(state: AgentState) -> Dict[str, Any]:
    """
    接收最终通过的草稿，剔除中间调试及反思标记，格式化为最终版报告。
    """
    # loop_count 包含首次生成，因此实际修正次数需要减一
    revision_rounds = max(state['loop_count'] - 1, 0)
    # 判断 Critic 是否最终通过，用于确定报告状态
    approved = _feedback_is_approved(state.get("feedback", ""))
    # 如果未通过且流程触顶，则保留未解决反馈给人工复核
    unresolved_feedback = _clean_feedback_text(state.get("feedback", "")) if not approved and "【未通过审核】" in state.get("feedback", "") else ""
    review_status = "双智能体（Auditor & Critic）协同会审完毕，报告已完成反思审计"
    # 对正文中需要补充全文的法条，统一在报告末尾追加
    supplemental_refs_section = _format_supplemental_refs_section(state["raw_audit"], state["retrieved_laws"])
    extra_sections = []
    if supplemental_refs_section:
        extra_sections.append(supplemental_refs_section)
    if unresolved_feedback:
        # 达到最大轮数仍未通过时，报告不丢弃，而是明确标记需人工复核
        review_status = "已达到最大反思轮数，仍存在未完全解决的质检问题，建议人工复核后使用" if state['loop_count'] >= MAX_REFLECTION_ROUNDS else "Critic 尚未通过审核，当前报告仍存在未解决质检问题，建议继续反思修正"
        # 如果已有补充法条章节，未解决质检提示顺延为第五章
        unresolved_heading = "## 五、 未解决质检提示" if supplemental_refs_section else "## 四、 未解决质检提示"
        extra_sections.append(f"{unresolved_heading}\n{unresolved_feedback}")
    extra_sections_text = "\n\n".join(extra_sections)
    # 组装最终 Markdown 报告，包含元数据、RAG 依据、风险明细和可选附加章节
    final_output = f"""# 劳动合同合规智能审查报告

## 一、 智能审查元数据
- **审查状态**: {review_status}
- **Auditor 生成轮次**: {state['loop_count']} 轮
- **实际反思修正次数**: {revision_rounds} 次

## 二、 检索到的法定背景参考依据
{state['retrieved_laws']}

## 三、 合规风险项明细与修正对策
{_clean_report_text(state['raw_audit'])}
{extra_sections_text}
"""
    # 返回最终报告字段，供 app.py 渲染和导出
    return {"final_report": final_output}

# 5. 条件路由逻辑 (Router)
def check_approval_router(state: AgentState) -> str:
    """
    控制流程分支：判断是进入下一次修改还是输出报告。
    """
    # 从共享状态中读取 Critic 反馈和当前循环轮数
    feedback = state.get("feedback", "")
    loop_count = state.get("loop_count", 0)
    # 只有以【通过审核】开头且不含【未通过审核】时才视为通过
    approved = _feedback_is_approved(feedback)
    
    # 条件1：Critic 判定通过审核
    # 条件2：触发最大反思循环次数限制（防止多智能体进入无限死锁循环）
    if approved or loop_count >= MAX_REFLECTION_ROUNDS:
        print(f"[Agent Workflow] 状态流转完成。通过原因: {'Critic 审核通过' if approved else '达到最大反思轮数限制'}。即将输出最终报告。")
        return "generate_report"
    else:
        # 未通过且未达到最大轮数时，携带反馈回到 Auditor 重新生成
        print(f"[Agent Workflow] 第 {loop_count} 轮反思未通过。Critic 反馈提示: {feedback[:100]}... 即将流转回 Auditor 重新审查。")
        return "re_audit"

# 6. 构建并编译 LangGraph 工作流拓扑图
def build_agent_graph():
    """
    构造状态图结构，配置节点与控制链路，并编译执行器。
    """
    # 初始化状态图对象，指定系统共享状态字典
    workflow = StateGraph(AgentState)
    
    # 注册节点逻辑
    workflow.add_node("Auditor", auditor_node)
    workflow.add_node("Critic", critic_node)
    workflow.add_node("ReportGenerator", report_generator_node)
    
    # 配置起点
    workflow.set_entry_point("Auditor")
    
    # 配置强连线：Auditor 初审后，必须流入 Critic 进行复审
    workflow.add_edge("Auditor", "Critic")
    
    # 配置条件分支连线：基于 Critic 评估结果分流
    workflow.add_conditional_edges(
        "Critic",
        check_approval_router,
        {
            "re_audit": "Auditor",              # 不合格：流回 Auditor 重新生成
            "generate_report": "ReportGenerator" # 合格或触顶：流向报告组装节点
        }
    )
    
    # 配置终点
    workflow.add_edge("ReportGenerator", END)
    
    # 编译成可运行的 LangGraph 应用
    return workflow.compile()

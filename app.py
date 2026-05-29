# -*- coding: utf-8 -*-
"""
应用名: app.py
作用: 劳动合同合规审查智能体的 Streamlit 前端交互系统与运营仪表盘大屏。
      基于“苹果极简亮色 (Apple Light Mode)”美学重构，实现信息解耦的多标签导航、
      数据本地安全脱敏比对展示、SQLite 存盘及 RAG 法律文本检索。
"""

import streamlit as st
import os
import sys
import re
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import base64
import html
import unicodedata
from dotenv import load_dotenv
from openai import OpenAI

# 将 src 目录临时加入模块查找路径~X
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from parser import extract_contract_text, desensitize_text, extract_metadata, get_last_parser_message
from database import init_db, insert_audit_log, get_kpi_metrics, get_recent_activities, get_monthly_risk_stats
from agent import build_agent_graph, _lookup_law_article_text
from retriever import query_laws

def count_risk_items(report_text: str, level: str) -> int:
    """
    统计最终审查报告中指定风险等级的条目数，用于看板入库与报告一致性展示。
    """
    # 匹配 Markdown 标题、加粗标题和“高风险项：”等模型可能输出的标题形式
    pattern = rf"(?m)^\s*(?:#{{1,6}}\s*)?(?:\*\*)?(?:【{level}】|{level}项\s*\d+[:：]|{level}项[:：])"
    return len(re.findall(pattern, report_text))

def _clean_highlight_candidate(text: str) -> str:
    """
    清理模型报告中用于高亮定位的候选原文片段，去除 Markdown 标记、标签名前缀与外围标点。
    """
    # 去掉 Markdown 加粗、行内代码和列表符号
    text = re.sub(r"\*\*|`|^[-*]\s*", "", text).strip()
    # 去掉“合同原文：”“违规条款：”等标签，只保留实际候选片段
    text = re.sub(r"^(合同原文|违规条款|问题条款|原文摘录|风险条款|涉及条款|条款内容)\s*[:：]\s*", "", text)
    text = re.sub(r"^片段\s*\d+\s*[:：]\s*", "", text)
    # 去掉外围引号、冒号、逗号和句末标点，提升与合同正文的精确匹配率
    return text.strip(" \t\r\n\u201c\u201d\"\u2018\u2019\uff1a:\uff0c,\u3002\uff1b;\u3001")

def _strip_clause_prefix(text: str) -> str:
    """
    去除报告摘录前常见的“第X条”前缀，解决报告条号与合同正文标题分离导致的漏高亮。
    """
    # 模型摘录经常把“第十二条：”带在片段前，而正文匹配时可能只需要条款内容
    text = re.sub(r"^第[一二三四五六七八九十百千万零〇\d]+条\s*[：:、，,。\s“”\"']*", "", text)
    return text.strip(" \t\r\n\u201c\u201d\"\u2018\u2019\uff1a:\uff0c,\u3002\uff1b;\u3001")

def _is_multi_sentence_fragment(text: str) -> bool:
    """
    判断候选片段是否包含多个句子，多个句子的整段候选只作为兜底匹配，不优先占用高亮区间。
    """
    parts = [
        _clean_highlight_candidate(part)
        for part in re.split(r"[。！？!?；;\n]", text)
        if _clean_highlight_candidate(part)
    ]
    return len(parts) > 1

def _expand_highlight_fragments(candidate: str) -> list[str]:
    """
    将较长的合同原文摘录扩展为多组可匹配片段。

    该函数专门处理模型报告常见的条号、省略号、换行、逗号分隔等格式差异，
    只生成合同中可能精确出现的文本片段，不做语义猜测，避免误高亮。
    """
    fragments = []
    # 同时尝试原始片段和去条号前缀片段，覆盖模型输出与合同正文格式差异
    variants = [candidate, _strip_clause_prefix(candidate)]
    for variant in variants:
        # 对每个变体先做标准清洗
        variant = _clean_highlight_candidate(variant)
        variant = _strip_clause_prefix(variant)
        # 只有真正存在“……/...”时才拆分省略号，避免无省略号的多句整段重新混入候选
        ellipsis_parts = []
        if re.search(r"(?:…{2,}|\.{3,})", variant):
            ellipsis_parts = [
                _strip_clause_prefix(_clean_highlight_candidate(part))
                for part in re.split(r"(?:…{2,}|\.{3,})", variant)
                if _strip_clause_prefix(_clean_highlight_candidate(part))
            ]
        # 后续会对完整片段和省略号拆分片段继续按句子、逗号扩展
        variants_to_split = [variant] + ellipsis_parts
        if variant and not _is_multi_sentence_fragment(variant):
            fragments.append(variant)
        fragments.extend(ellipsis_parts)
        for split_source in variants_to_split:
            # 按句号、分号、换行切句，得到更短、更容易精确定位的风险片段
            for sentence in re.split(r"[。！？!?；;\n]", split_source):
                sentence = _strip_clause_prefix(_clean_highlight_candidate(sentence))
                if sentence:
                    fragments.append(sentence)
                    # 再按逗号拆分子句，用于长句局部命中
                    comma_parts = [
                        _clean_highlight_candidate(part)
                        for part in re.split(r"[，,]", sentence)
                        if _clean_highlight_candidate(part)
                    ]
                    fragments.extend(comma_parts)
                    # 相邻逗号片段重新组合，兼顾片段太短和整句太长两种问题
                    for index in range(len(comma_parts) - 1):
                        fragments.append(f"{comma_parts[index]}，{comma_parts[index + 1]}")
                    # 3 片段滑窗组合，覆盖较长后半句命中
                    for index in range(len(comma_parts) - 2):
                        fragments.append(f"{comma_parts[index]}，{comma_parts[index + 1]}，{comma_parts[index + 2]}")
    return fragments

def _extract_risk_highlight_candidates(report_text: str) -> list[tuple[str, str]]:
    """
    从审查报告的高/中/低风险块中抽取可用于合同预览高亮的原文候选片段。

    优先读取 `合同原文`、`违规条款` 等显式标签；若报告没有标签，再退化读取中文或英文引号中的片段。
    """
    candidates = []
    # 定位每个风险项标题，并捕获风险等级作为高亮颜色依据
    heading_pattern = re.compile(r"(?m)^\s*(?:#{1,6}\s*)?(?:\*\*)?【(高风险|中风险|低风险)】[^\n]*")
    matches = list(heading_pattern.finditer(report_text or ""))
    # 在风险块内部查找模型输出的“合同原文：...”等字段
    label_pattern = re.compile(r"(?:\*\*)?合同原文(?:\*\*)?\s*[:：]\s*(.*)")
    for index, match in enumerate(matches):
        level = match.group(1)
        # 当前风险块范围：从当前标题结束到下一个风险标题开始
        block_end = matches[index + 1].start() if index + 1 < len(matches) else len(report_text)
        block = report_text[match.end():block_end]
        block_candidates = []
        block_lines = block.splitlines()
        for line_index, line in enumerate(block_lines):
            label_match = label_pattern.search(line)
            if label_match:
                inline_text = label_match.group(1)
                following = []
                for next_line in block_lines[line_index + 1:]:
                    stripped_next_line = next_line.strip()
                    if not stripped_next_line:
                        if following or inline_text.strip():
                            break
                        continue
                    if label_pattern.search(stripped_next_line):
                        break
                    if re.match(r"^(?:[-*]\s*)?(?:\*\*)?(风险分析|违规分析|问题分析|法律依据|建议修改后条款|修改后条款|整改建议|优化建议)(?:\*\*)?\s*[:：]", stripped_next_line):
                        break
                    following.append(stripped_next_line)
                combined_candidate = " ".join(part for part in [inline_text.strip(), *following] if part)
                if combined_candidate:
                    block_candidates.append(combined_candidate)
        for candidate in block_candidates:
            # 排除明显不是合同原文的法律名称或修改建议
            candidate = _clean_highlight_candidate(candidate)
            if not candidate or any(keyword in candidate for keyword in ["劳动合同法", "劳动法", "建议修改", "修改后条款"]):
                continue
            for fragment in re.split(r"[。！？!?；;\n]", candidate):
                fragment = _clean_highlight_candidate(fragment)
                # 过短容易误高亮，过长通常难以精确匹配
                if 8 <= len(fragment) <= 220:
                    candidates.append((fragment, level))
    return candidates

def _find_snippet_positions(contract_text: str, snippet: str) -> list[tuple[int, int]]:
    """
    在合同正文中定位候选片段，先做普通精确匹配，再做去空白后的紧凑匹配。
    """
    positions = []
    # 第一轮使用 Python 字符串精确查找，命中时可直接得到原文位置
    start = contract_text.find(snippet)
    while start != -1:
        positions.append((start, start + len(snippet)))
        start = contract_text.find(snippet, start + len(snippet))
    if positions:
        return positions
    # 第二轮构造"归一化紧凑文本"，解决 MinerU 解析产生的零宽字符、全角空格和换行打断问题
    def _is_invisible(c: str) -> bool:
        """判断字符是否为空白或 Unicode 格式/零宽字符，这类字符在紧凑匹配时应跳过"""
        if c.isspace():
            return True
        cat = unicodedata.category(c)
        return cat in ('Cf', 'Cc')
    # 逐字符 NFKC 归一化：全角数字/字母 → 半角；保持 compact_to_original 始终指向原文下标
    compact_chars = []
    compact_to_original = []
    for index, char in enumerate(contract_text):
        if _is_invisible(char):
            continue
        for nc in unicodedata.normalize('NFKC', char):
            compact_chars.append(nc)
            compact_to_original.append(index)
    compact_contract = "".join(compact_chars)
    compact_snippet = "".join(
        nc for c in snippet if not _is_invisible(c)
        for nc in unicodedata.normalize('NFKC', c)
    )
    # 过短片段在紧凑匹配中误命中概率较高，因此直接放弃
    if len(compact_snippet) < 8:
        return positions
    compact_start = compact_contract.find(compact_snippet)
    while compact_start != -1:
        # 将紧凑文本下标映射回原始合同文本下标，保证 HTML 高亮切片正确
        original_start = compact_to_original[compact_start]
        original_end = compact_to_original[compact_start + len(compact_snippet) - 1] + 1
        positions.append((original_start, original_end))
        compact_start = compact_contract.find(compact_snippet, compact_start + len(compact_snippet))
    return positions

def _build_highlight_spans(contract_text: str, report_text: str) -> list[tuple[int, int, str]]:
    """
    根据报告候选片段构造最终高亮区间，并按风险等级优先级避免重叠覆盖。
    """
    spans = []
    occupied = []
    # 风险等级优先级：高风险片段优先占用位置，避免被低风险重叠覆盖
    priority = {"高风险": 0, "中风险": 1, "低风险": 2}
    # 候选片段先去重，再按风险等级和片段长度排序；长片段优先可减少碎片化高亮
    candidates = sorted(
        set(_extract_risk_highlight_candidates(report_text)),
        key=lambda item: (priority.get(item[1], 9), -len(item[0]))
    )
    for snippet, level in candidates:
        for start, end in _find_snippet_positions(contract_text, snippet):
            # 若新区间与已占用区间重叠，则跳过，保持最终高亮区域互不覆盖
            if not any(start < old_end and end > old_start for old_start, old_end, _ in occupied):
                spans.append((start, end, level))
                occupied.append((start, end, level))
    # 按正文位置排序，便于后续从前到后拼接 HTML
    return sorted(spans, key=lambda item: item[0])

def render_contract_preview(contract_text: str, report_text: str = "") -> None:
    """
    渲染带风险高亮的合同预览区域。
    """
    # 根据报告风险项生成正文高亮区间
    spans = _build_highlight_spans(contract_text, report_text)
    level_class = {"高风险": "high", "中风险": "med", "低风险": "low"}
    pieces = []
    cursor = 0
    for start, end, level in spans:
        # 先加入高亮前的普通文本，并进行 HTML 转义防止合同内容破坏页面结构
        pieces.append(html.escape(contract_text[cursor:start]))
        # 当前风险片段用 mark 标签包裹，并根据风险等级绑定不同 CSS class
        pieces.append(f"<mark class='risk-highlight-{level_class.get(level, 'low')}'>{html.escape(contract_text[start:end])}</mark>")
        cursor = end
    # 加入最后一个高亮片段之后的剩余文本
    pieces.append(html.escape(contract_text[cursor:]))
    legend = ""
    if spans:
        # 只展示当前报告实际命中的风险等级图例
        active_levels = {level for _, _, level in spans}
        legend_items = []
        if "高风险" in active_levels:
            legend_items.append("<span class='legend-high'>高风险</span>")
        if "中风险" in active_levels:
            legend_items.append("<span class='legend-med'>中风险</span>")
        if "低风险" in active_levels:
            legend_items.append("<span class='legend-low'>低风险/优化</span>")
        legend = f"<div class='risk-legend'>{''.join(legend_items)}</div>"
    # 使用 unsafe_allow_html 渲染自定义高亮样式，合同内容本身已通过 html.escape 转义
    st.markdown(f"{legend}<div class='contract-preview'>{''.join(pieces)}</div>", unsafe_allow_html=True)

def _get_llm_client() -> OpenAI:
    """
    创建 OpenAI 兼容协议客户端，统一支持通用 LLM_* 配置与旧 DEEPSEEK_* 配置。
    """
    # 优先读取通用大模型配置，兼容历史 DeepSeek 配置名
    api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
    if not api_key or not base_url:
        raise ValueError("未配置大模型 API，请在 .env 中设置 LLM_API_KEY/LLM_BASE_URL 或兼容的 DEEPSEEK_API_KEY/DEEPSEEK_BASE_URL。")
    # OpenAI SDK 支持 base_url，因此可连接任意 OpenAI 兼容供应商
    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )

def _get_chat_model_name() -> str:
    """
    读取聊天模型名称，兼容新旧环境变量命名。
    """
    # 优先使用通用模型名，未配置时回退到旧 DEEPSEEK_MODEL_NAME
    model_name = os.getenv("CHAT_MODEL_NAME") or os.getenv("DEEPSEEK_MODEL_NAME")
    if not model_name:
        raise ValueError("未配置聊天模型名称，请在 .env 中设置 CHAT_MODEL_NAME 或兼容的 DEEPSEEK_MODEL_NAME。")
    return model_name

def _extract_law_reference_pairs(text: str) -> list[tuple[str, str]]:
    """
    从法律咨询首轮回答中提取“法律名称 + 条号”组合，支持一处引用多个条号。
    """
    references = []
    # 匹配《法律名称》第X条，并兼容后续追加“、第Y条、第Z条”
    pattern = re.compile(r"《([^》]{2,40})》\s*第([一二三四五六七八九十百千万零〇两\d]+)条((?:[、,，]\s*第?[一二三四五六七八九十百千万零〇两\d]+条)*)")
    for law_name, article_no, following_articles in pattern.findall(text or ""):
        # 先保存主条号，再解析同一句中的后续条号
        article_numbers = [article_no]
        article_numbers.extend(re.findall(r"第?([一二三四五六七八九十百千万零〇两\d]+)条", following_articles))
        for number in article_numbers:
            ref = (law_name, number)
            if ref not in references:
                references.append(ref)
    return references

def _arabic_to_chinese_article_number(article_no: str) -> str:
    """
    将阿拉伯数字条号转换为中文条号，便于精确匹配本地法规原文中的中文编号。
    """
    # 本地法条多为中文条号；如果输入本身不是纯数字，则直接返回
    if not article_no.isdigit():
        return article_no
    digits = "零一二三四五六七八九"
    number = int(article_no)
    if number == 0:
        return "零"
    if number < 10:
        return digits[number]
    if number < 20:
        return "十" + (digits[number % 10] if number % 10 else "")
    if number < 100:
        tens, ones = divmod(number, 10)
        return digits[tens] + "十" + (digits[ones] if ones else "")
    hundreds, rest = divmod(number, 100)
    result = digits[hundreds] + "百"
    if rest == 0:
        return result
    if rest < 10:
        return result + "零" + digits[rest]
    tens, ones = divmod(rest, 10)
    return result + digits[tens] + "十" + (digits[ones] if ones else "")

def _lookup_local_laws_for_consultation(first_answer: str, question: str, db_dir: str) -> str:
    """
    法律咨询助手的本地法条召回入口。

    先按首轮回答中的明确法名和条号做精确查找；精确查找失败时再回退到 FAISS 语义检索。
    """
    blocks = []
    # 最多处理前 5 个候选引用，控制一次咨询中的本地检索成本
    for law_name, article_no in _extract_law_reference_pairs(first_answer)[:5]:
        # 同时尝试原条号和阿拉伯数字转中文后的条号，提升本地精确命中率
        article_candidates = list(dict.fromkeys([article_no, _arabic_to_chinese_article_number(article_no)]))
        exact_hit = None
        for candidate in article_candidates:
            source, article_text = _lookup_law_article_text(law_name, candidate)
            if article_text:
                # 精确命中时直接返回本地法条全文
                exact_hit = f"【精确法条】《{law_name}》第{candidate}条\n【来源】{source}\n{article_text}"
                break
        if exact_hit:
            blocks.append(exact_hit)
        else:
            # 精确查找失败时，用“法名+条号”作为 query 进行语义召回
            blocks.append(f"【语义检索】《{law_name}》第{article_no}条\n{query_laws(f'《{law_name}》第{article_no}条', db_dir, top_k=3)}")
    if blocks:
        return "\n\n".join(blocks)
    # 首轮回答没有抽取到明确法条时，直接用用户问题做语义检索
    return f"【语义检索】用户问题\n{query_laws(question, db_dir, top_k=5)}"

def _ask_legal_consultant_first_round(question: str) -> str:
    """
    法律咨询第一轮：让模型先给出简短判断并显式列出可能相关的法条编号。
    """
    # 第一轮不直接给最终意见，只让模型识别问题和可能相关条文
    client = _get_llm_client()
    prompt = f"""
你是劳动合同合规系统中的法律咨询预检智能体。请围绕用户问题给出简短初步判断，并列出你认为可能相关的中国劳动法律条文。

要求：
1. 回答必须简洁，不超过 300 字。
2. “可能相关法条”必须一条法条占一行，不要把多个条文合并在同一行。
3. 法条格式固定为：- 《法律名称》第十九条。条号尽量使用中文数字。
4. 即使同一部法律涉及多条，也必须拆开写，例如：
   - 《中华人民共和国社会保险法》第五十八条
   - 《中华人民共和国社会保险法》第六十条
5. 不要把初步判断写成最终法律意见，因为系统稍后会用本地法条库复核。

用户问题：
{question}
"""
    response = client.chat.completions.create(
        model=_get_chat_model_name(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    # 返回首轮预检文本，第二轮会据此抽取法名和条号
    return response.choices[0].message.content.strip()

def _stream_legal_consultant_final_answer(question: str, first_answer: str, local_laws: str):
    """
    法律咨询第二轮：基于本地召回法条流式生成最终答复。
    """
    # 第二轮必须基于本地召回内容作答，降低模型凭空编造法条的概率
    client = _get_llm_client()
    prompt = f"""
你是劳动合同合规系统中的法律咨询助手。请只根据用户问题和本地法条库召回内容，给出正式答复。

约束：
1. 不得编造未出现在本地召回内容中的法条原文。
2. 如果本地召回内容不足以支撑结论，必须明确提示“本地法条依据不足，建议人工核验”。
3. 回答结构为：简要结论、法律依据、实务建议、风险提示。
4. “法律依据”部分必须按条分列，每条法律依据独立成项，不要把多个条文合并成一段。
5. 适合普通劳动者或 HR 阅读，避免过度冗长。

用户问题：
{question}

第一轮预检回答：
{first_answer}

本地法条库召回内容：
{local_laws}
"""
    stream = client.chat.completions.create(
        model=_get_chat_model_name(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.15,
        stream=True
    )
    # 将模型流式增量逐段 yield 给 st.write_stream，实现聊天式输出体验
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

# 预加载环境变量并初始化数据库
# load_dotenv 负责读取 .env；init_db 负责保证本地 audit_logs 表可用
load_dotenv()
init_db()

# 应用名和副标题允许通过环境变量定制，未配置时使用默认中文演示标题
APP_NAME = os.getenv("APP_NAME", "基于多智能体的个人劳动合同风险预警系统")
APP_SUBTITLE = os.getenv("APP_SUBTITLE", "极简、安全、专业的双智能体协同劳动合同风险预警系统")

# 配置 Streamlit 页面属性为宽版模式
st.set_page_config(page_title=APP_NAME, layout="wide", page_icon="🛡️")

# ==========================================
# 1. 苹果极简亮色 (Apple Light-Mode) CSS 样式注入
# ==========================================
st.markdown("""
<style>
    /* 隐藏 Streamlit 头部与发布按钮，消除顶部白条缺陷 */
    header {
        visibility: hidden;
        height: 0px !important;
    }
    .stDeployButton {
        display: none;
    }
    #MainMenu {
        visibility: hidden;
    }
    
    /* 容器间距微调，宽度拉满避免两侧留空 */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 100% !important;
    }

    /* 全局背景色与高对比文本基调 */
    .stApp {
        background-color: #f5f5f7;
        color: #1d1d1f;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Icons", "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    
    /* 自定义大标题 (无渐变) */
    .custom-title {
        font-size: 28px;
        font-weight: 700;
        color: #1d1d1f;
        margin-top: 10px;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }
    
    .custom-subtitle {
        font-size: 13px;
        color: #86868b;
        margin-bottom: 20px;
    }

    /* 改写 Streamlit container(border=True) 样式为苹果白银卡片 */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #d2d2d7 !important;
        border-radius: 12px !important;
        padding: 24px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.02) !important;
        margin-bottom: 16px !important;
    }
    
    /* 输入框与文本域样式苹果亮色化，保证字体清晰可见 */
    div[data-baseweb="textarea"], div[data-baseweb="input"] {
        border-radius: 8px !important;
        border: 1px solid #d2d2d7 !important;
        background-color: #ffffff !important;
        color: #1d1d1f !important;
    }
    textarea {
        color: #1d1d1f !important;
        font-family: monospace !important;
        font-size: 13px !important;
    }
    /* 强力覆盖 disabled 文本域字体颜色与透明度，确保高对比可读性 */
    textarea:disabled {
        color: #1d1d1f !important;
        -webkit-text-fill-color: #1d1d1f !important;
        opacity: 1 !important;
        background-color: #f5f5f7 !important;
    }
    /* 强力覆盖所有输入控件的 label 属性，确保高对比度 */
    .stTextArea label, .stTextInput label, .stFileUploader label {
        color: #1d1d1f !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    /* 列表与活动流表格样式苹果化 */
    .apple-table {
        width: 100%;
        border-collapse: collapse;
        color: #1d1d1f;
        margin-top: 8px;
    }
    .apple-table th {
        background-color: #f5f5f7;
        color: #86868b;
        text-align: left;
        padding: 12px 16px;
        font-weight: 600;
        font-size: 13px;
        border-bottom: 1px solid #d2d2d7;
    }
    .apple-table td {
        padding: 12px 16px;
        font-size: 13px;
        border-bottom: 1px solid #e5e5ea;
        color: #1d1d1f;
    }
    .apple-table th:nth-child(5),
    .apple-table td:nth-child(5),
    .apple-table th:nth-child(6),
    .apple-table td:nth-child(6) {
        white-space: nowrap;
    }
    .apple-table tr:hover {
        background-color: #fafafa;
    }
    
    /* 风险状态扁平化徽章 (高对比亮色背景) */
    .badge-high {
        background-color: rgba(255, 59, 48, 0.12);
        color: #ff3b30;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-med {
        background-color: rgba(255, 149, 0, 0.12);
        color: #ff9500;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-low {
        background-color: rgba(52, 199, 89, 0.12);
        color: #34c759;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
    }
    .contract-preview {
        height: 560px;
        overflow-y: auto;
        white-space: pre-wrap;
        color: #1d1d1f;
        background-color: #f5f5f7;
        border: 1px solid #d2d2d7;
        border-radius: 8px;
        padding: 12px;
        font-family: monospace;
        font-size: 13px;
        line-height: 1.6;
    }
    .risk-highlight-high {
        background-color: rgba(255, 59, 48, 0.22);
        color: #1d1d1f;
        border-bottom: 2px solid #ff3b30;
        padding: 1px 2px;
        border-radius: 3px;
    }
    .risk-highlight-med {
        background-color: rgba(255, 149, 0, 0.22);
        color: #1d1d1f;
        border-bottom: 2px solid #ff9500;
        padding: 1px 2px;
        border-radius: 3px;
    }
    .risk-highlight-low {
        background-color: rgba(52, 199, 89, 0.16);
        color: #1d1d1f;
        border-bottom: 2px solid #34c759;
        padding: 1px 2px;
        border-radius: 3px;
    }
    .risk-legend {
        display: flex;
        gap: 8px;
        margin-bottom: 8px;
        font-size: 11px;
        font-weight: 600;
    }
    .risk-legend span {
        padding: 3px 8px;
        border-radius: 6px;
    }
    .legend-high {
        color: #ff3b30;
        background-color: rgba(255, 59, 48, 0.12);
    }
    .legend-med {
        color: #ff9500;
        background-color: rgba(255, 149, 0, 0.12);
    }
    .legend-low {
        color: #34c759;
        background-color: rgba(52, 199, 89, 0.12);
    }
    
    /* Streamlit Tab 样式重塑为苹果扁平胶囊分段控制器 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #e5e5ea;
        padding: 4px;
        border-radius: 9px;
        border: none;
    }
    .stTabs [data-baseweb="tab"] {
        height: 34px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 7px;
        color: #86868b;
        font-weight: 500;
        font-size: 13px;
        border: none;
        padding: 0 18px;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #1d1d1f;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0071e3 !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
    }
    /* 隐藏默认下划线 */
    .stTabs [data-baseweb="tab-highlight-container"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Helper 函数：绘制苹果扁平卡片（带色边框强调）
def render_kpi_card(title: str, value: str, subtitle: str, border_color: str = "#d2d2d7", gradient_start: str = "#ffffff", gradient_end: str = "#f5f5f7"):
    """
    渲染运营看板顶部 KPI 卡片。
    """
    # 使用内联 HTML/CSS 绘制卡片，保证 Streamlit 多列布局下视觉一致
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {gradient_start} 0%, {gradient_end} 100%); padding: 20px; border-radius: 12px; border: 1px solid {border_color}; color: #1d1d1f; box-shadow: 0 4px 16px rgba(0,0,0,0.02);">
        <div style="font-size: 11px; color: #86868b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">{title}</div>
        <div style="font-size: 26px; font-weight: 700; margin: 6px 0; color: #1d1d1f;">{value}</div>
        <div style="font-size: 11px; color: #86868b; font-weight: 400;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 顶部主标题栏
# ==========================================
st.markdown(f'<div class="custom-title">🛡️ {html.escape(APP_NAME)}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="custom-subtitle">{html.escape(APP_SUBTITLE)}</div>', unsafe_allow_html=True)

# ==========================================
# 3. 多页面 Tab 标签导航设计 (解耦复杂界面)
# ==========================================
tab_audit, tab_consult, tab_dashboard, tab_library = st.tabs([
    "🛡️ 智能合规审计工作区", 
    "💬 法律咨询助手",
    "📊 运营数据看板大屏", 
    "📚 合规法律条款文库"
])

# ------------------------------------------
# TAB 1: 智能合规审计工作区
# ------------------------------------------
with tab_audit:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📂 上传劳动合同进行智能审计")
    
    # 极简上传区块
    uploaded_file = st.file_uploader(
        "支持拖拽或选择 Word (.docx) 或 PDF (.pdf) 格式的劳动合同文档进行本地脱敏解析", 
        type=["docx", "pdf"]
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 初始化审计结果和解析结果缓存，避免 Streamlit 每次按钮点击/rerun 都重新执行耗时的 MinerU 解析。
    if "audit_result" not in st.session_state:
        st.session_state["audit_result"] = None
    if "current_file" not in st.session_state:
        st.session_state["current_file"] = None
    if "parsed_contract" not in st.session_state:
        st.session_state["parsed_contract"] = None

    if uploaded_file is not None:
        # 读取上传文件字节，用于保存临时文件和计算轻量文件指纹
        uploaded_file_bytes = uploaded_file.getvalue()
        # 以“文件名 + 文件大小”作为轻量文件指纹；文件变化时清空旧报告和旧解析结果。
        uploaded_file_key = f"{uploaded_file.name}:{len(uploaded_file_bytes)}"
        if st.session_state["current_file"] != uploaded_file_key:
            # 用户上传新文件时，旧审查报告和旧解析缓存必须全部失效
            st.session_state["audit_result"] = None
            st.session_state["current_file"] = uploaded_file_key
            st.session_state["parsed_contract"] = None

        if st.session_state["parsed_contract"] is None:
            # MinerU CLI 只接受真实文件路径，因此先将上传内容落到 data/temp，再在 finally 中清理。
            temp_dir = os.path.join(os.path.dirname(__file__), "data", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_file.name)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file_bytes)

            parser_message = ""
            with st.spinner("正在使用本地增强解析能力提取并整理合同内容..."):
                try:
                    # 合同上传入口优先启用 MinerU；法律库解析仍由 retriever/agent 走 legacy，避免重型解析影响法规检索。
                    raw_text = extract_contract_text(temp_path, prefer_mineru=True)
                    parser_message = get_last_parser_message()
                    # 提取甲方、乙方、期限、薪资等轻量元数据用于前端展示和入库
                    metadata = extract_metadata(raw_text)
                    # 合同正文在发给大模型前先做本地脱敏
                    clean_text = desensitize_text(raw_text)
                    # 将解析结果缓存到 session_state，避免按钮点击触发重复解析
                    st.session_state["parsed_contract"] = {
                        "clean_text": clean_text,
                        "metadata": metadata,
                        "parser_message": parser_message
                    }
                except Exception as e:
                    st.error(f"提取合同失败: {e}")
                    clean_text = None
                finally:
                    # 临时上传文件使用后立即清理，避免 data/temp 累积用户合同
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        else:
            # 复用已解析结果，确保点击“开始会审”时不会再次触发本地文档解析。
            parsed_contract = st.session_state["parsed_contract"]
            clean_text = parsed_contract["clean_text"]
            metadata = parsed_contract["metadata"]
            parser_message = parsed_contract["parser_message"]

        if parser_message:
            # MinerU 回退属于可用但需提示的状态；成功增强解析则用普通信息提示
            if "回退" in parser_message or "未检测到" in parser_message:
                st.warning(parser_message)
            else:
                st.info(parser_message)
                
        if clean_text:
            # 双栏流式布局 (50% 对称排列)
            left_col, right_col = st.columns(2)
            
            with left_col:
                # 使用原生的 st.container(border=True) 作为卡片，杜绝 HTML 容器分裂导致的空卡片故障
                with st.container(border=True):
                    st.markdown("<h4 style='margin-top:0; color:#1d1d1f;'>📑 提取到的合同元数据</h4>", unsafe_allow_html=True)
                    
                    # 元数据 2x2 极简表格样式呈现
                    st.markdown(f"""
                    <table style='width:100%; border:none; color:#1d1d1f; font-size:14px; margin-bottom:15px;'>
                        <tr>
                            <td style='padding:8px 0; border:none; width:50%;'><span style='color:#86868b;'>甲方单位:</span> <strong style='color:#1d1d1f;'>{metadata['party_a']}</strong></td>
                            <td style='padding:8px 0; border:none; width:50%;'><span style='color:#86868b;'>乙方姓名:</span> <strong style='color:#1d1d1f;'>{metadata['party_b']}</strong></td>
                        </tr>
                        <tr>
                            <td style='padding:8px 0; border:none;'><span style='color:#86868b;'>合同期限:</span> <strong style='color:#1d1d1f;'>{metadata['duration']}</strong></td>
                            <td style='padding:8px 0; border:none;'><span style='color:#86868b;'>约定薪资:</span> <strong style='color:#1d1d1f;'>{metadata['salary']}</strong></td>
                        </tr>
                    </table>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<hr style='border: 0.5px solid #d2d2d7;'>", unsafe_allow_html=True)
                    st.markdown("<h4 style='color:#1d1d1f;'>📝 本地脱敏后的合同文本</h4>", unsafe_allow_html=True)
                    st.markdown(
                        "<div style='font-size:12px; color:#34c759; background-color:rgba(52,199,89,0.1); padding:8px 12px; border-radius:6px; border:1px solid rgba(52,199,89,0.25); margin-bottom:12px;'>"
                        "🛡️ [安全脱敏开启] 身份证号和联系电话已本地强制脱敏，不会传至任何第三方 API。"
                        "</div>", 
                        unsafe_allow_html=True
                    )
                    current_report = st.session_state["audit_result"]["report"] if st.session_state["audit_result"] is not None else ""
                    contract_preview_box = st.empty()
                    with contract_preview_box.container():
                        # 报告生成前只展示脱敏合同；报告生成后同一容器会重新渲染并叠加风险高亮。
                        render_contract_preview(clean_text, current_report)
                
            with right_col:
                with st.container(border=True):
                    st.markdown("<h4 style='margin-top:0; color:#1d1d1f;'>🤖 智能合规会审面板</h4>", unsafe_allow_html=True)
                    
                    # 苹果亮蓝高亮主行动按钮
                    st.markdown("""
                    <style>
                        div.stButton > button:first-child {
                            background-color: #0071e3 !important;
                            color: #ffffff !important;
                            border: none !important;
                            border-radius: 8px !important;
                            font-weight: 600 !important;
                            font-size: 14px !important;
                            height: 40px !important;
                            transition: background-color 0.2s ease, transform 0.1s ease !important;
                        }
                        div.stButton > button:first-child:hover {
                            background-color: #0077ed !important;
                            transform: translateY(-1px);
                        }
                        div.stButton > button:first-child:active {
                            transform: translateY(1px);
                        }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    run_audit = st.button("🚀 开始双智能体协同合规会审", use_container_width=True)
                    
                    if run_audit:
                        # 记录审查开始时间，用于最终写入平均耗时 KPI
                        audit_started_at = time.perf_counter()
                        status_box = st.empty()
                        
                        # 1. 运行检索匹配
                        status_box.markdown(
                            "<div style='color:#0071e3; font-size:13px; margin: 10px 0;'>🔍 [1/4] 正在从本地 FAISS 法律知识库召回相关劳动法条规约...</div>", 
                            unsafe_allow_html=True
                        )
                        db_dir = os.path.join(os.path.dirname(__file__), "data", "faiss_index")
                        try:
                            # 将完整的合同文本传入，底层机制已重构为段落切分、多路召回与合并去重
                            retrieved_laws = query_laws(clean_text, db_dir, top_k=10)
                        except Exception as ex:
                            # 检索失败不直接中断智能体流程，但会在报告中明确展示降级信息
                            retrieved_laws = f"【RAG 检索失败】未能从本地 FAISS 法律知识库召回可靠依据。错误信息：{ex}"
                            
                        # 2. 构造 LangGraph 双智能体状态输入
                        inputs = {
                            "contract_text": clean_text,
                            "retrieved_laws": retrieved_laws,
                            "raw_audit": "",
                            "feedback": "",
                            "final_report": "",
                            "loop_count": 0
                        }
                        
                        # 3. 调用多智能体流转引擎
                        status_box.markdown(
                            "<div style='color:#0071e3; font-size:13px; margin: 10px 0;'>🤖 [2/4] 启动 Auditor 初审智能体对条款进行法条比对与评估...</div>", 
                            unsafe_allow_html=True
                        )
                        app = build_agent_graph()
                        
                        try:
                            # LangGraph 执行 Auditor -> Critic -> Router -> ReportGenerator 的状态流转
                            result = app.invoke(inputs)
                            
                            status_box.markdown(
                                "<div style='color:#0071e3; font-size:13px; margin: 10px 0;'>⚖️ [3/4] 启动 Critic 反思审计节点，正在进行风险核校并整理修正对策...</div>", 
                                unsafe_allow_html=True
                            )
                            
                            # 4. 解析风险统计并持久化入库
                            raw_audit_text = result.get("raw_audit", "")
                            # 入库统计以 raw_audit 为准，避免最终报告附加的补充法条章节干扰风险项计数。
                            risk_high_cnt = count_risk_items(raw_audit_text, "高风险")
                            risk_med_cnt = count_risk_items(raw_audit_text, "中风险")
                            duration_seconds = time.perf_counter() - audit_started_at
                            
                            insert_audit_log(
                                filename=uploaded_file.name,
                                party_a=metadata["party_a"],
                                party_b=metadata["party_b"],
                                risk_high=risk_high_cnt,
                                risk_med=risk_med_cnt,
                                duration_seconds=duration_seconds
                            )
                            
                            # 将结果保存至会话状态
                            st.session_state["audit_result"] = {
                                "report": result["final_report"],
                                "filename": uploaded_file.name
                            }
                            # 报告生成后刷新左侧合同预览，使风险原文高亮立即生效
                            contract_preview_box.empty()
                            with contract_preview_box.container():
                                render_contract_preview(clean_text, result["final_report"])
                            
                            status_box.empty()
                            
                        except Exception as ex:
                            st.error(f"双智能体调用失败，请检查网络或大模型 API 密钥。错误: {ex}")
                            
                    # 判断会话状态中是否有已生成的审计报告，若有则持续渲染（避免由于 Streamlit 重新运行导致的下载中断）
                    if st.session_state["audit_result"] is not None:
                        # Streamlit 每次交互都会 rerun，因此报告必须从 session_state 持续渲染
                        report_data = st.session_state["audit_result"]
                        st.markdown("<div style='color:#34c759; font-size:13px; font-weight:600; margin: 10px 0;'>🎉 审查完成！以下为联合生成的《合规智能会审报告》：</div>", unsafe_allow_html=True)
                        st.markdown("### 📄 智能合规审查报告")
                        st.markdown(report_data["report"])
                        
                        # 5. 导出下载按钮（使用基于 Base64 的原生 HTML 下载链接，彻底绕过 Streamlit 下载拦截与 UUID 后缀 Bug）
                        st.markdown("---")
                        
                        # 恢复并组合包含原合同名称的高辨识度中文文件名
                        export_filename = f"劳动合同合规审查报告_{report_data['filename'].split('.')[0]}.md"
                        
                        # 将 Markdown 报告字符串转为基于 utf-8 的 Base64 数据流
                        b64_report = base64.b64encode(report_data["report"].encode("utf-8")).decode()
                        
                        # 构造纯前端下载锚点标签，利用浏览器原生下载属性 (download) 实现极速导出，UI 与 Streamlit 次级按钮保持 100% 视觉一致
                        download_href = f'''
                        <a href="data:text/markdown;charset=utf-8;base64,{b64_report}" download="{export_filename}"
                           style="display: block; width: 100%; text-align: center; background-color: #ffffff; 
                                  color: #1d1d1f; border: 1px solid #d2d2d7; padding: 10px 0; 
                                  border-radius: 8px; text-decoration: none; font-weight: 600; 
                                  font-size: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);">
                           💾 一键导出合规审查报告 (Markdown格式)
                        </a>
                        '''
                        st.markdown(download_href, unsafe_allow_html=True)
                    else:
                        if not run_audit:
                            # 激活状态等待提示
                            st.markdown("""
                            <div style="text-align: center; padding: 40px 10px; color: #86868b;">
                                <div style="font-size: 24px; margin-bottom: 8px;">⚖️</div>
                                <div style="font-size: 13px;">点击上方按钮，启动双智能体反思协同会审</div>
                            </div>
                            """, unsafe_allow_html=True)
                
    else:
        # 重置审计会话状态
        st.session_state["audit_result"] = None
        st.session_state["current_file"] = None
        st.session_state["parsed_contract"] = None
        
        # 清爽优雅的空置首屏引导区 (Onboard State)
        st.markdown("""
        <div style="text-align: center; padding: 70px 20px; color: #86868b; background-color: #ffffff; border-radius: 12px; border: 1px dashed #d2d2d7;">
            <div style="font-size: 52px; margin-bottom: 18px;">🛡️</div>
            <div style="font-size: 16px; font-weight: 600; color: #1d1d1f; margin-bottom: 8px;">准备好开始合规审查了吗？</div>
            <div style="font-size: 13px; max-width: 460px; margin: 0 auto 20px auto; line-height: 1.45;">
                请在上方拖入或选择一份需要审计的劳动合同。系统会利用本地正则表达式对身份证号与电话号码进行前置安全占位脱敏，保护数据不泄露，然后结合劳动法向量文库召回法律参考，协助 Auditor 与 Critic 进行专业审计。
            </div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------
# TAB 2: 法律咨询助手
# ------------------------------------------
with tab_consult:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💬 劳动法律咨询助手")
    st.markdown(
        "<div style='font-size:13px; color:#86868b; margin-bottom:16px;'>"
        "适用于劳动合同、试用期、社保、加班工资、竞业限制、培训服务期等简单咨询。"
        "系统会先由 AI 预判可能法条，再回到本地 RAG 法条库检索，最后基于本地法条流式生成正式答复。"
        "</div>",
        unsafe_allow_html=True
    )

    consult_question = st.chat_input("请输入你的劳动法律问题，例如：试用期六个月合法吗？")

    # 法律咨询采用“单问单答”模式，只保留当前问题、当前答案和本次法条召回过程
    if "consult_current_question" not in st.session_state:
        st.session_state["consult_current_question"] = ""
    if "consult_current_answer" not in st.session_state:
        st.session_state["consult_current_answer"] = ""
    if "consult_current_first_answer" not in st.session_state:
        st.session_state["consult_current_first_answer"] = ""
    if "consult_current_local_laws" not in st.session_state:
        st.session_state["consult_current_local_laws"] = ""

    if consult_question:
        # 提交新问题时清空上一轮答案，避免新旧问题内容混杂
        st.session_state["consult_current_question"] = consult_question
        st.session_state["consult_current_answer"] = ""
        st.session_state["consult_current_first_answer"] = ""
        st.session_state["consult_current_local_laws"] = ""

    if not consult_question and st.session_state["consult_current_question"] and st.session_state["consult_current_answer"]:
        # 没有新输入时，复显上一轮咨询结果，避免 Streamlit rerun 后聊天记录消失
        with st.chat_message("user"):
            st.markdown(st.session_state["consult_current_question"])
        with st.chat_message("assistant"):
            st.markdown(st.session_state["consult_current_answer"])
            with st.expander("查看本次咨询的预检结果与本地法条召回", expanded=False):
                st.markdown("#### 第一轮预检回答")
                st.markdown(st.session_state["consult_current_first_answer"])
                st.markdown("#### 本地法条库召回")
                st.markdown(f"```text\n{st.session_state['consult_current_local_laws']}\n```")

    if consult_question:
        # 先展示用户问题，再进入两轮咨询流程
        with st.chat_message("user"):
            st.markdown(consult_question)

        try:
            with st.status("正在进行两轮法律咨询推理...", expanded=True) as status:
                st.write("1/3 正在由预检智能体识别问题和可能相关法条...")
                # 第一轮让模型提出候选法条，便于后续本地精确查找
                first_answer = _ask_legal_consultant_first_round(consult_question)
                st.session_state["consult_current_first_answer"] = first_answer

                st.write("2/3 正在按法名和条号精确查找本地法条，必要时回退到 FAISS 语义检索...")
                db_dir = os.path.join(os.path.dirname(__file__), "data", "faiss_index")
                # 第二步回到本地法条库检索，避免最终答复只依赖模型记忆
                local_laws = _lookup_local_laws_for_consultation(first_answer, consult_question, db_dir)
                st.session_state["consult_current_local_laws"] = local_laws

                st.write("3/3 正在基于本地法条生成正式答复...")
                status.update(label="本地法条已召回，开始流式生成正式答复", state="complete", expanded=False)

            with st.chat_message("assistant"):
                # 第三步基于本地法条流式生成最终答案
                final_answer = st.write_stream(
                    _stream_legal_consultant_final_answer(
                        consult_question,
                        first_answer,
                        local_laws
                    )
                )
                with st.expander("查看本次咨询的预检结果与本地法条召回", expanded=False):
                    # 展开区用于教学演示：展示模型预检与本地 RAG 召回如何配合
                    st.markdown("#### 第一轮预检回答")
                    st.markdown(first_answer)
                    st.markdown("#### 本地法条库召回")
                    st.markdown(f"```text\n{local_laws}\n```")

            # 保存最终答案，供下一次 Streamlit rerun 后继续展示
            st.session_state["consult_current_answer"] = final_answer
        except Exception as ex:
            st.error(f"法律咨询助手调用失败，请检查大模型 API Key、本地 FAISS 索引或网络配置。错误: {ex}")

# ------------------------------------------
# TAB 3: 运营数据看板大屏
# ------------------------------------------
with tab_dashboard:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 时间范围筛选器
    st.markdown("### 📅 时间范围筛选", unsafe_allow_html=True)
    col_filter, col_refresh = st.columns([3, 1])
    with col_filter:
        time_range = st.selectbox(
            "选择数据统计时间范围",
            ["全部时间", "最近7天", "最近30天"],
            label_visibility="collapsed",
            horizontal=True
        )
    with col_refresh:
        if st.button("🔄 刷新数据", use_container_width=True):
            st.rerun()
    
    # 动态抓取库中最新统计
    kpis = get_kpi_metrics()
    
    # KPI 排布 (利用有色细线对不同卡片进行极简强调，规避大彩块)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card(
            "累计已审查合同", 
            f"{kpis['total_audits']} 份", 
            "SQLite 本地物理库统计数", 
            "#d2d2d7", # 银灰
            "#ffffff", "#f5f5f7"
        )
    with col2:
        render_kpi_card(
            "高风险合同占比", 
            kpis['high_risk_ratio'], 
            "含有 1 项以上高风险项合同", 
            "rgba(255, 59, 48, 0.4)", # 高亮淡红
            "#fff5f5", "#ffe5e5"
        )
    with col3:
        render_kpi_card(
            "平均审查速度", 
            kpis['average_duration'], 
            "双智能体流转反思耗时均值", 
            "rgba(0, 113, 227, 0.4)", # 高亮淡蓝
            "#f0f7ff", "#e0efff"
        )
    with col4:
        render_kpi_card(
            "今日新增上传", 
            f"{kpis['today_uploads']} 份", 
            "按本地审查完成时间统计", 
            "rgba(52, 199, 89, 0.4)", # 高亮淡绿
            "#f0fff5", "#e0ffe5"
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 图表与活动流并排排版
    chart_col, flow_col = st.columns([1.1, 1])
    
    # 中文字体防乱码设置
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    
    with chart_col:
        with st.container(border=True):
            st.markdown("<h4 style='margin-top:0; color:#1d1d1f;'>📈 历史审查风险分布统计</h4>", unsafe_allow_html=True)
            
            # 抓取图表统计数据
            filenames, highs, meds = get_monthly_risk_stats()
            
            if filenames:
                # 苹果风格亮色主题图表绘制
                fig, ax = plt.subplots(figsize=(7, 3.2), facecolor='#ffffff')
                ax.set_facecolor('#ffffff')
                
                # 使用等距横坐标绘制每份合同的高/中风险柱状对比
                x = np.arange(len(filenames))
                width = 0.35
                
                # 高风险红，中风险橙
                ax.bar(x - width/2, highs, width, label='高风险', color='#ff3b30', edgecolor='none')
                ax.bar(x + width/2, meds, width, label='中风险', color='#ff9500', edgecolor='none')
                
                ax.set_ylabel('风险条款数', color='#86868b', fontsize=8)
                ax.set_title('近期上传合同风险项对比图', color='#1d1d1f', fontsize=9, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(filenames, rotation=12, color='#86868b', fontsize=7)
                
                # 统一图例、刻度和轴线颜色，使 Matplotlib 图表融入亮色 UI
                ax.legend(facecolor='#f5f5f7', edgecolor='#d2d2d7', labelcolor='#1d1d1f', fontsize=7)
                ax.tick_params(colors='#86868b', labelsize=8)
                
                # 精简轴线
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#d2d2d7')
                ax.spines['bottom'].set_color('#d2d2d7')
                
                # 将 Matplotlib 图表嵌入 Streamlit 页面
                st.pyplot(fig)
                
                # 添加图表下载按钮
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    label="📥 下载图表",
                    data=buf,
                    file_name="风险分布统计.png",
                    mime="image/png",
                    use_container_width=True
                )
                plt.close(fig)
            else:
                st.info("数据为空，待上传审核合同生成数据图表。")
        
    with flow_col:
        with st.container(border=True):
            st.markdown("<h4 style='margin-top:0; color:#1d1d1f;'>📋 最新合同审查活动流列表</h4>", unsafe_allow_html=True)
            
            activities = get_recent_activities(limit=5)
            if activities:
                # 使用 st.dataframe 替代 HTML 表格，支持排序功能
                df = pd.DataFrame(activities)
                df = df[['filename', 'party_a', 'party_b', 'risk_high', 'risk_med', 'duration_seconds', 'created_at']]
                df.columns = ['合同名称', '甲方单位', '乙方姓名', '高风险', '中风险', '耗时(秒)', '审查时间']
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("尚无审查流水记录。")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 第三行：甲方单位统计表格
    with st.container(border=True):
        st.markdown("<h4 style='margin-top:0; color:#1d1d1f;'>🏢 甲方单位审查统计</h4>", unsafe_allow_html=True)
        
        party_stats = get_party_a_statistics(limit=10)
        if party_stats:
            # 使用 st.dataframe 替代 HTML 表格，支持排序功能
            df = pd.DataFrame(party_stats)
            df['风险率'] = df.apply(lambda row: round(((row['high_risks'] + row['med_risks']) / row['count']) * 100, 1) if row['count'] > 0 else 0, axis=1)
            df = df[['party_a', 'count', 'high_risks', 'med_risks', '风险率']]
            df.columns = ['甲方单位', '审查次数', '高风险项', '中风险项', '风险率(%)']
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("暂无甲方单位统计数据。")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 第四行：审查效率排行榜
    with st.container(border=True):
        st.markdown("<h4 style='margin-top:0; color:#1d1d1f;'>⚡ 审查效率排行榜</h4>", unsafe_allow_html=True)
        
        activities = get_recent_activities(limit=20)
        if activities:
            # 按耗时排序，取最快和最慢的各5条
            sorted_activities = sorted(activities, key=lambda x: x['duration_seconds'] or 0)
            fastest = sorted_activities[:5]
            slowest = sorted_activities[-5:][::-1]  # 最慢的5条，按耗时降序
            
            col_fast, col_slow = st.columns(2)
            
            with col_fast:
                st.markdown("<h5 style='color:#34c759; margin-top:0;'>🚀 最快审查（Top 5）</h5>", unsafe_allow_html=True)
                # 使用 st.dataframe 替代 HTML 表格，支持排序功能
                df_fast = pd.DataFrame(fastest)
                df_fast = df_fast[['filename', 'duration_seconds', 'risk_high', 'risk_med']]
                df_fast.columns = ['合同名称', '耗时(秒)', '高风险', '中风险']
                st.dataframe(df_fast, use_container_width=True, hide_index=True)
            
            with col_slow:
                st.markdown("<h5 style='color:#ff3b30; margin-top:0;'>🐌 最慢审查（Top 5）</h5>", unsafe_allow_html=True)
                # 使用 st.dataframe 替代 HTML 表格，支持排序功能
                df_slow = pd.DataFrame(slowest)
                df_slow = df_slow[['filename', 'duration_seconds', 'risk_high', 'risk_med']]
                df_slow.columns = ['合同名称', '耗时(秒)', '高风险', '中风险']
                st.dataframe(df_slow, use_container_width=True, hide_index=True)
        else:
            st.info("暂无审查效率数据。")

# ------------------------------------------
# TAB 4: 合规法律条款文库
# ------------------------------------------
with tab_library:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 扫描 laws 目录
    laws_dir = os.path.join(os.path.dirname(__file__), "data", "laws")
    if os.path.exists(laws_dir):
        # 只展示当前系统支持解析并可进入 RAG 的法规文件类型
        law_files = [f for f in os.listdir(laws_dir) if f.endswith((".docx", ".pdf", ".txt"))]
        if law_files:
            with st.container(border=True):
                st.markdown("<h5 style='margin-top:0; color:#1d1d1f;'>📂 已成功加载的法律与司法解释文本（本地 RAG 数据源）</h5>", unsafe_allow_html=True)
                for name in law_files:
                    st.markdown(f"- 📄 `{name}`")
        else:
            st.warning("法律法规目录为空，请将参考法条放入 data/laws/ 目录。")
            
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔍 法律法规语义检索比对器")
    st.markdown(
        "<div style='font-size:13px; color:#86868b; margin-bottom:16px;'>"
        "输入您关心的合同约定描述（例如：试用期约定六个月、离职不予退还保证金、拒绝缴纳社保等），"
        f"检索器将利用当前配置的 `{html.escape(os.getenv('EMBEDDING_MODEL_NAME', '未配置'))}` 语义向量模型对 FAISS 知识库进行余弦距离度量并召回最为匹配的背景法条。"
        "</div>", 
        unsafe_allow_html=True
    )
    
    # 检索查询输入框
    law_query = st.text_input(
        "输入合规匹配短语进行检索验证...", 
        placeholder="例如：本合同试用期为三个月，期间公司不为员工买社保"
    )
    
    if law_query:
        db_dir = os.path.join(os.path.dirname(__file__), "data", "faiss_index")
        with st.spinner("正在语义计算并召回相关法条..."):
            try:
                # 法条文库页用于人工验证 RAG 命中效果，因此固定展示 Top 5 结果
                results = query_laws(law_query, db_dir, top_k=5)
                st.markdown("##### ⚖️ RAG 检索命中的最相关背景法条（Top 5）")
                # 放在带有苹果细线边框的区域
                st.markdown(f"""
                <div style='background-color:#ffffff; padding:16px; border-radius:8px; border:1px solid #d2d2d7; white-space:pre-wrap; color:#1d1d1f; font-size:13px; line-height:1.6;'>
{results}
                </div>
                """, unsafe_allow_html=True)
            except Exception as ex:
                st.error(f"检索失败，请确认是否构建了本地 FAISS 索引。错误: {ex}")

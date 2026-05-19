# -*- coding: utf-8 -*-
"""
模块名: src/agent.py
作用: 基于 LangGraph 的 Auditor-Critic 双智能体协同审查状态机逻辑。
      LLM 调用通过 DeepSeek 官方 API 驱动，且法条匹配通过本地 RAG 向量检索进行。
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

MAX_REFLECTION_ROUNDS = 5

def _clean_report_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^(好的|当然|以下是|作为专业的中国劳动法务审查专家智能体|我将依据|我已仔细阅读)[^\n]*\n+", "", cleaned)
    cleaned = re.sub(r"(?m)^根据您提供的【?参考的劳动法律法规条款依据】?.*?报告如下：\s*$", "", cleaned)
    cleaned = re.sub(r"(?m)^好的，.*$", "", cleaned)
    return cleaned.strip()

def _normalize_law_name(name: str) -> str:
    for keyword in ["劳动合同法", "劳动法", "工伤保险条例", "女职工劳动保护特别规定"]:
        if keyword in name:
            return keyword
    return re.sub(r"中华人民共和国|_\d+|\.docx|\.pdf|\s", "", name)

def _extract_law_refs(text: str) -> list[tuple[str, str]]:
    return [(_normalize_law_name(name), article) for name, article in re.findall(r"《([^》]+)》第([一二三四五六七八九十百零〇\d]+)条", text)]

def _retrieved_laws_contain_ref(retrieved_laws: str, law_name: str, article: str) -> bool:
    normalized_law = _normalize_law_name(law_name)
    for match in re.finditer(rf"第{article}条", retrieved_laws):
        context = retrieved_laws[max(0, match.start() - 160):match.end() + 160]
        if normalized_law in context:
            return True
    return False

def _split_risk_blocks(raw_audit: str) -> list[tuple[str, str, str]]:
    matches = list(re.finditer(r"(?m)^#{1,6}\s*【(高风险|中风险|低风险)】[^\n]*", raw_audit))
    blocks = []
    for idx, match in enumerate(matches):
        next_risk_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_audit)
        next_section = re.search(r"(?m)^#{1,3}\s+", raw_audit[match.end():next_risk_start])
        end = match.end() + next_section.start() if next_section else next_risk_start
        blocks.append((match.group(1), match.group(0), raw_audit[match.start():end]))
    return blocks

def _find_missing_supplemental_refs(raw_audit: str, retrieved_laws: str) -> list[str]:
    missing_supplements = []
    for _, _, block in _split_risk_blocks(raw_audit):
        supplemental_refs = set(_extract_law_refs("\n".join(line for line in block.splitlines() if "补充法条依据" in line)))
        for law_name, article in _extract_law_refs(block):
            if (law_name, article) in supplemental_refs:
                continue
            formatted_ref = f"《{law_name}》第{article}条"
            if not _retrieved_laws_contain_ref(retrieved_laws, law_name, article) and formatted_ref not in missing_supplements:
                missing_supplements.append(formatted_ref)
    return missing_supplements

def _format_supplemental_refs_section(raw_audit: str, retrieved_laws: str) -> str:
    missing_supplements = _find_missing_supplemental_refs(raw_audit, retrieved_laws)
    if not missing_supplements:
        return ""
    lines = ["## 四、 补充法条标注说明", "以下法条未出现在本次 RAG 检索参考依据中，报告正文如未逐项标明“补充法条依据”，统一按补充法条依据处理："]
    lines.extend(f"- {ref}（补充法条依据）" for ref in missing_supplements[:12])
    return "\n".join(lines)

def _extract_declared_risk_counts(raw_audit: str) -> dict[str, int]:
    preface = raw_audit.split("#### 【", 1)[0]
    counts = {}
    for level in ["高风险", "中风险", "低风险"]:
        match = re.search(rf"{level}(?:项|违规|优化建议)?\s*[:：]?\s*(\d+)\s*项|(\d+)\s*项{level}", preface)
        if match:
            counts[level] = int(match.group(1) or match.group(2))
    return counts

def _feedback_is_only_supplemental_marking(feedback: str) -> bool:
    if "【未通过审核】" not in feedback:
        return False
    if "补充法条依据" not in feedback and "未出现在" not in feedback:
        return False
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

def _detect_audit_quality_issues(raw_audit: str, retrieved_laws: str, contract_text: str = "") -> list[str]:
    issues = []
    risk_blocks = _split_risk_blocks(raw_audit)
    declared_counts = _extract_declared_risk_counts(raw_audit)
    actual_counts = {
        "高风险": sum(1 for level, _, _ in risk_blocks if level == "高风险"),
        "中风险": sum(1 for level, _, _ in risk_blocks if level == "中风险"),
        "低风险": sum(1 for level, _, _ in risk_blocks if level == "低风险"),
    }
    mismatched_counts = [f"{level}声明{declared_counts[level]}项、实际{actual_counts[level]}项" for level in declared_counts if declared_counts[level] != actual_counts[level]]
    if mismatched_counts:
        issues.append("整体结论中的风险数量必须与下方具体风险项数量一致：" + "；".join(mismatched_counts))
    if re.search(r"一年.{0,12}试用期.{0,12}上限.{0,12}一个月|一年期限合同试用期上限为一个月", raw_audit):
        issues.append("一年期劳动合同试用期上限应为二个月，不是一个月。")
    if "其他法律风险摘要" in raw_audit or "不在上述风险评级" in raw_audit:
        issues.append("不得将实质性违法点放入不计入评级的其他摘要栏目。")
    if re.search(r"最低工资标准.{0,30}假设|假设.{0,30}最低工资标准|最低工资标准.{0,20}\d+\s*元", raw_audit):
        issues.append("不得使用未核实的最低工资标准具体金额作为风险依据；除非该数值来自检索依据，否则应表述为“以当地最新公布标准为准”。")

    for level, title, block in risk_blocks:
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
    # 实例化连接 DeepSeek 官方 API 的客户端
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL")
    )
    
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
1. 优先依据【参考的劳动法律法规条款依据】逐条核对合同条款。
2. 如果某个结论使用了未出现在参考依据中的法律条文，建议在该风险项中紧跟该法条标明“补充法条依据”；如正文漏标，系统会在最终报告末尾统一追加补充法条标注说明，不需要仅因此整篇返工。
3. 将审查出的隐患明确分为：【高风险】、【中风险】、【低风险】三档；没有明确违法事实的，不得强行判定风险。
4. 对每个违规点，必须精确引用对应的法条序号（如：《劳动合同法》第十九条）。
5. 针对每一个违规点，给出具体的、符合法律要求的“建议修改后条款”。
6. 如果合同整体合规，应明确输出“未发现高风险/中风险违规项”，只列出必要的低风险优化建议。
7. 试用期上限必须严格按《劳动合同法》第十九条判断：三个月以上不满一年不得超过一个月；一年以上不满三年不得超过二个月；三年以上固定期限和无固定期限不得超过六个月。因此，一年期劳动合同试用期上限是二个月，不是一个月；三年固定期限劳动合同约定六个月试用期不得仅因“三年整”表述判定为风险。
8. 合同已明确约定某项法定内容时，不得仅因表达不够详细就升格为中风险；可作为低风险优化建议。
9. 高风险仅限于明显违反强制性法律规定、剥夺劳动者核心权利或可能导致条款无效/行政处罚/重大赔偿的情形。
10. 中风险必须同时满足：合同存在明确不利于劳动者的具体安排，且该安排可能导致用人单位承担较明确的败诉、赔偿或行政责任。
11. 下列情形默认只能列为【低风险优化建议】，除非合同另有明显违法表述：法定代表人信息空缺但甲方名称住所清楚；休息休假条款重复；保密范围不够细；试用期工资已高于80%但未完整复述“同岗位最低档工资”；工作地点含业务出差区域但未授权甲方单方永久变更常驻城市。
12. 销售、技术、HR等岗位如因业务需要存在短期出差、客户拜访、项目现场支持，不得直接认定为中风险；只有出现“甲方可单方调整工作地点/常驻城市且劳动者必须服从”等重大变更授权时，才可评为中风险。
13. 社会保险缴纳、加班工资、工资按月足额支付、违法违约金、任意解除、工伤责任转嫁、女职工特殊保护、试用期上限等强制性义务，如合同存在明确违反表述，必须列入正式【高风险】或【中风险】风险项，不得放入“不计入评级”“其他摘要”“补充提示”等非正式栏目。
14. 以下明确违法表述原则上应判为【高风险】：以补贴替代或放弃社会保险；免除加班工资；工资可跨月或跨季度延期支付；要求劳动者承担普通离职违约金；将工伤、疾病或职业伤害责任转嫁给劳动者；约定孕期、产期、哺乳期可解除或降低待遇；赋予甲方无法定事由的任意解除权。
15. 单方调整工作地点、跨城市调动且劳动者不得拒绝，通常评为【中风险】；只有同时绑定违法解除、降薪、违约金、处罚或严重剥夺劳动条件时，才可升为【高风险】。
16. 同一违法事实不得重复计入多个风险等级；例如“工伤责任转嫁”已经作为高风险评价时，不得再作为中风险重复列项。
17. 工时和加班风险项必须引用《劳动法》第三十六条、第四十一条、第四十四条等对应依据；不得仅引用工资支付条款替代工时限制依据。
18. 餐饮门店合同中，要求劳动者自行承担防滑鞋、手套、口罩等劳动防护用品，以及健康证、体检费、岗前培训费的，应列为正式风险项或优化建议，不得遗漏。
19. 不得写死地方最低工资标准具体金额，除非该金额明确来自【参考的劳动法律法规条款依据】；否则统一表述为“不得低于用人单位所在地最低工资标准（以当地最新公布标准为准）”。
20. 禁止输出“其他法律风险摘要（不在上述风险评级中）”之类栏目；所有实质性违法点必须纳入风险项总数。
21. 如果上一轮 Critic 给出修正指令，必须逐条落实，不得重复原有问题。

【输出格式】:
- 直接从“### 一、整体合规性结论”开始。
- 不要重复外层标题“劳动合同合规智能审查报告”。
- 不要输出寒暄、身份说明或过程说明。
- 每个具体风险项标题必须使用如下格式之一，方便系统统计：
  - `#### 【高风险】风险项1：...`
  - `#### 【中风险】风险项1：...`
  - `#### 【低风险】优化建议1：...`
- 整体结论中的风险数量必须与下方具体风险项数量一致。
"""

    # 调用大语言模型进行初审
    response = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1  # 采用低温度系数确保法理审查的严谨性与确定性
    )
    
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
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL")
    )
    
    prompt = f"""
你是一个资深的中国劳动合同法务总监与反思审计智能体（Critic）。请对下属 Auditor 提交的劳动合同审查草稿进行双重反思审计。

你的职责是做劳动合同专项复核，而不是通用合同复核。程序质量闸门只负责格式、数量一致性、补充法条尾注和少数机器可判定底线；劳动法实质判断由你负责。

【Auditor 初步审查草稿】:
{state['raw_audit']}

【参考的法律依据】:
{state['retrieved_laws']}

【劳动合同专项复核评分表】:
1. 事实覆盖完整性：是否覆盖劳动合同期限、试用期、工作内容与地点、工时休假、劳动报酬、社会保险、劳动保护、解除终止、违约责任、女职工保护、工伤责任等劳动合同核心模块。
2. 强制性义务识别：是否准确识别放弃或替代社保、免除加班费、工资跨月压付或无故拖欠、普通离职违约金、任意解除、工伤责任转嫁、三期女职工解除或降待遇、违法试用期等劳动法强制性问题。
3. 风险等级合理性：高风险用于明显违反强制性规定、剥夺劳动者核心权利或可能导致行政处罚/重大赔偿/条款无效的问题；中风险用于明确不利安排且有较明确败诉或赔偿风险的问题；低风险仅用于表述优化、证据留痕、程序完善等非核心违法问题。不得把工资拖欠、社保替代、免除加班费、工伤责任转嫁、三期解除、违法违约金、任意解除等降为低风险。
4. 法条准确性：引用法条必须真实、方向正确；对真实存在但未出现在参考依据中的法条，如正文未标注“补充法条依据”，不需要仅因此要求返工，系统会在最终报告末尾统一追加补充法条标注说明。
5. 修改建议可执行性：建议条款应能直接替换合同条款，不得引入新的违法点；涉及最低工资、地方标准等动态事项时，不得编造、假设或写死具体数值。除非该数值明确来自参考依据，否则应表述为“以当地最新公布标准为准”。
6. 结构与可读性：每个风险项应包含违规条款/审查依据、法律分析或风险分析、依据、建议修改后条款；风险总数应与具体条目一致；不得输出寒暄、身份声明或“其他法律风险摘要（不计入评级）”。
7. 重复与拆分合理性：同一违法事实不得重复计入多个风险等级；但不同法益的问题可以拆分，例如“免除加班费”和“超长工时”可以分项，也可以合并，只要总数和分析自洽。
8. 劳动合同场景适配：餐饮门店劳动合同应特别关注长工时、节假日排班、后厨/传菜/清洁劳动保护、健康证和体检费用、工伤事故、工资压付、罚款扣款、社保替代、女职工特殊保护等高频劳动争议点。

【输出规范】：
- 只有在风险等级、法条引用、输出格式、修改建议均合格时，才可在回复最开头直接输出：“【通过审核】”，无需输出其他修改意见；仅存在补充法条依据漏标时，也视为可通过。
- 如果你发现其中有任何不严谨、法条引用有误、或修改意见不妥之处，请详细写下具体的“修正指令”，以便 Auditor 重新校对。
- 如果未通过，请优先指出劳动合同实质问题，不要只围绕格式措辞；修正指令应能指导 Auditor 直接改写风险项。
"""

    response = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2  # 稍微允许一些发散以促进更深入的潜在风险反思
    )
    
    feedback = response.choices[0].message.content
    deterministic_issues = _detect_audit_quality_issues(state["raw_audit"], state["retrieved_laws"], state["contract_text"])
    if deterministic_issues:
        feedback = "【未通过审核】\n" + "\n".join(f"{idx + 1}. {issue}" for idx, issue in enumerate(deterministic_issues)) + "\n" + feedback.replace("【通过审核】", "")
    elif _feedback_is_only_supplemental_marking(feedback):
        feedback = "【通过审核】\n仅存在补充法条依据漏标，最终报告将自动追加补充法条标注说明。"

    return {
        "feedback": feedback
    }

# 4. 节点3: ReportGenerator (最终报告润色组装节点)
def report_generator_node(state: AgentState) -> Dict[str, Any]:
    """
    接收最终通过的草稿，剔除中间调试及反思标记，格式化为最终版报告。
    """
    revision_rounds = max(state['loop_count'] - 1, 0)
    unresolved_feedback = state.get("feedback", "") if "【未通过审核】" in state.get("feedback", "") else ""
    review_status = "双智能体（Auditor & Critic）协同会审完毕，报告已完成反思审计"
    supplemental_refs_section = _format_supplemental_refs_section(state["raw_audit"], state["retrieved_laws"])
    extra_sections = []
    if supplemental_refs_section:
        extra_sections.append(supplemental_refs_section)
    if unresolved_feedback:
        review_status = "已达到最大反思轮数，仍存在未完全解决的质检问题，建议人工复核后使用"
        unresolved_heading = "## 五、 未解决质检提示" if supplemental_refs_section else "## 四、 未解决质检提示"
        extra_sections.append(f"{unresolved_heading}\n{unresolved_feedback}")
    extra_sections_text = "\n\n".join(extra_sections)
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
    return {"final_report": final_output}

# 5. 条件路由逻辑 (Router)
def check_approval_router(state: AgentState) -> str:
    """
    控制流程分支：判断是进入下一次修改还是输出报告。
    """
    feedback = state.get("feedback", "")
    loop_count = state.get("loop_count", 0)
    
    # 条件1：Critic 判定通过审核
    # 条件2：触发最大反思循环次数限制（防止多智能体进入无限死锁循环）
    if "【通过审核】" in feedback or loop_count >= MAX_REFLECTION_ROUNDS:
        print(f"[Agent Workflow] 状态流转完成。通过原因: {'Critic 审核通过' if '【通过审核】' in feedback else '达到最大反思轮数限制'}。即将输出最终报告。")
        return "generate_report"
    else:
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

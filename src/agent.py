# -*- coding: utf-8 -*-
"""
模块名: src/agent.py
作用: 基于 LangGraph 的 Auditor-Critic 双智能体协同审查状态机逻辑。
      LLM 调用通过 DeepSeek 官方 API 驱动，且法条匹配通过本地 RAG 向量检索进行。
"""

import os
import sys
from typing import TypedDict, Dict, Any
from openai import OpenAI
from langgraph.graph import StateGraph, END

# 将当前目录加入查找路径，确保正确引入 retriever 模块
sys.path.append(os.path.dirname(__file__))
from retriever import query_laws

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
            # 以合同原文作为 Query 检索最相关的 3 条劳动法规
            retrieved_laws = query_laws(state["contract_text"], db_dir, top_k=3)
        except Exception as e:
            # 容错降级：RAG 检索失败时，给出基础提示，防止流程中断
            retrieved_laws = f"【RAG 检索失败，已降级】: {e}"
            
    # 初审提示词模版
    prompt = f"""
你是一个专业的中国劳动法务审查专家智能体（Auditor）。请仔细阅读以下信息：

【参考的劳动法律法规条款依据】:
{retrieved_laws}

【待审查的合同条款原文】:
{state['contract_text']}

【历史审查建议与反思反馈 (若有)】:
{state.get('feedback', '暂无。这是第一轮初审，请输出全面细致的审查草稿。')}

【审查要求】:
1. 逐条核对合同条款是否违反上述法条依据（例如：未依法缴纳社保、试用期超出法定上限、违法约定劳动者违约金等）。
2. 将审查出的隐患明确分为：【高风险】、【中风险】、【低风险】三档。
3. 对每个违规点，必须精确引用对应的法条序号（如：《劳动合同法》第十九条）。
4. 针对每一个违规点，给出具体的、符合法律要求的“建议修改后条款”。
"""

    # 调用大语言模型进行初审
    response = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1  # 采用低温度系数确保法理审查的严谨性与确定性
    )
    
    return {
        "raw_audit": response.choices[0].message.content,
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
你是一个资深的劳动法务总监与反思审计智能体（Critic）。请对下属 Auditor 提交的审查草稿进行双重反思审计：

【Auditor 初步审查草稿】:
{state['raw_audit']}

【参考的法律依据】:
{state['retrieved_laws']}

【审计反思任务】:
1. 检查 Auditor 引用的法律法条是否真实存在，是否存在大模型幻觉捏造法条的行为？
2. 检查所设定的高/中/低风险评级是否过激或过宽泛，是否偏离司法实践？
3. 修改对策是否切实可行，是否反而增加了企业或劳动者的其他法律隐患？

【输出规范】：
- 如果你认为初步审查草稿【完全合格且无大模型幻觉】，请在回复的最开头直接输出：“【通过审核】”，无需输出其他修改意见。
- 如果你发现其中有任何不严谨、法条引用有误、或修改意见不妥之处，请详细写下具体的“修正指令”，以便 Auditor 重新校对。
"""

    response = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2  # 稍微允许一些发散以促进更深入的潜在风险反思
    )
    
    return {
        "feedback": response.choices[0].message.content
    }

# 4. 节点3: ReportGenerator (最终报告润色组装节点)
def report_generator_node(state: AgentState) -> Dict[str, Any]:
    """
    接收最终通过的草稿，剔除中间调试及反思标记，格式化为最终版报告。
    """
    final_output = f"""# 劳动合同合规智能审查报告

## 一、 智能审查元数据
- **审查状态**: 双智能体（Auditor & Critic）协同会审完毕并已通过合规校验
- **反思修正轮数**: {state['loop_count']} 轮

## 二、 检索到的法定背景参考依据
{state['retrieved_laws']}

## 三、 合规风险项明细与修正对策
{state['raw_audit']}
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
    # 条件2：触发最大反思循环次数限制（安全截断上限为 3 次，防止多智能体进入无限死锁循环）
    if "【通过审核】" in feedback or loop_count >= 3:
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

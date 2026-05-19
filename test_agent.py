# -*- coding: utf-8 -*-
"""
脚本名: test_agent.py
作用: 验证 LangGraph 双智能体状态机的流转、大模型 API 连接、以及合规报告的最终输出效果。
"""

import sys
import os
from dotenv import load_dotenv

# 1. 预加载全局环境变量
load_dotenv()

# 将 src 目录临时加入模块查找路径
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from agent import build_agent_graph

def run_test():
    print("=== [TEST] Start LangGraph Auditor-Critic multi-agent test ===")
    
    # 2. 模拟一份包含典型中国劳动法违规条款的测试劳动合同内容
    test_contract = """
    【第二章 试用期与社保条款】
    第八条：本劳动合同期限为一年（12个月），其中前三个月（90天）设定为新员工入职试用期。
    第九条：为了降低初创企业的行政成本与员工流失风险，试用期（前三个月）内公司不为员工缴纳社会保险。转正并签订正式补充协议后，公司再统一为员工办理社会保险缴纳手续。
    第十条：员工若在合同存续期内单方面提出离职，需向公司支付违约赔偿金 5000 元整，用以冲抵公司在招聘与前期岗前技能培训中产生的相关实际支出。
    """
    
    # 3. 构造状态机初始数据状态
    initial_state = {
        "contract_text": test_contract,
        "retrieved_laws": "",
        "raw_audit": "",
        "feedback": "",
        "final_report": "",
        "loop_count": 0
    }
    
    # 4. 构建并编译状态拓扑图
    try:
        app = build_agent_graph()
        print("[OK] LangGraph agent workflow compiled successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to compile LangGraph workflow: {e}")
        return
        
    # 5. 调用大语言模型，运行流转流程
    print("\n[Agent] Invoking agent workflow (this will take a few seconds to run Auditor & Critic)...")
    try:
        # 执行图运行，生成包含所有节点中间输出的最终状态字典
        final_state = app.invoke(initial_state)
        
        print("\n==================================================")
        print("[OK] Workflow execution completed successfully!")
        print("==================================================")
        
        # 6. 打印最终生成的审查报告
        final_report = final_state.get("final_report", "未生成最终报告。")
        print("\n--- 最终生成的审查报告如下 ---")
        # 替换换行输出，避免终端编码兼容报错
        print(final_report.encode('gbk', errors='ignore').decode('gbk'))
        
    except Exception as e:
        print(f"[ERROR] Agent run crashed: {e}")
        
    print("\n=== [TEST] Multi-agent system verification complete! ===")

if __name__ == "__main__":
    run_test()

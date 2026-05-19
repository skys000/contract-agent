# -*- coding: utf-8 -*-
"""
应用名: app.py
作用: 劳动合同合规审查智能体的 Streamlit 前端交互系统与运营仪表盘大屏。
      融合了玻璃拟态 UI 样式、数据脱敏、元数据提取、本地 SQLite 历史存盘及可视化图表。
"""

import streamlit as st
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

# 加载自定义包 (激活虚拟环境运行)
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from parser import extract_contract_text, desensitize_text, extract_metadata
from database import init_db, insert_audit_log, get_kpi_metrics, get_recent_activities, get_monthly_risk_stats
from agent import build_agent_graph

# 预加载环境变量并初始化数据库
load_dotenv()
init_db()

# 配置 Streamlit 页面属性为宽版模式
st.set_page_config(page_title="劳动合同合规审查智能体系统", layout="wide", page_icon="🛡️")

# ==========================================
# 1. 自定义 CSS 样式设计 (玻璃拟态 + 渐变色 + 现代字体)
# ==========================================
st.markdown("""
<style>
    /* 全局背景优化 */
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    
    /* 玻璃拟态区块卡片 */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* 自定义大标题样式 */
    .custom-title {
        font-size: 38px;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        font-family: 'Outfit', sans-serif;
    }
    
    .custom-subtitle {
        font-size: 16px;
        color: #94a3b8;
        margin-bottom: 30px;
    }
    
    /* 最近活动流表格优化 */
    .activity-table {
        width: 100%;
        border-collapse: collapse;
        color: #e2e8f0;
    }
    .activity-table th {
        background-color: rgba(51, 65, 85, 0.6);
        color: #38bdf8;
        text-align: left;
        padding: 12px;
        font-weight: 600;
        font-size: 14px;
        border-bottom: 2px solid rgba(255, 255, 255, 0.05);
    }
    .activity-table td {
        padding: 12px;
        font-size: 13px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .activity-table tr:hover {
        background-color: rgba(255, 255, 255, 0.03);
    }
    
    /* 风险标签样式 */
    .badge-high {
        background-color: #f43f5e;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: bold;
    }
    .badge-med {
        background-color: #f59e0b;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper 函数：绘制高画质 KPI 指标卡
def render_kpi_card(title: str, value: str, subtitle: str, gradient_css: str):
    st.markdown(f"""
    <div style="background: {gradient_css}; padding: 22px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.1); color: white;">
        <div style="font-size: 13px; opacity: 0.85; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">{title}</div>
        <div style="font-size: 32px; font-weight: 800; margin: 6px 0; font-family: 'Outfit', sans-serif;">{value}</div>
        <div style="font-size: 12px; opacity: 0.7; font-weight: 400;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 2. 顶栏标题区
# ==========================================
st.markdown('<div class="custom-title">🛡️ 劳动合同合规审查智能体系统</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-subtitle">基于 LangGraph 循环反思工作流 & FAISS 向量法规检索的高级 SaaS 审查大屏</div>', unsafe_allow_html=True)


# ==========================================
# 3. 运营仪表盘看板大屏 (读取 SQLite 数据)
# ==========================================
kpis = get_kpi_metrics()

col1, col2, col3, col4 = st.columns(4)
with col1:
    render_kpi_card(
        "累计已审查合同", 
        f"{kpis['total_audits']} 份", 
        "基于 SQLite 持久化统计", 
        "linear-gradient(135deg, #4f46e5, #06b6d4)"
    )
with col2:
    render_kpi_card(
        "高风险合同占比", 
        kpis['high_risk_ratio'], 
        "包含 1 项及以上高风险条款", 
        "linear-gradient(135deg, #e11d48, #f43f5e)"
    )
with col3:
    render_kpi_card(
        "平均审查速度", 
        kpis['average_duration'], 
        "RAG 检索与双智能体反思总耗时", 
        "linear-gradient(135deg, #10b981, #059669)"
    )
with col4:
    render_kpi_card(
        "安全性与数据隐私", 
        "本地脱敏开启", 
        "敏感数字/身份证号强制替换", 
        "linear-gradient(135deg, #7c3aed, #9333ea)"
    )

st.markdown("<br>", unsafe_allow_html=True)


# ==========================================
# 4. 图表统计与最新活动流展示
# ==========================================
chart_col, flow_col = st.columns([1.1, 1])

# 中文显示配置避免乱码
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

with chart_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📈 历史审查风险分布统计")
    
    # 读取历史统计数据
    filenames, highs, meds = get_monthly_risk_stats()
    
    if filenames:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        x = np.arange(len(filenames))
        width = 0.35
        
        # 绘制高风险与中风险并排柱状图
        ax.bar(x - width/2, highs, width, label='高风险', color='#f43f5e')
        ax.bar(x + width/2, meds, width, label='中风险', color='#f59e0b')
        
        ax.set_ylabel('风险条款数', color='#94a3b8')
        ax.set_title('近期上传合同风险项对比图', color='#f1f5f9', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(filenames, rotation=15, color='#94a3b8', fontsize=8)
        ax.legend(facecolor='#1e293b', edgecolor='none', labelcolor='#e2e8f0')
        ax.tick_params(colors='#94a3b8')
        
        # 移除多余的网格和外框，使视觉与暗色背景统一
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        ax.spines['bottom'].set_color('#334155')
        
        st.pyplot(fig)
    else:
        st.info("暂无足够的数据生成图表。")
    st.markdown('</div>', unsafe_allow_html=True)

with flow_col:
    st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
    st.subheader("📋 最新合同审查活动流列表")
    
    activities = get_recent_activities(limit=5)
    if activities:
        # 使用没有缩进的 HTML 字符串，防止 Markdown 解析器误判为代码块
        html_code = "<table class='activity-table'><thead><tr><th>合同名称</th><th>甲方单位</th><th>乙方姓名</th><th>风险分布</th><th>日期</th></tr></thead><tbody>"
        for act in activities:
            html_code += f"<tr><td>{act['filename']}</td><td>{act['party_a']}</td><td>{act['party_b']}</td><td><span class='badge-high'>高 {act['risk_high']}</span> <span class='badge-med'>中 {act['risk_med']}</span></td><td style='color:#94a3b8;'>{act['created_at'].split(' ')[0]}</td></tr>"
        html_code += "</tbody></table>"
        st.markdown(html_code, unsafe_allow_html=True)
    else:
        st.info("尚无上传审核记录。")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.08);'>", unsafe_allow_html=True)


# ==========================================
# 5. 上传与审查主工作区
# ==========================================
upload_card = st.container()
with upload_card:
    st.markdown('### 📂 上传劳动合同进行智能审计')
    uploaded_file = st.file_uploader("支持拖拽或选择 .docx 或 .pdf 合同源文档 (限制 20MB 以内)", type=["docx", "pdf"])

if uploaded_file is not None:
    # 模拟临时保存以供解析器解析
    temp_dir = os.path.join(os.path.dirname(__file__), "data", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.success(f"[OK] 文件 {uploaded_file.name} 接收成功，已存入安全本地缓冲区。")
    
    # 执行文本提取
    with st.spinner("正在提取并进行合同排版整理..."):
        try:
            raw_text = extract_contract_text(temp_path)
            
            # 1. 提取元数据 (SRS 3.1)
            metadata = extract_metadata(raw_text)
            
            # 2. 本地隐私数据脱敏 (SRS 5.3)
            clean_text = desensitize_text(raw_text)
            
        except Exception as e:
            st.error(f"合同提取失败: {e}")
            clean_text = None
            
    if clean_text:
        # 双栏渲染布局
        left_col, right_col = st.columns(2)
        
        with left_col:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📑 提取到的合同元数据")
            
            meta_col1, meta_col2 = st.columns(2)
            meta_col1.write(f"**甲方单位**: {metadata['party_a']}")
            meta_col1.write(f"**合同期限**: {metadata['duration']}")
            meta_col2.write(f"**乙方姓名**: {metadata['party_b']}")
            meta_col2.write(f"**约定薪资**: {metadata['salary']}")
            
            st.markdown("---")
            st.subheader("📝 本地脱敏后的合同文本原文")
            st.info("🛡️ [安全保护中] 敏感身份证号、手机号已自动脱敏，防止被传往公网大模型 API 泄露隐私。")
            st.text_area("清洗后文本预览", clean_text, height=350)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with right_col:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🤖 智能体协同会审面板")
            
            # 开始审核按钮
            run_audit = st.button("🚀 开始双智能体协同合规会审", use_container_width=True)
            
            if run_audit:
                # 状态流转可视化展示栏
                status_box = st.empty()
                
                # 1. 执行 RAG 法规库检索
                status_box.info("🔍 [1/4] 正在根据合同内容，从本地 FAISS 向量库检索比对我国最新劳动法条...")
                db_dir = os.path.join(os.path.dirname(__file__), "data", "faiss_index")
                try:
                    # 检索前 300 字符召回核心关联法条
                    retrieved_laws = query_laws(clean_text[:300], db_dir, top_k=3)
                except Exception:
                    # 降级检索
                    retrieved_laws = "《中华人民共和国劳动合同法》第十九条：试用期规定..."
                    
                # 2. 构造 LangGraph 初始状态
                inputs = {
                    "contract_text": clean_text,
                    "retrieved_laws": retrieved_laws,
                    "raw_audit": "",
                    "feedback": "",
                    "final_report": "",
                    "loop_count": 0
                }
                
                # 3. 运行多智能体图
                status_box.info("🤖 [2/4] 启动 Auditor 初审智能体进行全面条款审核与风险评级...")
                app = build_agent_graph()
                
                try:
                    # 调用状态机运行
                    result = app.invoke(inputs)
                    
                    status_box.info("⚖️ [3/4] 启动 Critic 反思纠错智能体，正在校正法律幻觉与核实风险等级...")
                    # 状态机内部已自动流转至 Critic 节点及通过路由
                    
                    status_box.success("🎉 [4/4] 审查完成！已格式化生成最终审查报告。")
                    
                    # 4. 展示最终合规审查报告
                    st.markdown("### 📄 最终审查报告展示")
                    st.markdown(result["final_report"])
                    
                    # 5. 自动解析风险点，写入本地 SQLite 历史库
                    raw_audit_text = result.get("raw_audit", "")
                    risk_high_cnt = raw_audit_text.count("高风险")
                    risk_med_cnt = raw_audit_text.count("中风险")
                    
                    insert_audit_log(
                        filename=uploaded_file.name,
                        party_a=metadata["party_a"],
                        party_b=metadata["party_b"],
                        risk_high=risk_high_cnt,
                        risk_med=risk_med_cnt
                    )
                    
                    # 6. 一键下载 Markdown 报告物理文件 (SRS 3.4)
                    st.markdown("---")
                    st.download_button(
                        label="💾 一键导出并下载合规报告 (Markdown格式)",
                        data=result["final_report"],
                        file_name=f"劳动合同合规审查报告_{uploaded_file.name.split('.')[0]}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                    
                except Exception as ex:
                    st.error(f"大模型审核发生错误，请检查网络或 .env 配置: {ex}")
            else:
                st.info("💡 请点击上方按钮，启动双智能体（Auditor-Critic）会审流程。")
            st.markdown('</div>', unsafe_allow_html=True)
            
    # 清理物理临时文件
    if os.path.exists(temp_path):
        os.remove(temp_path)

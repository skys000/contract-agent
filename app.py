# -*- coding: utf-8 -*-
"""
应用名: app.py
作用: 劳动合同合规审查智能体的 Streamlit 前端交互系统与运营仪表盘大屏。
      基于“苹果极简深空灰 (Apple Space-Gray)”美学重塑，实现信息解耦的多标签导航、
      数据本地安全脱敏比对展示、SQLite 存盘及 RAG 法律文本检索。
"""

import streamlit as st
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

# 将 src 目录临时加入模块查找路径
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from parser import extract_contract_text, desensitize_text, extract_metadata
from database import init_db, insert_audit_log, get_kpi_metrics, get_recent_activities, get_monthly_risk_stats
from agent import build_agent_graph
from retriever import query_laws

# 预加载环境变量并初始化数据库
load_dotenv()
init_db()

# 配置 Streamlit 页面属性为宽版模式
st.set_page_config(page_title="劳动合同合规审查智能体系统", layout="wide", page_icon="🛡️")

# ==========================================
# 1. 苹果深空灰 (Space-Gray) 极简 CSS 样式注入
# ==========================================
st.markdown("""
<style>
    /* 全局背景与基本文本优化 */
    .stApp {
        background-color: #161617;
        color: #f5f5f7;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* 苹果极简卡片 */
    .apple-card {
        background-color: #252526;
        border-radius: 12px;
        border: 1px solid #424245;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    /* 标题样式（无渐变，纯苹果风） */
    .custom-title {
        font-size: 30px;
        font-weight: 700;
        color: #f5f5f7;
        margin-top: 10px;
        margin-bottom: 5px;
        letter-spacing: -0.02em;
    }
    
    .custom-subtitle {
        font-size: 14px;
        color: #86868b;
        margin-bottom: 24px;
    }
    
    /* 表格样式重塑 */
    .apple-table {
        width: 100%;
        border-collapse: collapse;
        color: #f5f5f7;
        margin-top: 8px;
    }
    .apple-table th {
        background-color: #1d1d1f;
        color: #86868b;
        text-align: left;
        padding: 12px 16px;
        font-weight: 600;
        font-size: 13px;
        border-bottom: 1px solid #424245;
    }
    .apple-table td {
        padding: 12px 16px;
        font-size: 13px;
        border-bottom: 1px solid #2d2d2f;
    }
    .apple-table tr:hover {
        background-color: #2a2a2c;
    }
    
    /* 风险状态扁平化徽章 */
    .badge-high {
        background-color: rgba(255, 69, 58, 0.15);
        color: #ff453a;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-med {
        background-color: rgba(255, 159, 10, 0.15);
        color: #ff9f0a;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-low {
        background-color: rgba(48, 209, 88, 0.15);
        color: #30d158;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
    }
    
    /* Streamlit Tab 样式重塑为苹果扁平分段控制器 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1d1d1f;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #424245;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 6px;
        color: #86868b;
        font-weight: 500;
        font-size: 14px;
        border: none;
        padding: 0 18px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f5f5f7;
        background-color: rgba(255, 255, 255, 0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: #252526 !important;
        color: #2f9eef !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    /* 隐藏 tab 底线 */
    .stTabs [data-baseweb="tab-highlight-container"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Helper 函数：绘制苹果扁平卡片（带色边框强调）
def render_kpi_card(title: str, value: str, subtitle: str, border_color: str = "#424245"):
    st.markdown(f"""
    <div style="background-color: #252526; padding: 20px; border-radius: 12px; border: 1px solid {border_color}; color: #f5f5f7;">
        <div style="font-size: 12px; color: #86868b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">{title}</div>
        <div style="font-size: 28px; font-weight: 700; margin: 8px 0; color: #f5f5f7;">{value}</div>
        <div style="font-size: 11px; color: #86868b; font-weight: 400;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 顶部主标题栏
# ==========================================
st.markdown('<div class="custom-title">🛡️ 劳动合同合规审查智能体系统</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-subtitle">极简、安全、专业的双智能体协同合规审查系统（Space-Gray Minimalist）</div>', unsafe_allow_html=True)

# ==========================================
# 3. 多页面 Tab 标签导航设计 (解耦复杂界面)
# ==========================================
tab_audit, tab_dashboard, tab_library = st.tabs([
    "🛡️ 智能合规审计工作区", 
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
    
    if uploaded_file is not None:
        # 建立缓冲区存放临时处理文件
        temp_dir = os.path.join(os.path.dirname(__file__), "data", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner("正在提取并整理合同内容..."):
            try:
                # 段落提取与清洗
                raw_text = extract_contract_text(temp_path)
                # 元数据智能抓取
                metadata = extract_metadata(raw_text)
                # 本地隐私脱敏（身份证、手机号）
                clean_text = desensitize_text(raw_text)
            except Exception as e:
                st.error(f"提取合同失败: {e}")
                clean_text = None
                
        if clean_text:
            # 双栏流式布局 (50% 对称排列)
            left_col, right_col = st.columns(2)
            
            with left_col:
                st.markdown('<div class="apple-card">', unsafe_allow_html=True)
                st.subheader("📑 提取到的合同元数据")
                
                # 元数据 2x2 极简表格样式呈现
                st.markdown(f"""
                <table style='width:100%; border:none; color:#f5f5f7; font-size:14px; margin-bottom:15px;'>
                    <tr>
                        <td style='padding:8px 0; border:none; width:50%;'><span style='color:#86868b;'>甲方单位:</span> <strong style='color:#f5f5f7;'>{metadata['party_a']}</strong></td>
                        <td style='padding:8px 0; border:none; width:50%;'><span style='color:#86868b;'>乙方姓名:</span> <strong style='color:#f5f5f7;'>{metadata['party_b']}</strong></td>
                    </tr>
                    <tr>
                        <td style='padding:8px 0; border:none;'><span style='color:#86868b;'>合同期限:</span> <strong style='color:#f5f5f7;'>{metadata['duration']}</strong></td>
                        <td style='padding:8px 0; border:none;'><span style='color:#86868b;'>约定薪资:</span> <strong style='color:#f5f5f7;'>{metadata['salary']}</strong></td>
                    </tr>
                </table>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.subheader("📝 本地脱敏后的合同文本")
                st.markdown(
                    "<div style='font-size:12px; color:#30d158; background-color:rgba(48,209,88,0.1); padding:8px 12px; border-radius:6px; border:1px solid rgba(48,209,88,0.25); margin-bottom:12px;'>"
                    "🛡️ [安全脱敏开启] 身份证号和联系电话已本地强制脱敏，不会传至任何第三方 API。"
                    "</div>", 
                    unsafe_allow_html=True
                )
                st.text_area("清洗后文本预览 (只读)", clean_text, height=380, disabled=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with right_col:
                st.markdown('<div class="apple-card">', unsafe_allow_html=True)
                st.subheader("🤖 智能合规会审面板")
                
                # 苹果淡蓝高亮主行动按钮
                st.markdown("""
                <style>
                    div.stButton > button:first-child {
                        background-color: #2f9eef !important;
                        color: #ffffff !important;
                        border: none !important;
                        border-radius: 8px !important;
                        font-weight: 600 !important;
                        font-size: 15px !important;
                        height: 44px !important;
                        transition: background-color 0.2s ease, transform 0.1s ease !important;
                    }
                    div.stButton > button:first-child:hover {
                        background-color: #1a8ad4 !important;
                        transform: translateY(-1px);
                    }
                    div.stButton > button:first-child:active {
                        transform: translateY(1px);
                    }
                </style>
                """, unsafe_allow_html=True)
                
                run_audit = st.button("🚀 开始双智能体协同合规会审", use_container_width=True)
                
                if run_audit:
                    status_box = st.empty()
                    
                    # 1. 运行检索匹配
                    status_box.markdown(
                        "<div style='color:#2f9eef; font-size:13px; margin: 10px 0;'>🔍 [1/4] 正在从本地 FAISS 法律知识库召回相关劳动法条规约...</div>", 
                        unsafe_allow_html=True
                    )
                    db_dir = os.path.join(os.path.dirname(__file__), "data", "faiss_index")
                    try:
                        retrieved_laws = query_laws(clean_text[:400], db_dir, top_k=3)
                    except Exception:
                        retrieved_laws = "《中华人民共和国劳动合同法》第十九条：试用期规定..."
                        
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
                        "<div style='color:#2f9eef; font-size:13px; margin: 10px 0;'>🤖 [2/4] 启动 Auditor 初审智能体对条款进行法条比对与评估...</div>", 
                        unsafe_allow_html=True
                    )
                    app = build_agent_graph()
                    
                    try:
                        result = app.invoke(inputs)
                        
                        status_box.markdown(
                            "<div style='color:#2f9eef; font-size:13px; margin: 10px 0;'>⚖️ [3/4] 启动 Critic 反思审计节点，正在进行风险核校并整理修正对策...</div>", 
                            unsafe_allow_html=True
                        )
                        
                        status_box.markdown(
                            "<div style='color:#30d158; font-size:13px; font-weight:600; margin: 10px 0;'>🎉 [4/4] 审查完成！以下为联合生成的《合规智能会审报告》：</div>", 
                            unsafe_allow_html=True
                        )
                        
                        st.markdown("### 📄 智能合规审查报告")
                        st.markdown(result["final_report"])
                        
                        # 4. 解析风险统计并持久化入库
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
                        
                        # 5. 苹果风次级下载按钮
                        st.markdown("---")
                        st.download_button(
                            label="💾 一键导出合规审查报告 (Markdown格式)",
                            data=result["final_report"],
                            file_name=f"劳动合同合规审查报告_{uploaded_file.name.split('.')[0]}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                        
                    except Exception as ex:
                        st.error(f"双智能体调用失败，请检查网络或 DeepSeek API 密钥。错误: {ex}")
                else:
                    # 激活状态等待提示
                    st.markdown("""
                    <div style="text-align: center; padding: 40px 10px; color: #86868b;">
                        <div style="font-size: 24px; margin-bottom: 8px;">⚖️</div>
                        <div style="font-size: 13px;">点击上方按钮，启动双智能体反思协同会审</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
        # 物理清理本地缓存
        if os.path.exists(temp_path):
            os.remove(temp_path)
    else:
        # 清爽优雅的空置首屏引导区 (Onboard State)
        st.markdown("""
        <div style="text-align: center; padding: 70px 20px; color: #86868b; background-color: #252526; border-radius: 12px; border: 1px dashed #424245;">
            <div style="font-size: 52px; margin-bottom: 18px;">🛡️</div>
            <div style="font-size: 16px; font-weight: 600; color: #f5f5f7; margin-bottom: 8px;">准备好开始合规审查了吗？</div>
            <div style="font-size: 13px; max-width: 460px; margin: 0 auto 20px auto; line-height: 1.45;">
                请在上方拖入或选择一份需要审计的劳动合同。系统会利用本地正则表达式对身份证号与电话号码进行前置安全占位脱敏，保护数据不泄露，然后结合劳动法向量文库召回法律参考，协助 Auditor 与 Critic 进行专业审计。
            </div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------
# TAB 2: 运营数据看板大屏
# ------------------------------------------
with tab_dashboard:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 动态抓取库中最新统计
    kpis = get_kpi_metrics()
    
    # KPI 排布 (利用有色细线对不同卡片进行极简强调，规避大彩块)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card(
            "累计已审查合同", 
            f"{kpis['total_audits']} 份", 
            "SQLite 本地物理库统计数", 
            "#424245" # 默认银灰
        )
    with col2:
        render_kpi_card(
            "高风险合同占比", 
            kpis['high_risk_ratio'], 
            "含有 1 项以上高风险项合同", 
            "rgba(255, 69, 58, 0.4)" # 高亮淡红
        )
    with col3:
        render_kpi_card(
            "平均审查速度", 
            kpis['average_duration'], 
            "双智能体流转反思耗时均值", 
            "rgba(47, 158, 239, 0.4)" # 高亮淡蓝
        )
    with col4:
        render_kpi_card(
            "安全性与数据隐私", 
            "本地脱敏激活", 
            "敏感身份证/电话号强制转换", 
            "rgba(48, 209, 88, 0.4)" # 高亮淡绿
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 图表与活动流并排排版
    chart_col, flow_col = st.columns([1.1, 1])
    
    # 中文字体防乱码设置
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    
    with chart_col:
        st.markdown('<div class="apple-card">', unsafe_allow_html=True)
        st.subheader("📈 历史审查风险分布统计")
        
        # 抓取图表统计数据
        filenames, highs, meds = get_monthly_risk_stats()
        
        if filenames:
            # 苹果风格深空灰配色主题图表绘制
            fig, ax = plt.subplots(figsize=(7, 3.2), facecolor='#252526')
            ax.set_facecolor('#252526')
            
            x = np.arange(len(filenames))
            width = 0.35
            
            # 高风险红，中风险橙
            ax.bar(x - width/2, highs, width, label='高风险', color='#ff453a', edgecolor='none')
            ax.bar(x + width/2, meds, width, label='中风险', color='#ff9f0a', edgecolor='none')
            
            ax.set_ylabel('风险条款数', color='#86868b', fontsize=8)
            ax.set_title('近期上传合同风险项对比图', color='#f5f5f7', fontsize=9, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(filenames, rotation=12, color='#86868b', fontsize=7)
            
            ax.legend(facecolor='#161617', edgecolor='#424245', labelcolor='#f5f5f7', fontsize=7)
            ax.tick_params(colors='#86868b', labelsize=8)
            
            # 精简轴线
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#424245')
            ax.spines['bottom'].set_color('#424245')
            
            st.pyplot(fig)
        else:
            st.info("数据为空，待上传审核合同生成数据图表。")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with flow_col:
        st.markdown('<div class="apple-card" style="height: 100%;">', unsafe_allow_html=True)
        st.subheader("📋 最新合同审查活动流列表")
        
        activities = get_recent_activities(limit=5)
        if activities:
            html_code = "<table class='apple-table'><thead><tr><th>合同名称</th><th>甲方单位</th><th>乙方姓名</th><th>风险分布</th><th>日期</th></tr></thead><tbody>"
            for act in activities:
                html_code += f"<tr><td>{act['filename']}</td><td>{act['party_a']}</td><td>{act['party_b']}</td><td><span class='badge-high'>高 {act['risk_high']}</span> <span class='badge-med'>中 {act['risk_med']}</span></td><td style='color:#86868b;'>{act['created_at'].split(' ')[0]}</td></tr>"
            html_code += "</tbody></table>"
            st.markdown(html_code, unsafe_allow_html=True)
        else:
            st.info("尚无审查流水记录。")
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------
# TAB 3: 合规法律条款文库
# ------------------------------------------
with tab_library:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📚 劳动合规法律法规库")
    
    # 扫描 laws 目录
    laws_dir = os.path.join(os.path.dirname(__file__), "data", "laws")
    if os.path.exists(laws_dir):
        law_files = [f for f in os.listdir(laws_dir) if f.endswith((".docx", ".pdf"))]
        if law_files:
            st.markdown('<div class="apple-card">', unsafe_allow_html=True)
            st.markdown("##### 📂 已成功加载的法律与司法解释文本（本地 RAG 数据源）")
            for name in law_files:
                st.markdown(f"- 📄 `{name}`")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("法律法规目录为空，请将参考法条放入 data/laws/ 目录。")
            
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔍 法律法规语义检索比对器")
    st.markdown(
        "<div style='font-size:13px; color:#86868b; margin-bottom:16px;'>"
        "输入您关心的合同约定描述（例如：试用期约定六个月、离职不予退还保证金、拒绝缴纳社保等），"
        "检索器将利用 BAAI/bge-m3 语义向量对 FAISS 知识库进行余弦距离度量并召回最为匹配的背景法条。"
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
                results = query_laws(law_query, db_dir, top_k=2)
                st.markdown("##### ⚖️ RAG 检索命中的最相关背景法条（Top 2）")
                # 放在带有苹果细线边框的区域
                st.markdown(f"""
                <div style='background-color:#1d1d1f; padding:16px; border-radius:8px; border:1px solid #424245; white-space:pre-wrap; color:#f5f5f7; font-size:13px; line-height:1.6;'>
{results}
                </div>
                """, unsafe_allow_html=True)
            except Exception as ex:
                st.error(f"检索失败，请确认是否构建了本地 FAISS 索引。错误: {ex}")

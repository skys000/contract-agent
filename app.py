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
import base64
from dotenv import load_dotenv

# 将 src 目录临时加入模块查找路径
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from parser import extract_contract_text, desensitize_text, extract_metadata
from database import init_db, insert_audit_log, get_kpi_metrics, get_recent_activities, get_monthly_risk_stats
from agent import build_agent_graph
from retriever import query_laws

def count_risk_items(report_text: str, level: str) -> int:
    pattern = rf"(?m)^\s*(?:#{{1,6}}\s*)?(?:\*\*)?(?:【{level}】|{level}项\s*\d+[:：]|{level}项[:：])"
    return len(re.findall(pattern, report_text))

# 预加载环境变量并初始化数据库
load_dotenv()
init_db()

# 配置 Streamlit 页面属性为宽版模式
st.set_page_config(page_title="劳动合同合规审查智能体系统", layout="wide", page_icon="🛡️")

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
def render_kpi_card(title: str, value: str, subtitle: str, border_color: str = "#d2d2d7"):
    st.markdown(f"""
    <div style="background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid {border_color}; color: #1d1d1f; box-shadow: 0 4px 16px rgba(0,0,0,0.02);">
        <div style="font-size: 11px; color: #86868b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">{title}</div>
        <div style="font-size: 26px; font-weight: 700; margin: 6px 0; color: #1d1d1f;">{value}</div>
        <div style="font-size: 11px; color: #86868b; font-weight: 400;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 顶部主标题栏
# ==========================================
st.markdown('<div class="custom-title">🛡️ 劳动合同合规审查智能体系统</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-subtitle">极简、安全、专业的双智能体协同合规审查系统（Apple Light Mode）</div>', unsafe_allow_html=True)

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
    
    # 初始化审计结果会话状态以支持一键导出下载（避免 Streamlit 刷新导致下载失败）
    if "audit_result" not in st.session_state:
        st.session_state["audit_result"] = None
    if "current_file" not in st.session_state:
        st.session_state["current_file"] = None

    if uploaded_file is not None:
        if st.session_state["current_file"] != uploaded_file.name:
            st.session_state["audit_result"] = None
            st.session_state["current_file"] = uploaded_file.name
            
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
                    st.text_area("清洗后文本预览 (只读)", clean_text, height=380, disabled=True)
                
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
                            result = app.invoke(inputs)
                            
                            status_box.markdown(
                                "<div style='color:#0071e3; font-size:13px; margin: 10px 0;'>⚖️ [3/4] 启动 Critic 反思审计节点，正在进行风险核校并整理修正对策...</div>", 
                                unsafe_allow_html=True
                            )
                            
                            # 4. 解析风险统计并持久化入库
                            raw_audit_text = result.get("raw_audit", "")
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
                            
                            status_box.empty()
                            
                        except Exception as ex:
                            st.error(f"双智能体调用失败，请检查网络或 DeepSeek API 密钥。错误: {ex}")
                            
                    # 判断会话状态中是否有已生成的审计报告，若有则持续渲染（避免由于 Streamlit 重新运行导致的下载中断）
                    if st.session_state["audit_result"] is not None:
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
                
        # 物理清理本地缓存
        if os.path.exists(temp_path):
            os.remove(temp_path)
    else:
        # 重置审计会话状态
        st.session_state["audit_result"] = None
        st.session_state["current_file"] = None
        
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
            "#d2d2d7" # 银灰
        )
    with col2:
        render_kpi_card(
            "高风险合同占比", 
            kpis['high_risk_ratio'], 
            "含有 1 项以上高风险项合同", 
            "rgba(255, 59, 48, 0.4)" # 高亮淡红
        )
    with col3:
        render_kpi_card(
            "平均审查速度", 
            kpis['average_duration'], 
            "双智能体流转反思耗时均值", 
            "rgba(0, 113, 227, 0.4)" # 高亮淡蓝
        )
    with col4:
        render_kpi_card(
            "安全性与数据隐私", 
            "本地脱敏激活", 
            "敏感身份证/电话号强制转换", 
            "rgba(52, 199, 89, 0.4)" # 高亮淡绿
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
                
                x = np.arange(len(filenames))
                width = 0.35
                
                # 高风险红，中风险橙
                ax.bar(x - width/2, highs, width, label='高风险', color='#ff3b30', edgecolor='none')
                ax.bar(x + width/2, meds, width, label='中风险', color='#ff9500', edgecolor='none')
                
                ax.set_ylabel('风险条款数', color='#86868b', fontsize=8)
                ax.set_title('近期上传合同风险项对比图', color='#1d1d1f', fontsize=9, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(filenames, rotation=12, color='#86868b', fontsize=7)
                
                ax.legend(facecolor='#f5f5f7', edgecolor='#d2d2d7', labelcolor='#1d1d1f', fontsize=7)
                ax.tick_params(colors='#86868b', labelsize=8)
                
                # 精简轴线
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#d2d2d7')
                ax.spines['bottom'].set_color('#d2d2d7')
                
                st.pyplot(fig)
            else:
                st.info("数据为空，待上传审核合同生成数据图表。")
        
    with flow_col:
        with st.container(border=True):
            st.markdown("<h4 style='margin-top:0; color:#1d1d1f;'>📋 最新合同审查活动流列表</h4>", unsafe_allow_html=True)
            
            activities = get_recent_activities(limit=5)
            if activities:
                html_code = "<table class='apple-table'><thead><tr><th>合同名称</th><th>甲方单位</th><th>乙方姓名</th><th>风险分布</th><th>耗时</th><th>审查时间</th></tr></thead><tbody>"
                for act in activities:
                    html_code += f"<tr><td>{act['filename']}</td><td>{act['party_a']}</td><td>{act['party_b']}</td><td><span class='badge-high'>高 {act['risk_high']}</span> <span class='badge-med'>中 {act['risk_med']}</span></td><td style='color:#86868b;'>{round(act['duration_seconds'] or 0, 1)}秒</td><td style='color:#86868b;'>{act['created_at']}</td></tr>"
                html_code += "</tbody></table>"
                st.markdown(html_code, unsafe_allow_html=True)
            else:
                st.info("尚无审查流水记录。")

# ------------------------------------------
# TAB 3: 合规法律条款文库
# ------------------------------------------
with tab_library:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 扫描 laws 目录
    laws_dir = os.path.join(os.path.dirname(__file__), "data", "laws")
    if os.path.exists(laws_dir):
        law_files = [f for f in os.listdir(laws_dir) if f.endswith((".docx", ".pdf"))]
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

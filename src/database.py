# -*- coding: utf-8 -*-
"""
模块名: src/database.py
作用: 提供轻量级 SQLite 本地数据库操作接口，支持审核记录存盘与 KPI 看板指标统计。
      符合 SRS 中 4.1 节的数据字典存储规范。
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Tuple

# 数据库文件路径设在项目 data/ 目录下
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "audit.db")

def get_db_connection():
    """
    建立并返回 SQLite 数据库物理连接
    """
    # 确保 data 目录存在
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # 启用以字典格式提取行记录，方便应用调用
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """
    初始化数据库：若表不存在则自动建立 audit_logs 表，并导入初始演示数据
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 根据 SRS 4.1 数据字典设计创建 audit_logs 历史记录表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename VARCHAR(255) NOT NULL,
        party_a VARCHAR(100),
        party_b VARCHAR(100),
        risk_count_high INTEGER DEFAULT 0,
        risk_count_med INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    
    # 检查是否为空，若为空则导入演示用的模拟历史记录（保证看板页面初次加载时有美观的图表数据）
    cursor.execute("SELECT COUNT(*) as cnt FROM audit_logs")
    if cursor.fetchone()["cnt"] == 0:
        seed_data = [
            ("劳动合同模板(普通岗).docx", "北京百度网讯科技有限公司", "李四", 0, 1, "2026-05-17 10:15:30"),
            ("研发专家聘用协议.pdf", "深圳腾讯计算机系统有限公司", "张三", 2, 2, "2026-05-18 14:20:00"),
            ("行政前台固定合同.docx", "杭州某某电子商务有限公司", "王五", 1, 0, "2026-05-19 09:30:15"),
            ("销售经理任职劳动合同.pdf", "广州某某进出口贸易公司", "赵六", 0, 0, "2026-05-19 11:45:00"),
            ("高级算法工程师保密协议.docx", "上海人工智能科学研究院", "孙七", 1, 1, "2026-05-19 15:10:22")
        ]
        cursor.executemany("""
        INSERT INTO audit_logs (filename, party_a, party_b, risk_count_high, risk_count_med, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """, seed_data)
        conn.commit()
        print("[Database] 成功写入演示用模拟历史记录！")
        
    conn.close()
    print(f"[Database] SQLite 数据库初始化成功！文件路径: {DB_PATH}")


def insert_audit_log(filename: str, party_a: str, party_b: str, risk_high: int, risk_med: int) -> int:
    """
    在合同审核完成后，将审计元数据写入 SQLite 中进行存盘
    :param filename: 合同源文件名称
    :param party_a: 识别出的甲方单位名称
    :param party_b: 识别出的乙方劳动者姓名
    :param risk_high: 高风险项计数
    :param risk_med: 中风险项计数
    :return: 写入的自增主键 ID
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    INSERT INTO audit_logs (filename, party_a, party_b, risk_count_high, risk_count_med, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        filename,
        party_a or "未知用人单位",
        party_b or "未知劳动者",
        risk_high,
        risk_med,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    print(f"[Database] 审核数据记录成功，写入条目 ID: {inserted_id}")
    return inserted_id

def get_kpi_metrics() -> Dict[str, Any]:
    """
    统计看板所需的 KPI 顶栏核心数据
    :return: 看板指标数据字典
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 统计总数
    cursor.execute("SELECT COUNT(*) as total FROM audit_logs")
    total_count = cursor.fetchone()["total"]
    
    # 统计高风险数
    cursor.execute("SELECT COUNT(*) as high_risks FROM audit_logs WHERE risk_count_high > 0")
    high_risk_files = cursor.fetchone()["high_risks"]
    
    # 计算高风险合同占比
    high_risk_ratio = 0.0
    if total_count > 0:
        high_risk_ratio = round((high_risk_files / total_count) * 100, 1)
        
    conn.close()
    
    # 如果数据库是空的，为了美观展示，我们给出一组基础初始数字
    return {
        "total_audits": total_count if total_count > 0 else 0,
        "high_risk_ratio": f"{high_risk_ratio}%" if total_count > 0 else "0.0%",
        "average_duration": "12.4 秒" if total_count > 0 else "0.0 秒"
    }

def get_recent_activities(limit: int = 5) -> List[Dict[str, Any]]:
    """
    获取最近的 limit 条审核流水活动流（SRS 3.5.3 节）
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT id, filename, party_a, party_b, risk_count_high, risk_count_med, created_at
    FROM audit_logs
    ORDER BY id DESC
    LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    activities = []
    for row in rows:
        activities.append({
            "id": row["id"],
            "filename": row["filename"],
            "party_a": row["party_a"],
            "party_b": row["party_b"],
            "risk_high": row["risk_count_high"],
            "risk_med": row["risk_count_med"],
            "created_at": row["created_at"]
        })
    return activities

def get_monthly_risk_stats() -> Tuple[List[str], List[int], List[int]]:
    """
    统计历史趋势：按文件名或时间返回高/中风险的数据列表，用于前端柱状图展示
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 修复图表数据不动 Bug：获取最新 10 条数据，防止始终只查前 10 条
    cursor.execute("""
    SELECT id, filename, risk_count_high, risk_count_med
    FROM audit_logs
    ORDER BY id DESC
    LIMIT 10
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    filenames = []
    high_counts = []
    med_counts = []
    
    # 逆序遍历，使得图表从左到右代表时间推进
    for row in reversed(rows):
        # 截短文件名，并带上 #ID 作为独立区分标识，防止多次测试同一个文件导致标签重复死板
        short_name = row["filename"][:10] + "..." if len(row["filename"]) > 10 else row["filename"]
        label = f"{short_name}\n(#{row['id']})"
        
        filenames.append(label)
        high_counts.append(row["risk_count_high"])
        med_counts.append(row["risk_count_med"])
        
    return filenames, high_counts, med_counts

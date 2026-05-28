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

# 数据库文件路径设在项目 data/ 目录下，避免和源码文件混放
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "audit.db")

def get_db_connection():
    """
    建立并返回 SQLite 数据库物理连接
    """
    # 确保 data 目录存在
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    # 创建 SQLite 连接；如果 audit.db 不存在，sqlite3 会自动创建文件
    conn = sqlite3.connect(DB_PATH)
    # 启用以字典格式提取行记录，方便应用调用
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """
    初始化数据库：若表不存在则自动建立 audit_logs 表，并执行必要的表结构迁移

    当前系统采用轻量本地 SQLite，主要服务于演示看板和历史审计流水，不承载敏感合同全文。
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
        duration_seconds REAL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    # 先提交建表语句，确保后续 PRAGMA 能读取到最新结构
    conn.commit()
    
    # 读取当前表字段列表，用于兼容旧版本数据库文件
    cursor.execute("PRAGMA table_info(audit_logs)")
    columns = {row["name"] for row in cursor.fetchall()}
    if "duration_seconds" not in columns:
        # 旧版本 audit_logs 没有耗时字段时，自动追加列，避免用户手动删库
        cursor.execute("ALTER TABLE audit_logs ADD COLUMN duration_seconds REAL DEFAULT 0")

    # 提交迁移操作
    conn.commit()
        
    # 初始化完成后及时关闭连接，释放文件句柄
    conn.close()
    print(f"[Database] SQLite 数据库初始化成功！文件路径: {DB_PATH}")


def insert_audit_log(filename: str, party_a: str, party_b: str, risk_high: int, risk_med: int, duration_seconds: float = 0.0) -> int:
    """
    在合同审核完成后，将审计元数据写入 SQLite 中进行存盘
    :param filename: 合同源文件名称
    :param party_a: 识别出的甲方单位名称
    :param party_b: 识别出的乙方劳动者姓名
    :param risk_high: 高风险项计数
    :param risk_med: 中风险项计数
    :param duration_seconds: 本次审计从点击按钮到报告落库的耗时
    :return: 写入的自增主键 ID
    """
    # 获取数据库连接和写入游标
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 使用参数化 SQL 写入审查元数据，避免文件名或主体名称中的特殊字符破坏 SQL
    cursor.execute("""
    INSERT INTO audit_logs (filename, party_a, party_b, risk_count_high, risk_count_med, duration_seconds, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        filename,
        # 如果正则无法识别甲乙方，使用统一占位文本保证看板展示完整
        party_a or "未知用人单位",
        party_b or "未知劳动者",
        risk_high,
        risk_med,
        # 耗时保留两位小数，便于看板展示
        round(duration_seconds, 2),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    
    # 提交事务后读取自增主键，作为本次审计流水 ID
    conn.commit()
    inserted_id = cursor.lastrowid
    # 写入完成立即关闭连接，避免长时间占用 SQLite 文件
    conn.close()
    print(f"[Database] 审核数据记录成功，写入条目 ID: {inserted_id}")
    return inserted_id

def get_kpi_metrics() -> Dict[str, Any]:
    """
    统计看板所需的 KPI 顶栏核心数据

    统计口径采用“同一文件名只取最新一次审查记录”，避免反复测试同一合同导致看板指标虚高。
    :return: 看板指标数据字典
    """
    # 获取只读统计连接
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 统一去重口径：同一文件名只取 id 最大的最新审查记录
    latest_rows_filter = "id IN (SELECT MAX(id) FROM audit_logs GROUP BY filename)"

    # 统计每个合同文件的最新审查结果，避免重复测试同一文件污染看板
    cursor.execute(f"SELECT COUNT(*) as total FROM audit_logs WHERE {latest_rows_filter}")
    total_count = cursor.fetchone()["total"]

    # 统计今天上传/审查过的最新合同数量，使用 SQLite localtime 对齐本地演示时间
    cursor.execute(f"SELECT COUNT(*) as today_count FROM audit_logs WHERE {latest_rows_filter} AND date(created_at) = date('now', 'localtime')")
    today_count = cursor.fetchone()["today_count"]
    
    # 统计高风险数
    cursor.execute(f"SELECT COUNT(*) as high_risks FROM audit_logs WHERE {latest_rows_filter} AND risk_count_high > 0")
    high_risk_files = cursor.fetchone()["high_risks"]

    # 统计平均耗时，仅统计大于 0 的有效运行记录
    cursor.execute(f"SELECT AVG(duration_seconds) as avg_duration FROM audit_logs WHERE {latest_rows_filter} AND duration_seconds > 0")
    avg_duration = cursor.fetchone()["avg_duration"] or 0
    
    # 计算高风险合同占比
    high_risk_ratio = 0.0
    if total_count > 0:
        high_risk_ratio = round((high_risk_files / total_count) * 100, 1)
        
    # 统计完成后关闭连接
    conn.close()
    
    # 如果数据库是空的，为了美观展示，我们给出一组基础初始数字
    return {
        "total_audits": total_count if total_count > 0 else 0,
        "today_uploads": today_count if today_count > 0 else 0,
        "high_risk_ratio": f"{high_risk_ratio}%" if total_count > 0 else "0.0%",
        "average_duration": f"{round(avg_duration, 1)} 秒" if avg_duration > 0 else "0.0 秒"
    }

def get_recent_activities(limit: int = 5) -> List[Dict[str, Any]]:
    """
    获取最近的 limit 条审核流水活动流（SRS 3.5.3 节）

    与 KPI 口径一致，同一文件名只展示最新一次审查记录。
    """
    # 获取数据库连接
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 查询最近 limit 个“按文件名去重后的最新审查记录”
    cursor.execute("""
    SELECT id, filename, party_a, party_b, risk_count_high, risk_count_med, duration_seconds, created_at
    FROM audit_logs
    WHERE id IN (SELECT MAX(id) FROM audit_logs GROUP BY filename)
    ORDER BY id DESC
    LIMIT ?
    """, (limit,))
    
    # fetchall 后即可关闭连接，后续只在内存中组装字典
    rows = cursor.fetchall()
    conn.close()
    
    activities = []
    for row in rows:
        # 将 sqlite3.Row 转换为普通 dict，便于 Streamlit 前端直接读取
        activities.append({
            "id": row["id"],
            "filename": row["filename"],
            "party_a": row["party_a"],
            "party_b": row["party_b"],
            "risk_high": row["risk_count_high"],
            "risk_med": row["risk_count_med"],
            "duration_seconds": row["duration_seconds"],
            "created_at": row["created_at"]
        })
    return activities

def get_monthly_risk_stats() -> Tuple[List[str], List[int], List[int]]:
    """
    统计历史趋势：按文件名或时间返回高/中风险的数据列表，用于前端柱状图展示

    当前图表面向课程演示，展示最近若干份合同的风险分布，而不是严格自然月聚合。
    """
    # 获取数据库连接
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 修复图表数据不动 Bug：获取最新 10 条数据，防止始终只查前 10 条
    cursor.execute("""
    SELECT id, filename, risk_count_high, risk_count_med
    FROM audit_logs
    WHERE id IN (SELECT MAX(id) FROM audit_logs GROUP BY filename)
    ORDER BY id DESC
    LIMIT 10
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    # 分别构造 ECharts/Altair 等图表所需的横轴标签和两组风险数量
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

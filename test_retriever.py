# -*- coding: utf-8 -*-
"""
脚本名: test_retriever.py
作用: 验证本地 RAG 向量库构建流程与中文字符相似度召回效果。
"""

import os
import sys
from dotenv import load_dotenv

# 加载全局环境变量
load_dotenv()

# 将 src 目录临时加入模块查找路径
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from retriever import build_law_vector_db, query_laws

def run_test():
    print("=== [TEST] Start RAG database builder and search test ===")
    
    # 路径定义
    laws_dir = os.path.join(os.path.dirname(__file__), "data", "laws")
    db_dir = os.path.join(os.path.dirname(__file__), "data", "faiss_index")
    
    # 1. 构建向量数据库
    try:
        build_law_vector_db(laws_dir, db_dir)
        print("[OK] Vector database successfully created and saved locally!")
    except Exception as e:
        print(f"[ERROR] Failed to build vector database: {e}")
        return
        
    # 2. 测试语义匹配检索
    test_query = "试用期内不给员工缴纳社保，转正后再买"
    print(f"\nTesting Query: '{test_query}'")
    print("--------------------------------------------------")
    
    try:
        reference_laws = query_laws(test_query, db_dir, top_k=2)
        print("[OK] Similarity search completed! Retrieved references below:\n")
        # 替换换行输出，避免终端编码兼容报错
        print(reference_laws.encode('gbk', errors='ignore').decode('gbk'))
    except Exception as e:
        print(f"[ERROR] Failed to execute similarity search: {e}")
        
    print("\n=== [TEST] RAG system verification complete! ===")

if __name__ == "__main__":
    run_test()

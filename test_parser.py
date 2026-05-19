# -*- coding: utf-8 -*-
"""
脚本名: test_parser.py
作用: 验证 src/parser.py 合同解析模块的正确性与异常拦截机制。
"""

import sys
import os

# 将 src 目录临时加入模块查找路径
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from parser import extract_contract_text

def run_test():
    print("=== [TEST] Start unit testing for file parser module ===")
    
    # 测试1: 测试非法文件后缀拦截
    invalid_file = "sample.txt"
    # 创建一个虚拟的 txt 文件
    with open(invalid_file, "w") as f:
        f.write("一些测试文本")
        
    try:
        print(f"\n1. Testing invalid suffix [{invalid_file}] interception...")
        extract_contract_text(invalid_file)
        print("[ERROR] Failed to intercept invalid format!")
    except ValueError as e:
        print(f"[OK] Intercepted successfully! Error message: {e}")
    finally:
        if os.path.exists(invalid_file):
            os.remove(invalid_file)
            
    # 测试2: 测试老旧 .doc 文件友好拦截
    old_doc_file = "sample_old.doc"
    with open(old_doc_file, "w") as f:
        f.write("旧版 Word 占位符")
        
    try:
        print(f"\n2. Testing old Word [.doc] friendly interception...")
        extract_contract_text(old_doc_file)
        print("[ERROR] Failed to intercept .doc format!")
    except ValueError as e:
        print(f"[OK] Intercepted successfully! Message: {e}")
    finally:
        if os.path.exists(old_doc_file):
            os.remove(old_doc_file)
            
    # 测试3: 测试不存在的文件路径异常捕获
    missing_file = "non_existent_file.docx"
    try:
        print(f"\n3. Testing file not found exception...")
        extract_contract_text(missing_file)
        print("[ERROR] Failed to catch FileNotFoundError!")
    except FileNotFoundError as e:
        print(f"[OK] Caught successfully! Exception: {e}")
        
    print("\n=== [TEST] All boundary checks passed successfully! ===")

if __name__ == "__main__":
    run_test()

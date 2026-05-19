# -*- coding: utf-8 -*-
"""
模块名: src/parser.py
作用: 提供对 .docx 和 .pdf 格式劳动合同文件的底层读取与段落空白行清洗提取功能。
"""

import os
from docx import Document
import pypdf

def parse_docx(file_path: str) -> str:
    """
    提取并清洗 Word (.docx) 文档中的全部段落文本
    :param file_path: Word 文件的绝对或相对路径
    :return: 清洗合并后的合同文本字符串，以换行符分隔
    """
    # 初始化 Word 文档解析对象
    doc = Document(file_path)
    full_text_list = []
    
    # 遍历文档中的每一个自然段
    for paragraph in doc.paragraphs:
        # 去除段落前后的空白字符
        clean_text = paragraph.text.strip()
        # 仅保留非空段落，过滤掉文档中的冗余空行
        if clean_text:
            full_text_list.append(clean_text)
            
    # 用换行符连接所有有内容的段落并返回
    return "\n".join(full_text_list)

def parse_pdf(file_path: str) -> str:
    """
    读取并过滤 PDF (.pdf) 文档中的文本数据，保持基本的物理分段
    :param file_path: PDF 文件的绝对或相对路径
    :return: 经过清洗、去空格处理后的 PDF 文本字符串
    """
    full_text_list = []
    
    # 以二进制只读模式打开 PDF 文件
    with open(file_path, "rb") as pdf_file:
        # 初始化 PDF 阅读器
        reader = pypdf.PdfReader(pdf_file)
        
        # 遍历 PDF 的每一页
        for page in reader.pages:
            # 提取当前页的原始文本
            extracted_text = page.extract_text()
            if extracted_text:
                # 对提取出的多行文本按换行拆分，进行精细化去空格清洗
                lines = [line.strip() for line in extracted_text.split("\n") if line.strip()]
                # 将本页清洗后的文本行用换行符重新连接，并加入总列表
                full_text_list.append("\n".join(lines))
                
    # 用换行符拼接所有页面提取的内容并返回
    return "\n".join(full_text_list)

def extract_contract_text(file_path: str) -> str:
    """
    合同文本解析的中心分流函数，依据文件扩展名分流处理并进行格式拦截
    :param file_path: 输入文件的绝对或相对路径
    :return: 解析清洗后的合同原文内容
    :raises ValueError: 当上传不支持的格式或不存在文件时抛出异常
    """
    # 检查物理文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到指定的合同文件: {file_path}")
        
    # 获取并统一转为小写的文件后缀名
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # 分支选择解析引擎
    if file_extension == ".docx":
        return parse_docx(file_path)
    elif file_extension == ".pdf":
        return parse_pdf(file_path)
    elif file_extension == ".doc":
        # 对旧版 Word 格式进行友情拦截，引导用户转换格式
        raise ValueError("系统暂不支持 .doc 格式，请在 Office 中打开并另存为 .docx 格式后再行上传。")
    else:
        # 对不支持的非法扩展名进行强拦截
        raise ValueError(f"系统不支持的文件格式: {file_extension}。请上传 .docx 或 .pdf 合同文档。")

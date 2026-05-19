# -*- coding: utf-8 -*-
"""
模块名: src/retriever.py
作用: 扫描 data/laws 目录，利用 parser.py 提取法条文本并利用 FAISS 构建本地 RAG 向量知识库。
"""

import os
import sys
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

# 确保能正常导入同一目录下的 parser 模块
sys.path.append(os.path.dirname(__file__))
from parser import extract_contract_text

def build_law_vector_db(laws_dir: str, save_dir: str) -> None:
    """
    扫描并解析 laws_dir 目录下的所有 Word/PDF 法律文本，构建本地 FAISS 向量数据库
    :param laws_dir: 存放法律法规原始文件的目录 (data/laws/)
    :param save_dir: FAISS 向量库的本地持久化输出目录 (data/faiss_index/)
    """
    raw_documents: List[LangchainDocument] = []
    
    if not os.path.exists(laws_dir):
        raise FileNotFoundError(f"法律法规源目录不存在: {laws_dir}")
        
    print(f"[RAG] 开始扫描法条目录: {laws_dir}...")
    
    # 1. 遍历 laws_dir 目录下的所有文件并解析
    for file_name in os.listdir(laws_dir):
        file_path = os.path.join(laws_dir, file_name)
        # 仅处理文件，跳过目录或隐藏文件
        if not os.path.isfile(file_path) or file_name.startswith("."):
            continue
            
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext in [".docx", ".pdf", ".txt"]:
            print(f"[RAG] 正在解析法条文件: {file_name}...")
            try:
                # 调用 parser.py 中的高鲁棒分流提取逻辑
                text_content = extract_contract_text(file_path)
                
                # 如果是空白文件则跳过
                if not text_content.strip():
                    continue
                    
                # 封装为 LangChain 的 Document 格式，并附带源文件名元数据
                doc = LangchainDocument(
                    page_content=text_content,
                    metadata={"source": file_name}
                )
                raw_documents.append(doc)
            except Exception as e:
                print(f"[RAG 警告] 解析文件 {file_name} 时出错: {e}，已跳过。")
                
    if not raw_documents:
        raise ValueError(f"未能在目录 {laws_dir} 中找到或成功解析任何有效的法律条文文件。")
        
    # 2. 文本语义切分
    # 设置切分快大小为 450 字符，保留 50 字符的重叠区，防止法条的段落边界被拦腰切断
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", "。", "；", " ", ""]
    )
    split_docs = text_splitter.split_documents(raw_documents)
    print(f"[RAG] 法规文本切分完毕，共生成 {len(split_docs)} 个语义文本块。")
    
    # 3. 初始化 OpenAI 兼容协议的 Embedding 客户端
    # 使用 .env 中配置的 SiliconFlow API-KEY 与 BASE-URL
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("API_KEY"),
        openai_api_base=os.getenv("BASE_URL"),
        model="BAAI/bge-m3",
        chunk_size=32  # 限制分批提交的向量数量，满足硅基流动 API 最大单次 64 的限制
    )
    
    # 4. 构建 FAISS 本地索引
    print("[RAG] 正在调用 BAAI/bge-m3 接口计算法条向量，请稍候...")
    db = FAISS.from_documents(split_docs, embeddings)
    
    # 5. 持久化索引到本地磁盘，以供系统主程序以毫秒级时延高速检索
    os.makedirs(save_dir, exist_ok=True)
    db.save_local(save_dir)
    print(f"[RAG] 向量数据库构建并持久化成功！已保存至: {save_dir}")

def query_laws(query: str, db_dir: str, top_k: int = 3) -> str:
    """
    加载本地 FAISS 索引并检索与 Query 最相关的 Top-K 法条文本
    :param query: 合同待审条款文本
    :param db_dir: 本地 FAISS 向量库索引所在目录 (data/faiss_index/)
    :param top_k: 召回的相似条文数量，默认为 3
    :return: 拼接后的格式化参考法条文本
    """
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("API_KEY"),
        openai_api_base=os.getenv("BASE_URL"),
        model="BAAI/bge-m3",
        chunk_size=32
    )
    
    # 校验本地索引是否存在
    if not os.path.exists(os.path.join(db_dir, "index.faiss")):
        raise FileNotFoundError(f"本地向量数据库未构建，未在 {db_dir} 下找到 index.faiss 文件。")
        
    # 安全加载本地向量库
    # allow_dangerous_deserialization=True 允许反序列化本地持久化的 Pickle 索引结构
    db = FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)
    
    # 执行余弦相似度语义检索
    results = db.similarity_search(query, k=top_k)
    
    # 拼接条文及来源元数据
    formatted_references = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "未知法条来源")
        ref_text = f"【参考依据 {i}】(来源: {source})\n{doc.page_content}"
        formatted_references.append(ref_text)
        
    return "\n\n".join(formatted_references)

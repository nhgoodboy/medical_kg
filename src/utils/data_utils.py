#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
from pathlib import Path
import pandas as pd
import re
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)

def read_medical_texts(data_dir: Union[str, Path]) -> Dict[str, str]:
    """
    读取医学文本数据
    
    Args:
        data_dir: 数据目录
        
    Returns:
        文件名到文本内容的映射
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        logger.warning(f"数据目录不存在: {data_dir}")
        return {}
    
    texts = {}
    
    # 读取TXT文件
    txt_files = list(data_dir.glob("**/*.txt"))
    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 使用相对路径作为文件标识
            rel_path = txt_file.relative_to(data_dir)
            texts[str(rel_path)] = content
            
            logger.info(f"已读取 {rel_path}")
        except Exception as e:
            logger.error(f"读取文件 {txt_file} 失败: {e}")
    
    # 读取JSON文件并提取文本内容
    json_files = list(data_dir.glob("**/*.json"))
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 提取文本内容
            if isinstance(data, dict) and "text" in data:
                # 单个文档
                rel_path = json_file.relative_to(data_dir)
                texts[str(rel_path)] = data["text"]
            elif isinstance(data, list):
                # 多个文档
                for i, doc in enumerate(data):
                    if isinstance(doc, dict) and "text" in doc:
                        rel_path = f"{json_file.relative_to(data_dir)}#{i}"
                        texts[rel_path] = doc["text"]
            
            logger.info(f"已读取 {json_file.relative_to(data_dir)}")
        except Exception as e:
            logger.error(f"读取文件 {json_file} 失败: {e}")
    
    logger.info(f"共读取了 {len(texts)} 个文本文件")
    return texts

def load_json_data(file_path: Union[str, Path]) -> Any:
    """
    加载JSON数据
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        加载的数据
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.warning(f"文件不存在: {file_path}")
        return None
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        logger.info(f"已从 {file_path} 加载数据")
        return data
    except Exception as e:
        logger.error(f"加载JSON文件 {file_path} 失败: {e}")
        return None

def load_medical_data(data_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    加载医学数据
    
    Args:
        data_dir: 数据目录
        
    Returns:
        医学文档列表
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        logger.warning(f"数据目录不存在: {data_dir}")
        return []
    
    documents = []
    
    # 尝试加载不同格式的数据
    # 1. JSON文件
    json_files = list(data_dir.glob("*.json"))
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # 处理不同JSON格式
                if isinstance(data, list):
                    documents.extend(data)
                elif isinstance(data, dict):
                    if "documents" in data:
                        documents.extend(data["documents"])
                    else:
                        documents.append(data)
                        
            logger.info(f"从 {json_file} 加载了 {len(documents)} 个文档")
        except Exception as e:
            logger.error(f"加载JSON文件 {json_file} 失败: {e}")
    
    # 2. CSV文件
    csv_files = list(data_dir.glob("*.csv"))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # 将每一行转换为文档
            for _, row in df.iterrows():
                doc = row.to_dict()
                documents.append(doc)
                
            logger.info(f"从 {csv_file} 加载了 {len(df)} 个文档")
        except Exception as e:
            logger.error(f"加载CSV文件 {csv_file} 失败: {e}")
    
    # 3. TXT文件
    txt_files = list(data_dir.glob("*.txt"))
    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()
                
                # 将一个TXT文件视为一个文档
                doc = {
                    "id": txt_file.stem,
                    "text": content,
                    "source": str(txt_file)
                }
                documents.append(doc)
                
            logger.info(f"从 {txt_file} 加载了 1 个文档")
        except Exception as e:
            logger.error(f"加载TXT文件 {txt_file} 失败: {e}")
    
    # 确保所有文档都有必要的字段
    for i, doc in enumerate(documents):
        if "id" not in doc:
            doc["id"] = f"doc_{i}"
        
        if "text" not in doc:
            # 尝试从其他字段构建文本内容
            text_fields = ["content", "body", "description"]
            for field in text_fields:
                if field in doc:
                    doc["text"] = doc[field]
                    break
            
            if "text" not in doc:
                logger.warning(f"文档 {doc.get('id', f'doc_{i}')} 没有文本内容，将被跳过")
                continue
    
    # 过滤掉没有文本内容的文档
    documents = [doc for doc in documents if "text" in doc and doc["text"]]
    
    logger.info(f"共加载了 {len(documents)} 个有效文档")
    
    return documents

def preprocess_text(text: str) -> str:
    """
    预处理文本
    
    Args:
        text: 原始文本
        
    Returns:
        预处理后的文本
    """
    if not text:
        return ""
    
    # 去除多余空白字符
    text = re.sub(r'\s+', ' ', text.strip())
    
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 统一标点符号（全角转半角）
    text = text.replace('，', ',')
    text = text.replace('。', '.')
    text = text.replace('：', ':')
    text = text.replace('；', ';')
    text = text.replace('？', '?')
    text = text.replace('！', '!')
    text = text.replace('"', '"')
    text = text.replace('"', '"')
    text = text.replace(''', "'")
    text = text.replace(''', "'")
    
    # 去除URL
    text = re.sub(r'https?://\S+', '', text)
    
    return text

def split_text_into_chunks(text: str, max_chunk_size: int = 1000) -> List[str]:
    """
    将文本分割成多个块
    
    Args:
        text: 文本
        max_chunk_size: 每个块的最大字符数
        
    Returns:
        文本块列表
    """
    # 如果文本长度小于最大块大小，直接返回
    if len(text) <= max_chunk_size:
        return [text]
    
    # 按句子分割
    sentences = re.split(r'([.!?。！？])', text)
    chunks = []
    current_chunk = ""
    
    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        
        # 如果当前句子后面有标点符号，加上它
        if i + 1 < len(sentences):
            sentence += sentences[i + 1]
        
        # 如果当前块加上新句子后超过最大块大小，保存当前块并开始新块
        if len(current_chunk) + len(sentence) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            
            # 如果单个句子超过最大块大小，则继续分割
            if len(sentence) > max_chunk_size:
                for j in range(0, len(sentence), max_chunk_size):
                    chunks.append(sentence[j:j + max_chunk_size])
            else:
                current_chunk = sentence
        else:
            current_chunk += sentence
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def save_json_data(data: Union[List, Dict], output_path: Union[str, Path]) -> None:
    """
    保存数据为JSON格式
    
    Args:
        data: 要保存的数据
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    
    # 确保目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"数据已保存到 {output_path}")

def save_csv_data(data: List[Dict], output_path: Union[str, Path]) -> None:
    """
    保存数据为CSV格式
    
    Args:
        data: 要保存的数据
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    
    # 确保目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    logger.info(f"数据已保存到 {output_path}") 
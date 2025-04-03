#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import requests
from openai import OpenAI

logger = logging.getLogger(__name__)

class DeepSeekAPIClient:
    """DeepSeek API客户端"""
    
    def __init__(self, api_key=None, model_name="deepseek-chat"):
        """
        初始化DeepSeek API客户端
        
        Args:
            api_key: DeepSeek API密钥，如果不提供，则从环境变量DEEPSEEK_API_KEY获取
            model_name: 使用的模型名称，默认为"deepseek-chat"
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            logger.warning("未提供DeepSeek API密钥，请设置环境变量DEEPSEEK_API_KEY或在初始化时提供")
        
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        
        logger.info(f"DeepSeek API客户端初始化完成，使用模型: {model_name}")
    
    def generate(self, prompt, max_tokens=1024, temperature=0.7, top_p=0.9):
        """
        生成文本
        
        Args:
            prompt: 提示文本
            max_tokens: 最大生成的token数
            temperature: 温度参数，控制随机性
            top_p: top-p采样参数
            
        Returns:
            生成的文本
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"调用DeepSeek API时出错: {str(e)}")
            raise e
    
    def generate_json(self, prompt, max_tokens=512, temperature=0.1):
        """
        生成JSON格式的文本
        
        Args:
            prompt: 提示文本
            max_tokens: 最大生成的token数
            temperature: 温度参数，控制随机性
            
        Returns:
            生成的JSON对象
        """
        try:
            # 为提示添加JSON格式的要求
            prompt += "\n请仅返回标准的JSON格式结果，确保所有引号匹配、逗号使用正确，不要有任何其他内容。"
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                response_format={"type": "json_object"}
            )
            
            text_response = response.choices[0].message.content
            logger.info(f"API响应原始文本: {text_response[:100]}...")
            
            # 尝试解析JSON
            try:
                return json.loads(text_response)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON解析错误: {str(e)}")
                
                # 尝试清理和修复JSON
                cleaned_text = text_response.strip()
                
                # 尝试提取JSON对象
                try:
                    import re
                    # 尝试匹配JSON对象
                    json_match = re.search(r'({.*})', cleaned_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        # 修复常见的JSON格式错误
                        json_str = re.sub(r',\s*}', '}', json_str)  # 删除对象末尾多余的逗号
                        json_str = re.sub(r',\s*]', ']', json_str)  # 删除数组末尾多余的逗号
                        return json.loads(json_str)
                    
                    # 尝试匹配JSON数组
                    array_match = re.search(r'(\[.*\])', cleaned_text, re.DOTALL)
                    if array_match:
                        array_str = array_match.group(1)
                        array_str = re.sub(r',\s*]', ']', array_str)  # 删除数组末尾多余的逗号
                        array_result = json.loads(array_str)
                        return {"entities": array_result}
                    
                    # 如果无法匹配，则手动构造简单的结果
                    logger.warning("无法从响应中提取有效JSON，构造默认实体列表")
                    
                    # 尝试手动提取实体名称和类型
                    entity_pattern = re.compile(r'"name"\s*:\s*"([^"]+)"\s*,\s*"type"\s*:\s*"([^"]+)"', re.DOTALL)
                    entities = []
                    for match in entity_pattern.finditer(cleaned_text):
                        name, entity_type = match.groups()
                        entities.append({"name": name, "type": entity_type})
                    
                    if entities:
                        return {"entities": entities}
                    
                    return {"entities": []}
                    
                except Exception as inner_e:
                    logger.error(f"尝试修复JSON时出错: {str(inner_e)}")
                    return {"entities": []}
        
        except Exception as e:
            logger.error(f"调用DeepSeek API生成JSON时出错: {str(e)}")
            return {"entities": []} 
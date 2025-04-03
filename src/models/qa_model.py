#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import networkx as nx
from typing import List, Dict, Tuple, Any, Optional
import json

logger = logging.getLogger(__name__)

class MedicalQAModel:
    """医学知识图谱问答模型"""
    
    def __init__(self, api_client, graph=None):
        """
        初始化医学问答模型
        
        Args:
            api_client: DeepSeek API客户端
            graph: 医学知识图谱 (NetworkX DiGraph)
        """
        self.api_client = api_client
        self.graph = graph
        
        # 定义医学实体类型
        self.entity_types = [
            "疾病", "症状", "药物", "治疗方法", "检查项目",
            "解剖部位", "病因", "并发症", "医院", "科室"
        ]
        
        # 定义医学关系类型
        self.relation_types = [
            "治疗", "预防", "导致", "检查", "诊断", "属于",
            "并发", "用于", "发生部位", "相关症状", "副作用"
        ]
        
        logger.info("医学问答模型初始化完成")
    
    def answer_question(self, question: str) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        回答医学问题
        
        Args:
            question: 用户问题
            
        Returns:
            答案, 相关实体列表, 相关关系列表
        """
        logger.info(f"处理问题: {question}")
        
        # 分析问题，识别关键实体和关系
        question_entities, question_relations = self._analyze_question(question)
        
        # 从知识图谱中检索相关信息
        kg_entities, kg_relations = self._retrieve_kg_information(question_entities, question_relations)
        
        # 构建提示
        prompt = self._build_prompt(question, kg_entities, kg_relations)
        
        # 使用DeepSeek API生成答案
        answer = self._generate_answer(prompt)
        
        # 返回答案和相关的实体、关系
        return answer, kg_entities, kg_relations
    
    def _analyze_question(self, question: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        分析问题，识别关键实体和关系
        
        Args:
            question: 用户问题
            
        Returns:
            识别的实体列表, 识别的关系列表
        """
        # 创建提示来识别问题中的医学实体和关系
        entity_types_str = ", ".join(self.entity_types)
        relation_types_str = ", ".join(self.relation_types)
        
        entity_prompt = f"""
从以下医学问题中识别出所有相关的医学实体以及它们的类型。
实体类型包括: {entity_types_str}

问题: {question}

输出格式 (JSON):
{{
    "entities": [
        {{"name": "实体名称", "type": "实体类型"}},
        ...
    ]
}}
"""
        
        relation_prompt = f"""
从以下医学问题中识别出所有相关的医学关系类型。
关系类型包括: {relation_types_str}

问题: {question}

输出格式 (单个字符串数组):
["关系类型1", "关系类型2", ...]
"""
        
        # 使用模型识别实体
        entities_json = self._generate_structured_text(entity_prompt)
        try:
            entities_data = json.loads(entities_json)
            entities = entities_data.get("entities", [])
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"解析实体JSON时出错: {e}")
            logger.error(f"原始JSON: {entities_json}")
            entities = []
        
        # 使用模型识别关系
        relations_json = self._generate_structured_text(relation_prompt)
        try:
            relations = json.loads(relations_json)
            if not isinstance(relations, list):
                relations = []
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"解析关系JSON时出错: {e}")
            logger.error(f"原始JSON: {relations_json}")
            relations = []
        
        logger.info(f"从问题中识别出 {len(entities)} 个实体和 {len(relations)} 种关系")
        
        return entities, relations
    
    def _retrieve_kg_information(
        self, 
        question_entities: List[Dict[str, Any]], 
        question_relations: List[str]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        从知识图谱中检索相关信息
        
        Args:
            question_entities: 问题中识别的实体列表
            question_relations: 问题中识别的关系列表
            
        Returns:
            知识图谱中的相关实体列表, 知识图谱中的相关关系列表
        """
        if not self.graph:
            logger.warning("知识图谱未加载，无法检索信息")
            return [], []
        
        retrieved_entities = []
        retrieved_relations = []
        
        # 检索与问题实体匹配的图谱节点
        for question_entity in question_entities:
            entity_name = question_entity.get("name", "")
            entity_type = question_entity.get("type", "")
            
            # 在图谱中查找匹配的实体节点
            matched_nodes = []
            for node, node_data in self.graph.nodes(data=True):
                node_name = node_data.get("name", "")
                node_type = node_data.get("type", "")
                
                # 如果节点名称包含实体名称，且类型相符（如果有指定类型）
                if (entity_name.lower() in node_name.lower() and 
                    (not entity_type or entity_type == node_type)):
                    matched_nodes.append((node, node_data))
            
            # 将匹配的节点添加到结果中
            for node_id, node_data in matched_nodes:
                retrieved_entity = {"id": node_id}
                retrieved_entity.update({k: v for k, v in node_data.items() 
                                       if isinstance(v, (str, int, float, bool))})
                retrieved_entities.append(retrieved_entity)
                
                # 检索与该节点相关的边
                for source, target, edge_data in self.graph.edges([node_id], data=True):
                    edge_type = edge_data.get("type", "")
                    
                    # 如果边的类型与问题中的关系类型匹配，添加到结果中
                    if not question_relations or edge_type in question_relations:
                        retrieved_relation = {
                            "source": source,
                            "target": target,
                            "source_name": self.graph.nodes[source].get("name", ""),
                            "target_name": self.graph.nodes[target].get("name", "")
                        }
                        retrieved_relation.update({k: v for k, v in edge_data.items() 
                                                 if isinstance(v, (str, int, float, bool))})
                        retrieved_relations.append(retrieved_relation)
        
        logger.info(f"从知识图谱中检索到 {len(retrieved_entities)} 个相关实体和 {len(retrieved_relations)} 个相关关系")
        
        return retrieved_entities, retrieved_relations
    
    def _build_prompt(
        self, 
        question: str, 
        kg_entities: List[Dict[str, Any]], 
        kg_relations: List[Dict[str, Any]]
    ) -> str:
        """
        构建提示文本
        
        Args:
            question: A用户问题
            kg_entities: 从知识图谱中检索的相关实体
            kg_relations: 从知识图谱中检索的相关关系
            
        Returns:
            用于生成答案的提示文本
        """
        # 格式化实体信息
        entities_text = ""
        for i, entity in enumerate(kg_entities[:5]):  # 限制实体数量，避免提示过长
            entity_text = f"{i+1}. {entity.get('name', 'Unknown')} (类型: {entity.get('type', 'Unknown')})"
            if "description" in entity:
                entity_text += f"\n   描述: {entity['description']}"
            entities_text += entity_text + "\n"
        
        # 格式化关系信息
        relations_text = ""
        for i, relation in enumerate(kg_relations[:5]):  # 限制关系数量
            source_name = relation.get("source_name", "Unknown")
            target_name = relation.get("target_name", "Unknown")
            relation_type = relation.get("type", "Unknown")
            
            relation_text = f"{i+1}. {source_name} --[{relation_type}]--> {target_name}"
            if "description" in relation:
                relation_text += f"\n   描述: {relation['description']}"
            relations_text += relation_text + "\n"
        
        # 构建最终提示
        prompt = f"""你是一个医学领域的AI助手，擅长回答与医学相关的问题。请基于以下从医学知识图谱中提取的信息来回答问题。

问题：{question}

相关医学实体：
{entities_text if entities_text else "未找到相关实体。"}

相关医学关系：
{relations_text if relations_text else "未找到相关关系。"}

请提供准确、专业的医学回答，内容应当简明扼要且易于理解。如果知识图谱中的信息不足以回答问题，请基于你的医学知识提供答案，但明确指出哪些内容是基于知识图谱，哪些是基于模型知识。

回答：
"""
        
        return prompt
    
    def _generate_structured_text(self, prompt: str) -> str:
        """
        使用DeepSeek API生成结构化文本（如JSON）
        
        Args:
            prompt: 提示文本
            
        Returns:
            生成的结构化文本
        """
        try:
            result = self.api_client.generate_json(prompt=prompt, max_tokens=256, temperature=0.1)
            return json.dumps(result)
        except Exception as e:
            logger.error(f"生成结构化文本时出错: {e}")
            return "{}"
    
    def _generate_answer(self, prompt: str) -> str:
        """
        使用DeepSeek API生成答案
        
        Args:
            prompt: 提示文本
            
        Returns:
            生成的答案
        """
        try:
            return self.api_client.generate(prompt=prompt, max_tokens=512, temperature=0.7, top_p=0.9)
        except Exception as e:
            logger.error(f"生成答案时出错: {e}")
            return "抱歉，我无法回答这个问题。" 
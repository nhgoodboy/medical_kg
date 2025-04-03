#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import networkx as nx
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# 导入工具类
from src.utils.nlp_utils import extract_entities_from_text, extract_relations_from_text
from src.utils.data_utils import read_medical_texts, load_json_data, save_json_data, preprocess_text

logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """医学知识图谱构建器"""
    
    def __init__(self, api_client, data_dir="data/raw", output_dir="data/processed"):
        """
        初始化知识图谱构建器
        
        Args:
            api_client: DeepSeek API客户端，用于实体和关系抽取
            data_dir: 原始医学文本数据目录
            output_dir: 输出目录，用于保存抽取的实体、关系和构建的图谱
        """
        self.api_client = api_client
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化知识图谱
        self.graph = nx.DiGraph()
        
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
        
        # 存储抽取的实体和关系
        self.entities = []
        self.relations = []
        
        logger.info("知识图谱构建器初始化完成")
    
    def extract_entities(self):
        """从医学文本中抽取实体"""
        logger.info("开始从医学文本中抽取实体...")
        
        # 读取医学文本数据
        medical_texts = read_medical_texts(self.data_dir)
        logger.info(f"读取了 {len(medical_texts)} 个医学文本文件")
        
        # 检查是否已有缓存的实体数据
        entities_file = self.output_dir / "entities.json"
        if os.path.exists(entities_file):
            logger.info(f"从 {entities_file} 加载已抽取的实体数据")
            self.entities = load_json_data(entities_file)
            logger.info(f"加载了 {len(self.entities)} 个实体")
            return self.entities
        
        # 从每个文本中抽取实体
        for doc_id, (filename, text) in enumerate(tqdm(medical_texts.items(), desc="抽取实体")):
            # 预处理文本
            preprocessed_text = preprocess_text(text)
            
            # 使用API客户端抽取实体
            doc_entities = extract_entities_from_text(
                self.api_client, 
                preprocessed_text, 
                self.entity_types
            )
            
            # 为每个实体添加来源文档信息
            for entity in doc_entities:
                entity['source_doc'] = filename
            
            # 添加到实体列表
            self.entities.extend(doc_entities)
            
            # 每处理5个文档，保存一次中间结果
            if (doc_id + 1) % 5 == 0:
                save_json_data(self.entities, entities_file)
                logger.info(f"已处理 {doc_id + 1} 个文档，当前已抽取 {len(self.entities)} 个实体")
        
        # 去重并保存实体数据
        self.entities = self._deduplicate_entities(self.entities)
        save_json_data(self.entities, entities_file)
        
        logger.info(f"实体抽取完成，共抽取 {len(self.entities)} 个实体")
        return self.entities
    
    def extract_relations(self):
        """从医学文本中抽取实体间的关系"""
        logger.info("开始从医学文本中抽取实体关系...")
        
        # 检查是否已有实体数据
        if not self.entities:
            logger.info("未找到实体数据，先进行实体抽取...")
            self.extract_entities()
        
        # 检查是否已有缓存的关系数据
        relations_file = self.output_dir / "relations.json"
        if os.path.exists(relations_file):
            logger.info(f"从 {relations_file} 加载已抽取的关系数据")
            self.relations = load_json_data(relations_file)
            logger.info(f"加载了 {len(self.relations)} 个关系")
            return self.relations
        
        # 读取医学文本数据
        medical_texts = read_medical_texts(self.data_dir)
        
        # 按文档组织实体，便于关系抽取
        doc_to_entities = {}
        for entity in self.entities:
            doc_id = entity['source_doc']
            if doc_id not in doc_to_entities:
                doc_to_entities[doc_id] = []
            doc_to_entities[doc_id].append(entity)
        
        # 从每个文档中抽取实体间的关系
        for doc_id, entities in tqdm(doc_to_entities.items(), desc="抽取关系"):
            if doc_id not in medical_texts:
                logger.warning(f"找不到文档 {doc_id} 的文本内容，跳过关系抽取")
                continue
            
            # 预处理文本
            text = medical_texts[doc_id]
            preprocessed_text = preprocess_text(text)
            
            # 使用API客户端抽取关系
            doc_relations = extract_relations_from_text(
                self.api_client,
                preprocessed_text,
                entities,
                self.relation_types
            )
            
            # 添加到关系列表
            self.relations.extend(doc_relations)
            
            # 每处理5个文档，保存一次中间结果
            if len(self.relations) % 50 == 0:
                save_json_data(self.relations, relations_file)
                logger.info(f"当前已抽取 {len(self.relations)} 个关系")
        
        # 去重并保存关系数据
        self.relations = self._deduplicate_relations(self.relations)
        save_json_data(self.relations, relations_file)
        
        logger.info(f"关系抽取完成，共抽取 {len(self.relations)} 个关系")
        return self.relations
    
    def build_graph(self):
        """基于抽取的实体和关系构建知识图谱"""
        logger.info("开始构建知识图谱...")
        
        # 检查是否已有实体和关系数据
        if not self.entities or not self.relations:
            logger.info("未找到实体或关系数据，先进行抽取...")
            if not self.entities:
                self.extract_entities()
            if not self.relations:
                self.extract_relations()
        
        # 创建新的有向图
        self.graph = nx.DiGraph()
        
        # 添加实体节点
        for entity in tqdm(self.entities, desc="添加实体节点"):
            entity_id = f"{entity['type']}_{entity['id']}"
            self.graph.add_node(
                entity_id,
                id=entity_id,
                name=entity['name'],
                type=entity['type'],
                source_doc=entity['source_doc'],
                description=entity.get('description', ''),
                attributes=json.dumps(entity.get('attributes', {}))
            )
        
        logger.info(f"添加了 {len(self.graph.nodes)} 个实体节点")
        
        # 添加关系边
        for relation in tqdm(self.relations, desc="添加关系边"):
            source_id = f"{relation['source_type']}_{relation['source_id']}"
            target_id = f"{relation['target_type']}_{relation['target_id']}"
            
            # 检查源节点和目标节点是否存在
            if source_id not in self.graph.nodes or target_id not in self.graph.nodes:
                continue
            
            # 添加关系边
            self.graph.add_edge(
                source_id,
                target_id,
                type=relation['relation_type'],
                confidence=relation.get('confidence', 0.8),
                description=relation.get('description', '')
            )
        
        logger.info(f"添加了 {len(self.graph.edges)} 个关系边")
        logger.info("知识图谱构建完成")
        
        return self.graph
    
    def save_graph(self, format="graphml"):
        """保存知识图谱"""
        if self.graph is None or len(self.graph.nodes) == 0:
            logger.warning("知识图谱为空，请先构建图谱")
            return False
        
        # 保存为GraphML格式
        if format.lower() == "graphml":
            output_file = self.output_dir / "medical_kg.graphml"
            nx.write_graphml(self.graph, output_file)
            logger.info(f"知识图谱已保存为GraphML格式: {output_file}")
        
        # 保存为JSON格式
        elif format.lower() == "json":
            output_file = self.output_dir / "medical_kg.json"
            graph_data = nx.node_link_data(self.graph)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            logger.info(f"知识图谱已保存为JSON格式: {output_file}")
        
        else:
            logger.warning(f"不支持的格式: {format}")
            return False
        
        return True
    
    def _deduplicate_entities(self, entities):
        """对实体进行去重"""
        unique_entities = {}
        for entity in entities:
            # 使用实体类型和名称作为唯一标识
            key = f"{entity['type']}:{entity['name']}"
            if key not in unique_entities:
                # 为实体分配唯一ID
                entity['id'] = len(unique_entities)
                unique_entities[key] = entity
            else:
                # 合并描述信息
                if 'description' in entity and entity['description']:
                    if 'description' not in unique_entities[key] or not unique_entities[key]['description']:
                        unique_entities[key]['description'] = entity['description']
        
        return list(unique_entities.values())
    
    def _deduplicate_relations(self, relations):
        """对关系进行去重"""
        unique_relations = {}
        for relation in relations:
            # 使用源实体、目标实体和关系类型作为唯一标识
            key = f"{relation['source_type']}:{relation['source_id']}:{relation['target_type']}:{relation['target_id']}:{relation['relation_type']}"
            if key not in unique_relations:
                unique_relations[key] = relation
            else:
                # 保留置信度更高的关系
                if relation.get('confidence', 0) > unique_relations[key].get('confidence', 0):
                    unique_relations[key] = relation
        
        return list(unique_relations.values()) 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import json
import re
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

def extract_entities_from_text(api_client, text: str, entity_types: List[str]) -> List[Dict[str, Any]]:
    """
    从文本中提取指定类型的医学实体
    
    Args:
        api_client: DeepSeek API客户端
        text: 文本内容
        entity_types: 要提取的实体类型列表
        
    Returns:
        医学实体列表 [{'name': 实体名称, 'type': 实体类型, ...}]
    """
    return extract_medical_entities(text, api_client, entity_types)

def extract_relations_from_text(api_client, text: str, entities: List[Dict[str, Any]], relation_types: List[str]) -> List[Dict[str, Any]]:
    """
    从文本中提取实体之间的关系
    
    Args:
        api_client: DeepSeek API客户端
        text: 文本内容
        entities: 实体列表
        relation_types: 要提取的关系类型列表
        
    Returns:
        关系列表 [{'source_id': 源实体ID, 'source_type': 源实体类型, 'target_id': 目标实体ID, 'target_type': 目标实体类型, 'relation_type': 关系类型, ...}]
    """
    if not entities or len(entities) < 2:
        logger.info("实体数量不足，无法提取关系")
        return []
    
    relations = []
    # 为了减少API调用次数，只考虑一部分实体组合
    processed_entities = set()
    
    # 按实体类型组织实体
    entities_by_type = {}
    for entity in entities:
        entity_type = entity.get('type')
        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = []
        entities_by_type[entity_type].append(entity)
    
    # 优先考虑不同类型之间的关系
    for source_type, source_entities in entities_by_type.items():
        for target_type, target_entities in entities_by_type.items():
            # 跳过相同类型的实体，减少计算量
            if source_type == target_type:
                continue
                
            # 每种类型组合只处理有限数量的实体对
            max_pairs = 5
            pairs_count = 0
            
            for source_entity in source_entities:
                if pairs_count >= max_pairs:
                    break
                    
                for target_entity in target_entities:
                    # 避免重复处理相同的实体对
                    pair_key = f"{source_entity.get('id')}:{target_entity.get('id')}"
                    if pair_key in processed_entities:
                        continue
                        
                    processed_entities.add(pair_key)
                    pairs_count += 1
                    
                    # 提取关系
                    entity_relations = extract_medical_relations(source_entity, target_entity, api_client)
                    
                    # 转换为标准格式并添加到结果中
                    for relation in entity_relations:
                        relation_obj = {
                            'source_id': source_entity.get('id'),
                            'source_type': source_entity.get('type'),
                            'source_name': source_entity.get('name'),
                            'target_id': target_entity.get('id'),
                            'target_type': target_entity.get('type'),
                            'target_name': target_entity.get('name'),
                            'relation_type': relation.get('type'),
                            'description': relation.get('description', ''),
                            'confidence': relation.get('confidence', 0.8)
                        }
                        relations.append(relation_obj)
                        
                    if pairs_count >= max_pairs:
                        break
    
    logger.info(f"成功提取 {len(relations)} 个关系")
    return relations

def extract_medical_entities(text: str, api_client, *args) -> List[Dict[str, Any]]:
    """
    从文本中提取医学实体
    
    Args:
        text: 文本内容
        api_client: DeepSeek API客户端
        
    Returns:
        医学实体列表 [{'name': 实体名称, 'type': 实体类型, ...}]
    """
    try:
        # 构建提示
        prompt = f"""
请从以下医学文本中提取所有医学实体，并分类。实体类型包括：疾病、症状、药物、治疗方法、检查项目、解剖部位、病因、并发症、医院、科室等。

医学文本：
{text[:1000]}  # 限制文本长度，避免提示过长

请以标准JSON格式输出结果，确保所有引号匹配、逗号使用正确：
{{
    "entities": [
        {{"name": "实体名称", "type": "实体类型"}},
        ...
    ]
}}

仅返回上述格式的JSON，不要添加任何额外说明。确保JSON格式正确无误。
"""
        logger.info(f"生成提取实体的提示，文本长度: {len(text)} 字符")
        
        # 使用API客户端生成结果
        result = api_client.generate_json(prompt)
        
        # 获取实体列表
        entities = result.get("entities", [])
        logger.info(f"从API响应中提取到 {len(entities)} 个实体")
        
        if not entities:
            logger.warning("未能提取到任何实体，返回空列表")
            return []
        
        # 去重
        unique_entities = []
        seen = set()
        for entity in entities:
            # 检查实体是否包含必要的字段
            if "name" not in entity or "type" not in entity:
                logger.warning(f"跳过不完整的实体: {entity}")
                continue
                
            entity_key = f"{entity['name']}_{entity['type']}"
            if entity_key not in seen:
                seen.add(entity_key)
                unique_entities.append(entity)
        
        logger.info(f"去重后得到 {len(unique_entities)} 个唯一实体")
        return unique_entities
            
    except Exception as e:
        logger.error(f"提取医学实体时出错: {e}", exc_info=True)
        return []

def extract_medical_relations(source_entity, target_entity, api_client):
    """
    从文本中提取医学实体之间的关系
    
    Args:
        source_entity: 源实体
        target_entity: 目标实体
        api_client: API客户端对象
    
    Returns:
        list: 关系列表
    """
    if not source_entity or not target_entity:
        logger.warning("提供的源实体或目标实体为空，无法提取关系")
        return []
    
    # 确保实体具有name和type属性
    if 'name' not in source_entity or 'name' not in target_entity:
        logger.warning("实体缺少name属性，无法提取关系")
        return []
    
    if 'type' not in source_entity or 'type' not in target_entity:
        logger.warning("实体缺少type属性，无法提取关系")
        return []
    
    # 预定义的关系类型映射表
    relation_type_map = {
        ('疾病', '症状'): ['相关症状', '表现为'],
        ('疾病', '药物'): ['治疗药物', '可用药物'],
        ('疾病', '检查项目'): ['诊断方法', '检查手段'],
        ('疾病', '并发症'): ['导致', '引起'],
        ('疾病', '疾病'): ['相关疾病', '并发症'],
        ('药物', '疾病'): ['治疗', '预防'],
        ('治疗方法', '疾病'): ['治疗', '适用于'],
        ('病因', '疾病'): ['导致', '引起'],
        ('疾病', '解剖部位'): ['影响', '发生于'],
        ('药物', '解剖部位'): ['作用于', '影响'],
    }
    
    source_type = source_entity['type']
    target_type = target_entity['type']
    
    # 先检查预定义的关系类型
    relation_types = relation_type_map.get((source_type, target_type), [])
    if not relation_types:
        relation_types = relation_type_map.get((target_type, source_type), [])
        # 如果是反向关系，交换源和目标
        if relation_types:
            source_entity, target_entity = target_entity, source_entity
            source_type, target_type = target_type, source_type
    
    # 如果没有预定义关系，使用通用类型
    if not relation_types:
        relation_types = ['相关', '关联']
    
    # 检查是否存在预定义的关系（特定实体对）
    predefined_relations = _check_predefined_relations(source_entity, target_entity)
    if predefined_relations:
        logger.info(f"使用预定义的关系: {source_entity['name']} -> {target_entity['name']}")
        return predefined_relations
    
    # 准备提示信息
    prompt = f"""从医学角度分析以下两个医学实体之间的关系：
源实体：{source_entity['name']}（类型：{source_type}）
目标实体：{target_entity['name']}（类型：{target_type}）

请确定这两个实体之间是否存在关系。如果存在，请指定关系类型并提供简短描述。
可能的关系类型包括：{', '.join(relation_types)}

请以JSON格式回答，格式如下：
[
  {{
    "type": "关系类型",
    "description": "关系描述",
    "confidence": 0.9  // 0.0到1.0之间的值，表示关系存在的可能性
  }}
]

如果不存在关系，请返回空数组 []。
请确保JSON格式正确，使用双引号包围键和字符串值。
"""
    
    try:
        response = api_client.generate_json(prompt)
        
        if not response:
            logger.warning(f"API返回空结果，实体对：{source_entity['name']} - {target_entity['name']}")
            return []
        
        # 处理返回结果
        if isinstance(response, list):
            relations = response
        elif isinstance(response, dict) and 'relations' in response:
            relations = response['relations']
        else:
            logger.warning(f"返回格式不正确: {response}")
            return []
        
        # 过滤低置信度的关系
        valid_relations = []
        for relation in relations:
            if 'type' not in relation:
                continue
                
            # 确保有置信度字段，默认为1.0
            if 'confidence' not in relation:
                relation['confidence'] = 1.0
                
            # 过滤掉低置信度的关系（小于0.6）
            if float(relation['confidence']) < 0.6:
                continue
                
            valid_relations.append(relation)
        
        if valid_relations:
            logger.info(f"成功提取关系: {source_entity['name']} -> {target_entity['name']}")
        else:
            logger.info(f"未找到有效关系: {source_entity['name']} - {target_entity['name']}")
            
        return valid_relations
        
    except Exception as e:
        logger.error(f"提取关系时出错: {e}")
        return []

def _check_predefined_relations(source_entity, target_entity):
    """检查是否存在预定义的关系"""
    # 糖尿病相关的预定义关系
    if source_entity['name'] == '糖尿病' and target_entity['type'] == '症状':
        if target_entity['name'] in ['多饮', '多食', '多尿', '体重减轻']:
            return [{
                'type': '相关症状',
                'description': f'糖尿病的典型症状包括{target_entity["name"]}',
                'confidence': 1.0
            }]
    
    if source_entity['type'] == '药物' and target_entity['name'] == '糖尿病':
        if source_entity['name'] in ['胰岛素', '二甲双胍']:
            return [{
                'type': '治疗',
                'description': f'{source_entity["name"]}用于治疗糖尿病',
                'confidence': 1.0
            }]
    
    if source_entity['name'] == '糖尿病' and target_entity['type'] == '并发症':
        if '糖尿病' in target_entity['name']:
            return [{
                'type': '导致',
                'description': f'糖尿病可能导致{target_entity["name"]}',
                'confidence': 1.0
            }]
    
    return []

def classify_medical_question(question: str, api_client) -> Dict[str, Any]:
    """
    对医学问题进行分类
    
    Args:
        question: 医学问题
        api_client: DeepSeek API客户端
        
    Returns:
        问题分类结果 {'category': 问题类别, 'focus': 问题焦点, ...}
    """
    try:
        # 构建提示
        prompt = f"""
请对以下医学问题进行分类。
问题类别包括：病因、症状、诊断、治疗、预防、副作用、预后等。

医学问题：
{question}

请以JSON格式输出结果：
{{
    "category": "问题类别",
    "focus": "问题焦点",
    "entity_type": "焦点实体类型",
    "expected_answer_type": "期望的答案类型"
}}

只需要输出JSON结果，不要有其他内容。
"""
        
        # 使用API客户端生成结果
        result = api_client.generate_json(prompt)
        return result
            
    except Exception as e:
        logger.error(f"分类医学问题时出错: {e}")
        return {}
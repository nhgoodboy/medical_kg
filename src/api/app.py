#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import networkx as nx
import sys
from flask import Flask, request, jsonify, render_template

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.models.qa_model import MedicalQAModel
from src.utils.api_client import DeepSeekAPIClient

logger = logging.getLogger(__name__)

def create_app(kg_path="data/processed/medical_kg.graphml", api_key=None, model_name="deepseek-chat"):
    """创建Flask应用程序"""
    app = Flask(__name__)
    
    # 加载知识图谱
    graph = None
    if os.path.exists(kg_path):
        logger.info(f"加载知识图谱: {kg_path}")
        graph = nx.read_graphml(kg_path)
        logger.info(f"知识图谱加载完成: {len(graph.nodes)} 节点, {len(graph.edges)} 边")
    else:
        logger.warning(f"知识图谱文件不存在: {kg_path}")
    
    # 创建DeepSeek API客户端
    logger.info(f"初始化DeepSeek API客户端, 使用模型: {model_name}")
    api_client = DeepSeekAPIClient(api_key=api_key, model_name=model_name)
    
    # 创建问答模型
    qa_model = MedicalQAModel(api_client, graph)
    
    @app.route('/')
    def index():
        """主页"""
        return render_template('index.html')
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """健康检查接口"""
        return jsonify({
            'status': 'ok',
            'kg_loaded': graph is not None,
            'model_loaded': api_client is not None,
            'nodes_count': len(graph.nodes) if graph else 0,
            'edges_count': len(graph.edges) if graph else 0
        })
    
    @app.route('/api/query', methods=['POST'])
    def query():
        """问答查询接口"""
        if not request.json or 'question' not in request.json:
            return jsonify({'error': '请提供问题内容'}), 400
        
        question = request.json['question']
        logger.info(f"收到问答请求: {question}")
        
        try:
            # 使用问答模型生成答案
            answer, entities, relations = qa_model.answer_question(question)
            
            response = {
                'question': question,
                'answer': answer,
                'related_entities': entities,
                'related_relations': relations
            }
            
            return jsonify(response)
        
        except Exception as e:
            logger.error(f"处理问题时出错: {str(e)}", exc_info=True)
            return jsonify({'error': f'处理问题时出错: {str(e)}'}), 500
    
    @app.route('/api/entities', methods=['GET'])
    def get_entities():
        """获取实体列表接口"""
        if not graph:
            return jsonify({'error': '知识图谱未加载'}), 500
        
        entity_type = request.args.get('type')
        limit = int(request.args.get('limit', 100))
        
        entities = []
        for node, data in graph.nodes(data=True):
            if entity_type and data.get('type') != entity_type:
                continue
            
            entity_data = {'id': node}
            entity_data.update({k: v for k, v in data.items() 
                              if isinstance(v, (str, int, float, bool))})
            entities.append(entity_data)
            
            if len(entities) >= limit:
                break
        
        return jsonify({'entities': entities, 'count': len(entities)})
    
    @app.route('/api/entity/<entity_id>', methods=['GET'])
    def get_entity(entity_id):
        """获取特定实体接口"""
        if not graph:
            return jsonify({'error': '知识图谱未加载'}), 500
            
        if entity_id not in graph.nodes:
            return jsonify({'error': f'实体不存在: {entity_id}'}), 404
            
        # 获取实体信息
        entity_data = {'id': entity_id}
        entity_data.update(graph.nodes[entity_id])
        
        # 获取相关的关系
        relations = []
        
        # 获取出边
        for _, target, data in graph.out_edges(entity_id, data=True):
            relation = {
                'source': entity_id,
                'target': target,
                'direction': 'outgoing'
            }
            relation.update({k: v for k, v in data.items() 
                           if isinstance(v, (str, int, float, bool))})
            relations.append(relation)
        
        # 获取入边
        for source, _, data in graph.in_edges(entity_id, data=True):
            relation = {
                'source': source,
                'target': entity_id,
                'direction': 'incoming'
            }
            relation.update({k: v for k, v in data.items() 
                           if isinstance(v, (str, int, float, bool))})
            relations.append(relation)
        
        return jsonify({
            'entity': entity_data,
            'relations': relations
        })
    
    logger.info("API应用创建完成")
    return app 
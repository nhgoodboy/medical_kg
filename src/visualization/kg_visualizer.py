#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import json
from pathlib import Path
from pyvis.network import Network
import plotly.graph_objects as go
import pandas as pd
import platform
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont
import math
import io

# 设置中文字体，确保中文显示正常
system = platform.system()
if system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'STHeiti', 'Heiti TC', 'sans-serif']
elif system == 'Windows':  # Windows
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
else:  # Linux 或其他
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'sans-serif']
    
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

class KnowledgeGraphVisualizer:
    """医学知识图谱可视化工具"""
    
    def __init__(self, graph=None, graph_path=None, output_dir="data/visualization"):
        """
        初始化可视化器
        
        Args:
            graph: NetworkX图对象
            graph_path: 图谱文件路径(.graphml或.json)
            output_dir: 可视化结果输出目录
        """
        self.graph = graph
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 如果提供了图谱文件路径，则加载图谱
        if graph_path and not self.graph:
            self.load_graph(graph_path)
            
        # 实体类型到颜色的映射
        self.entity_colors = {
            "疾病": "#FF5733",
            "症状": "#33FF57",
            "药物": "#3357FF",
            "治疗方法": "#FF33A8",
            "检查项目": "#33A8FF",
            "解剖部位": "#A833FF",
            "病因": "#FFA833",
            "并发症": "#FF3333",
            "医院": "#33FFFF",
            "科室": "#FFFF33"
        }
        
        # 关系类型到颜色的映射
        self.relation_colors = {
            "治疗": "#FF5733",
            "预防": "#33FF57",
            "导致": "#3357FF",
            "检查": "#FF33A8",
            "诊断": "#33A8FF",
            "属于": "#A833FF",
            "并发": "#FFA833",
            "用于": "#FF3333",
            "发生部位": "#33FFFF",
            "相关症状": "#FFFF33",
            "副作用": "#8C33FF"
        }
        
        logger.info("知识图谱可视化器初始化完成")
    
    def load_graph(self, graph_path: Union[str, Path]):
        """
        加载知识图谱
        
        Args:
            graph_path: 知识图谱文件路径
        """
        graph_path = Path(graph_path)
        
        if not graph_path.exists():
            logger.error(f"图谱文件不存在: {graph_path}")
            return False
        
        try:
            # 根据文件扩展名选择加载方式
            if graph_path.suffix.lower() == ".graphml":
                self.graph = nx.read_graphml(graph_path)
                logger.info(f"成功加载GraphML格式图谱: {graph_path}")
                
            elif graph_path.suffix.lower() == ".json":
                with open(graph_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.graph = nx.node_link_graph(data)
                logger.info(f"成功加载JSON格式图谱: {graph_path}")
                
            else:
                logger.error(f"不支持的文件格式: {graph_path.suffix}")
                return False
                
            logger.info(f"图谱信息: {len(self.graph.nodes)} 节点, {len(self.graph.edges)} 边")
            return True
            
        except Exception as e:
            logger.error(f"加载图谱失败: {e}")
            return False
    
    def visualize_with_matplotlib(self, title="医学知识图谱", figsize=(20, 16), 
                                 output_file="medical_kg_matplotlib.png", 
                                 node_size=300, edge_width=1.0, 
                                 max_nodes=100, layout='spring'):
        """
        使用Matplotlib可视化知识图谱
        
        Args:
            title: 图表标题
            figsize: 图表大小
            output_file: 输出文件名
            node_size: 节点大小
            edge_width: 边的宽度
            max_nodes: 最大显示节点数
            layout: 布局算法 ('spring', 'circular', 'random', 'shell', 'kamada_kawai')
        """
        if not self.graph:
            logger.error("图谱未加载，无法可视化")
            return False
        
        # 如果节点过多，随机选择一部分节点
        if len(self.graph.nodes) > max_nodes:
            logger.info(f"节点数量 ({len(self.graph.nodes)}) 超过最大限制 ({max_nodes})，将随机选择子图")
            nodes = list(self.graph.nodes())
            selected_nodes = np.random.choice(nodes, max_nodes, replace=False)
            subgraph = self.graph.subgraph(selected_nodes)
        else:
            subgraph = self.graph
        
        # 创建图表
        plt.figure(figsize=figsize)
        plt.title(title, fontsize=20)
        
        # 选择布局算法
        if layout == 'spring':
            pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(subgraph)
        elif layout == 'random':
            pos = nx.random_layout(subgraph)
        elif layout == 'shell':
            pos = nx.shell_layout(subgraph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(subgraph)
        else:
            logger.warning(f"未知的布局算法: {layout}，使用默认的spring布局")
            pos = nx.spring_layout(subgraph)
        
        # 按实体类型绘制节点
        for entity_type, color in self.entity_colors.items():
            node_list = [node for node, data in subgraph.nodes(data=True) 
                         if data.get('type', '') == entity_type]
            
            if node_list:
                nx.draw_networkx_nodes(
                    subgraph, pos,
                    nodelist=node_list,
                    node_color=color,
                    node_size=node_size,
                    alpha=0.8,
                    label=entity_type
                )
        
        # 按关系类型绘制边
        for relation_type, color in self.relation_colors.items():
            edge_list = [(u, v) for u, v, data in subgraph.edges(data=True) 
                         if data.get('type', '') == relation_type]
            
            if edge_list:
                nx.draw_networkx_edges(
                    subgraph, pos,
                    edgelist=edge_list,
                    width=edge_width,
                    alpha=0.6,
                    edge_color=color,
                    label=relation_type
                )
        
        # 绘制节点标签
        node_labels = {node: data.get('name', node) 
                      for node, data in subgraph.nodes(data=True)}
        
        # 使用系统适配的字体
        system = platform.system()
        if system == 'Darwin':  # macOS
            font_family = 'Arial Unicode MS'
        elif system == 'Windows':  # Windows
            font_family = 'Microsoft YaHei'
        else:  # Linux 或其他
            font_family = 'DejaVu Sans'
            
        nx.draw_networkx_labels(subgraph, pos, 
                               labels=node_labels, 
                               font_size=8, 
                               font_weight='bold',
                               font_family=font_family)
        
        # 添加图例
        plt.legend(loc='upper right', fontsize=10)
        plt.axis('off')
        
        # 保存图表
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"图谱已保存为Matplotlib图像: {output_path}")
        return True
    
    def visualize_with_pyvis(self, output_file="medical_kg_interactive.html", 
                           height="800px", width="100%", 
                           bgcolor="#ffffff", max_nodes=500,
                           show_buttons=True, notebook=False):
        """
        使用PyVis生成交互式知识图谱可视化
        
        Args:
            output_file: 输出HTML文件名
            height: 可视化高度
            width: 可视化宽度
            bgcolor: 背景颜色
            max_nodes: 最大显示节点数
            show_buttons: 是否显示控制按钮
            notebook: 是否在Jupyter Notebook中显示
        """
        if not self.graph:
            logger.error("图谱未加载，无法可视化")
            return False
        
        # 如果节点过多，随机选择一部分节点
        if len(self.graph.nodes) > max_nodes:
            logger.info(f"节点数量 ({len(self.graph.nodes)}) 超过最大限制 ({max_nodes})，将随机选择子图")
            nodes = list(self.graph.nodes())
            selected_nodes = np.random.choice(nodes, max_nodes, replace=False)
            subgraph = self.graph.subgraph(selected_nodes)
        else:
            subgraph = self.graph
        
        # 创建Pyvis网络
        net = Network(height=height, width=width, bgcolor=bgcolor, 
                     font_color="black", notebook=notebook)
        
        # 设置物理布局选项
        net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=150, 
                      spring_strength=0.05, damping=0.09)
        
        # 添加节点
        for node, data in subgraph.nodes(data=True):
            node_type = data.get('type', '')
            node_name = data.get('name', str(node))
            node_desc = data.get('description', '')
            
            node_color = self.entity_colors.get(node_type, "#CCCCCC")
            
            # 添加节点并设置属性
            net.add_node(
                node,
                label=node_name,
                title=f"类型: {node_type}<br>描述: {node_desc}",
                color=node_color,
                shape="dot" if node_type == "疾病" else "ellipse",
                size=20 if node_type == "疾病" else 15
            )
        
        # 添加边
        for source, target, data in subgraph.edges(data=True):
            relation_type = data.get('type', '')
            relation_desc = data.get('description', '')
            confidence = data.get('confidence', 1.0)
            
            edge_color = self.relation_colors.get(relation_type, "#999999")
            edge_width = 1 + 2 * float(confidence)  # 根据置信度调整边的宽度
            
            # 添加边并设置属性
            net.add_edge(
                source, target,
                title=f"关系: {relation_type}<br>描述: {relation_desc}<br>置信度: {confidence:.2f}",
                color=edge_color,
                width=edge_width,
                arrowStrikethrough=True,
                smooth={'type': 'continuous'}
            )
        
        # 设置显示选项
        if show_buttons:
            net.show_buttons(filter_=['physics', 'nodes', 'edges'])
        
        # 保存为HTML文件
        output_path = self.output_dir / output_file
        net.save_graph(str(output_path))
        
        logger.info(f"交互式图谱已保存为: {output_path}")
        return True
    
    def visualize_with_plotly(self, output_file="medical_kg_plotly.html", 
                             max_nodes=300, title="医学知识图谱"):
        """
        使用Plotly生成交互式知识图谱可视化
        
        Args:
            output_file: 输出HTML文件名
            max_nodes: 最大显示节点数
            title: 图表标题
        """
        if not self.graph:
            logger.error("图谱未加载，无法可视化")
            return False
        
        # 如果节点过多，随机选择一部分节点
        if len(self.graph.nodes) > max_nodes:
            logger.info(f"节点数量 ({len(self.graph.nodes)}) 超过最大限制 ({max_nodes})，将随机选择子图")
            nodes = list(self.graph.nodes())
            selected_nodes = np.random.choice(nodes, max_nodes, replace=False)
            subgraph = self.graph.subgraph(selected_nodes)
        else:
            subgraph = self.graph
        
        # 使用Force Atlas 2布局算法
        pos = nx.spring_layout(subgraph, k=0.2, iterations=50)
        
        # 准备节点数据
        node_traces = []
        
        # 按实体类型分组创建节点
        entity_type_groups = {}
        for node, data in subgraph.nodes(data=True):
            entity_type = data.get('type', 'unknown')
            if entity_type not in entity_type_groups:
                entity_type_groups[entity_type] = []
            entity_type_groups[entity_type].append((node, data))
        
        # 为每种实体类型创建一个trace
        for entity_type, nodes_data in entity_type_groups.items():
            x = []
            y = []
            node_names = []
            node_descs = []
            node_ids = []
            
            for node, data in nodes_data:
                x.append(pos[node][0])
                y.append(pos[node][1])
                node_names.append(data.get('name', str(node)))
                node_descs.append(data.get('description', ''))
                node_ids.append(node)
            
            node_trace = go.Scatter(
                x=x, y=y,
                mode='markers',
                hoverinfo='text',
                name=entity_type,
                marker=dict(
                    showscale=False,
                    color=self.entity_colors.get(entity_type, "#CCCCCC"),
                    size=15,
                    line_width=2
                ),
                text=[f"ID: {nid}<br>名称: {name}<br>类型: {entity_type}<br>描述: {desc}" 
                     for nid, name, desc in zip(node_ids, node_names, node_descs)],
                ids=node_ids
            )
            
            node_traces.append(node_trace)
        
        # 准备边数据
        edge_traces = []
        
        # 按关系类型分组创建边
        relation_type_groups = {}
        for source, target, data in subgraph.edges(data=True):
            relation_type = data.get('type', 'unknown')
            if relation_type not in relation_type_groups:
                relation_type_groups[relation_type] = []
            relation_type_groups[relation_type].append((source, target, data))
        
        # 为每种关系类型创建一个trace
        for relation_type, edges_data in relation_type_groups.items():
            edge_x = []
            edge_y = []
            edge_text = []
            
            for source, target, data in edges_data:
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                
                # 绘制边为曲线（添加中间点）
                edge_x.append(x0)
                edge_x.append((x0 + x1) / 2)
                edge_x.append(x1)
                edge_x.append(None)  # 添加断点
                
                edge_y.append(y0)
                edge_y.append((y0 + y1) / 2 + 0.05)  # 稍微向上弯曲
                edge_y.append(y1)
                edge_y.append(None)  # 添加断点
                
                # 边的提示信息
                confidence = data.get('confidence', 1.0)
                description = data.get('description', '')
                source_data = subgraph.nodes[source]
                target_data = subgraph.nodes[target]
                
                text = f"关系: {relation_type}<br>" \
                       f"源: {source_data.get('name', source)}<br>" \
                       f"目标: {target_data.get('name', target)}<br>" \
                       f"描述: {description}<br>" \
                       f"置信度: {confidence:.2f}"
                
                # 为边的每个片段添加相同的文本
                edge_text.extend([text, text, text, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color=self.relation_colors.get(relation_type, "#999999")),
                hoverinfo='text',
                text=edge_text,
                mode='lines',
                name=relation_type
            )
            
            edge_traces.append(edge_trace)
        
        # 创建图表布局
        layout = go.Layout(
            title=title,
            showlegend=True,
            legend=dict(x=1.05, y=0.5),
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template='plotly_white',
            height=800
        )
        
        # 创建图表
        fig = go.Figure(data=edge_traces + node_traces, layout=layout)
        
        # 保存为HTML文件
        output_path = self.output_dir / output_file
        fig.write_html(str(output_path), include_plotlyjs='cdn')
        
        logger.info(f"Plotly交互式图谱已保存为: {output_path}")
        return True
    
    def create_entity_subgraph(self, entity_name: str, depth: int = 2, max_nodes: int = 100):
        """
        创建以特定实体为中心的子图
        
        Args:
            entity_name: 实体名称
            depth: 遍历深度
            max_nodes: 最大节点数
            
        Returns:
            NetworkX子图
        """
        if not self.graph:
            logger.error("图谱未加载，无法创建子图")
            return None
        
        # 查找匹配实体名称的节点
        matched_nodes = []
        for node, data in self.graph.nodes(data=True):
            if data.get('name') == entity_name:
                matched_nodes.append(node)
        
        if not matched_nodes:
            logger.warning(f"未找到名称为 '{entity_name}' 的实体")
            return None
        
        # 使用第一个匹配的节点
        start_node = matched_nodes[0]
        logger.info(f"以实体 '{entity_name}' (ID: {start_node}) 为中心创建子图")
        
        # BFS遍历图，最大深度为depth
        nodes_to_include = set([start_node])
        current_nodes = set([start_node])
        
        for d in range(depth):
            new_nodes = set()
            for node in current_nodes:
                # 获取邻居节点
                neighbors = set(self.graph.predecessors(node)) | set(self.graph.successors(node))
                new_nodes.update(neighbors)
                
                # 如果节点总数超过最大限制，则停止添加
                if len(nodes_to_include) + len(new_nodes) > max_nodes:
                    # 随机选择一部分新节点
                    remaining = max_nodes - len(nodes_to_include)
                    if remaining > 0:
                        new_nodes = set(list(new_nodes)[:remaining])
                    break
            
            nodes_to_include.update(new_nodes)
            current_nodes = new_nodes
            
            if len(nodes_to_include) >= max_nodes:
                break
        
        # 创建子图
        subgraph = self.graph.subgraph(nodes_to_include)
        logger.info(f"创建了包含 {len(subgraph.nodes)} 个节点和 {len(subgraph.edges)} 条边的子图")
        
        return subgraph
    
    def export_to_d3_json(self, output_file="medical_kg_d3.json", max_nodes=1000):
        """
        导出图谱为D3.js可用的JSON格式
        
        Args:
            output_file: 输出JSON文件名
            max_nodes: 最大节点数
        """
        if not self.graph:
            logger.error("图谱未加载，无法导出")
            return False
        
        # 如果节点过多，随机选择一部分节点
        if len(self.graph.nodes) > max_nodes:
            logger.info(f"节点数量 ({len(self.graph.nodes)}) 超过最大限制 ({max_nodes})，将随机选择子图")
            nodes = list(self.graph.nodes())
            selected_nodes = np.random.choice(nodes, max_nodes, replace=False)
            subgraph = self.graph.subgraph(selected_nodes)
        else:
            subgraph = self.graph
        
        # 转换为D3.js格式
        nodes = []
        node_id_map = {}  # 将原始节点ID映射到数组索引
        
        for i, (node, data) in enumerate(subgraph.nodes(data=True)):
            node_type = data.get('type', '')
            node_id_map[node] = i
            
            nodes.append({
                "id": i,
                "name": data.get('name', str(node)),
                "type": node_type,
                "description": data.get('description', ''),
                "group": list(self.entity_colors.keys()).index(node_type) if node_type in self.entity_colors else 0,
                "color": self.entity_colors.get(node_type, "#CCCCCC")
            })
        
        links = []
        for source, target, data in subgraph.edges(data=True):
            relation_type = data.get('type', '')
            
            links.append({
                "source": node_id_map[source],
                "target": node_id_map[target],
                "type": relation_type,
                "value": float(data.get('confidence', 1.0)),
                "description": data.get('description', ''),
                "color": self.relation_colors.get(relation_type, "#999999")
            })
        
        d3_data = {"nodes": nodes, "links": links}
        
        # 保存为JSON文件
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(d3_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"D3.js格式图谱已保存为: {output_path}")
        return True
    
    def generate_statistics(self, output_file="medical_kg_stats.json"):
        """
        生成图谱统计信息
        
        Args:
            output_file: 输出JSON文件名
        """
        if not self.graph:
            logger.error("图谱未加载，无法生成统计信息")
            return False
        
        stats = {
            "总节点数": len(self.graph.nodes),
            "总边数": len(self.graph.edges),
            "平均度": sum(dict(self.graph.degree()).values()) / len(self.graph.nodes) if len(self.graph.nodes) > 0 else 0,
            "实体类型分布": {},
            "关系类型分布": {},
            "度数最高的实体": [],
            "中心性最高的实体": []
        }
        
        # 统计实体类型分布
        entity_type_counts = {}
        for node, data in self.graph.nodes(data=True):
            entity_type = data.get('type', 'unknown')
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
        
        stats["实体类型分布"] = entity_type_counts
        
        # 统计关系类型分布
        relation_type_counts = {}
        for source, target, data in self.graph.edges(data=True):
            relation_type = data.get('type', 'unknown')
            relation_type_counts[relation_type] = relation_type_counts.get(relation_type, 0) + 1
        
        stats["关系类型分布"] = relation_type_counts
        
        # 找出度数最高的实体
        degree_centrality = nx.degree_centrality(self.graph)
        top_degree_entities = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for node, centrality in top_degree_entities:
            node_data = self.graph.nodes[node]
            stats["度数最高的实体"].append({
                "id": node,
                "name": node_data.get('name', str(node)),
                "type": node_data.get('type', ''),
                "centrality": centrality,
                "degree": self.graph.degree(node)
            })
        
        # 找出中心性最高的实体
        try:
            betweenness_centrality = nx.betweenness_centrality(self.graph, k=min(100, len(self.graph.nodes)))
            top_betweenness_entities = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            
            for node, centrality in top_betweenness_entities:
                node_data = self.graph.nodes[node]
                stats["中心性最高的实体"].append({
                    "id": node,
                    "name": node_data.get('name', str(node)),
                    "type": node_data.get('type', ''),
                    "betweenness_centrality": centrality
                })
        except:
            logger.warning("计算中心性时出错，可能是图不连通")
        
        # 保存统计信息
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"图谱统计信息已保存为: {output_path}")
        return stats
    
    def visualize_with_pil(self, output_file="medical_kg_pil.png", 
                         width=2000, height=1600, 
                         max_nodes=100, node_size=15,
                         background_color=(255, 255, 255),
                         title="医学知识图谱"):
        """
        使用PIL库直接生成知识图谱图像，避免matplotlib的字体问题
        
        Args:
            output_file: 输出文件名
            width: 图像宽度
            height: 图像高度
            max_nodes: 最大显示节点数
            node_size: 节点大小
            background_color: 背景颜色 (R,G,B)
            title: 图表标题
        """
        if not self.graph:
            logger.error("图谱未加载，无法可视化")
            return False

        # 如果节点过多，随机选择一部分节点
        if len(self.graph.nodes) > max_nodes:
            logger.info(f"节点数量 ({len(self.graph.nodes)}) 超过最大限制 ({max_nodes})，将随机选择子图")
            nodes = list(self.graph.nodes())
            selected_nodes = np.random.choice(nodes, max_nodes, replace=False)
            subgraph = self.graph.subgraph(selected_nodes)
        else:
            subgraph = self.graph
            
        # 使用NetworkX计算布局
        pos = nx.spring_layout(subgraph, k=0.2, iterations=50)
        
        # 标准化坐标，使其适合图像尺寸
        # 找出坐标的最小值和最大值
        min_x = min(x for x, y in pos.values())
        max_x = max(x for x, y in pos.values())
        min_y = min(y for x, y in pos.values())
        max_y = max(y for x, y in pos.values())
        
        # 创建用于绘图的PIL图像
        img = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(img)
        
        # 尝试加载中文字体
        try:
            system = platform.system()
            if system == 'Darwin':  # macOS
                font_path = '/System/Library/Fonts/PingFang.ttc'  # macOS上的PingFang字体
                title_font = ImageFont.truetype(font_path, 40)
                node_font = ImageFont.truetype(font_path, 14)
                legend_font = ImageFont.truetype(font_path, 20)
            elif system == 'Windows':  # Windows
                font_path = 'C:\\Windows\\Fonts\\msyh.ttc'  # Windows上的微软雅黑字体
                title_font = ImageFont.truetype(font_path, 40)
                node_font = ImageFont.truetype(font_path, 14)
                legend_font = ImageFont.truetype(font_path, 20)
            else:  # Linux或其他
                # 使用默认字体
                title_font = ImageFont.load_default()
                node_font = ImageFont.load_default()
                legend_font = ImageFont.load_default()
        except Exception as e:
            logger.warning(f"无法加载字体: {e}，使用默认字体")
            title_font = ImageFont.load_default()
            node_font = ImageFont.load_default()
            legend_font = ImageFont.load_default()
        
        # 绘制标题
        draw.text((width // 2 - 200, 20), title, fill=(0, 0, 0), font=title_font)
        
        # 转换颜色格式
        def hex_to_rgb(hex_color):
            """将十六进制颜色转换为RGB元组"""
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # 为每种实体类型定义RGB颜色
        entity_colors_rgb = {k: hex_to_rgb(v) for k, v in self.entity_colors.items()}
        relation_colors_rgb = {k: hex_to_rgb(v) for k, v in self.relation_colors.items()}
        
        # 留出边距
        margin = 100
        draw_width = width - 2 * margin
        draw_height = height - 2 * margin
        
        # 缩放因子，将布局坐标映射到图像坐标
        x_scale = draw_width / (max_x - min_x) if max_x != min_x else 1
        y_scale = draw_height / (max_y - min_y) if max_y != min_y else 1
        
        # 绘制边
        edge_drawn = {}  # 避免重复绘制相同类型的边
        for source, target, data in subgraph.edges(data=True):
            relation_type = data.get('type', '')
            
            if relation_type in edge_drawn:
                continue
                
            # 获取源节点和目标节点的坐标
            x1, y1 = pos[source]
            x2, y2 = pos[target]
            
            # 将坐标映射到图像坐标系
            x1_px = int((x1 - min_x) * x_scale + margin)
            y1_px = int((y1 - min_y) * y_scale + margin)
            x2_px = int((x2 - min_x) * x_scale + margin)
            y2_px = int((y2 - min_y) * y_scale + margin)
            
            # 获取关系类型的颜色
            edge_color = relation_colors_rgb.get(relation_type, (150, 150, 150))
            
            # 绘制边
            draw.line((x1_px, y1_px, x2_px, y2_px), fill=edge_color, width=1)
            
            # 记录已绘制的边类型（用于生成图例）
            edge_drawn[relation_type] = edge_color
        
        # 绘制节点
        node_drawn = {}  # 避免重复绘制相同类型的节点
        node_positions = {}  # 记录节点位置，用于标签定位
        
        for node, data in subgraph.nodes(data=True):
            entity_type = data.get('type', '')
            node_name = data.get('name', str(node))
            
            # 获取节点坐标
            x, y = pos[node]
            
            # 将坐标映射到图像坐标系
            x_px = int((x - min_x) * x_scale + margin)
            y_px = int((y - min_y) * y_scale + margin)
            
            # 记录节点位置
            node_positions[node] = (x_px, y_px)
            
            # 获取实体类型的颜色
            node_color = entity_colors_rgb.get(entity_type, (200, 200, 200))
            
            # 绘制节点（椭圆）
            draw.ellipse((x_px - node_size, y_px - node_size, 
                          x_px + node_size, y_px + node_size), 
                         fill=node_color, outline=(0, 0, 0))
            
            # 记录已绘制的节点类型（用于生成图例）
            node_drawn[entity_type] = node_color
        
        # 绘制节点标签
        for node, (x_px, y_px) in node_positions.items():
            node_name = subgraph.nodes[node].get('name', str(node))
            
            # 测量文本宽度
            try:
                text_width = node_font.getlength(node_name)
            except AttributeError:
                # 对于旧版PIL
                text_width = 7 * len(node_name)  # 近似估计
            
            # 绘制标签
            draw.text((x_px - text_width/2, y_px + node_size + 2), 
                      node_name, fill=(0, 0, 0), font=node_font)
        
        # 绘制图例
        legend_x = width - 250
        legend_y = 100
        legend_spacing = 30
        
        # 节点类型图例
        draw.text((legend_x, legend_y), "实体类型:", fill=(0, 0, 0), font=legend_font)
        legend_y += 30
        
        for entity_type, color in node_drawn.items():
            # 绘制图例中的节点示例
            draw.ellipse((legend_x, legend_y, legend_x + 20, legend_y + 20), 
                         fill=color, outline=(0, 0, 0))
            draw.text((legend_x + 30, legend_y), entity_type, 
                      fill=(0, 0, 0), font=legend_font)
            legend_y += legend_spacing
        
        # 关系类型图例
        legend_y += 20
        draw.text((legend_x, legend_y), "关系类型:", fill=(0, 0, 0), font=legend_font)
        legend_y += 30
        
        for relation_type, color in edge_drawn.items():
            # 绘制图例中的边示例
            draw.line((legend_x, legend_y + 10, legend_x + 20, legend_y + 10), 
                      fill=color, width=2)
            draw.text((legend_x + 30, legend_y), relation_type, 
                      fill=(0, 0, 0), font=legend_font)
            legend_y += legend_spacing
        
        # 保存图像
        output_path = self.output_dir / output_file
        img.save(output_path)
        
        logger.info(f"图谱已保存为PIL图像: {output_path}")
        return True 
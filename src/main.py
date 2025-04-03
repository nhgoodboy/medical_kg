#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from data.kg_builder import KnowledgeGraphBuilder
from api.app import create_app
from utils.api_client import DeepSeekAPIClient
from visualization.kg_visualizer import KnowledgeGraphVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def build_knowledge_graph(args):
    """构建医学知识图谱"""
    logger.info("开始构建医学知识图谱...")
    
    # 创建API客户端
    api_client = DeepSeekAPIClient(api_key=args.api_key, model_name=args.model_name)
    
    # 创建知识图谱构建器
    kg_builder = KnowledgeGraphBuilder(
        api_client=api_client, 
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    kg_builder.extract_entities()
    kg_builder.extract_relations()
    kg_builder.build_graph()
    kg_builder.save_graph()
    
    logger.info("医学知识图谱构建完成")

def serve_api(args):
    """启动知识图谱问答API服务"""
    logger.info(f"启动知识图谱问答API服务，端口：{args.port}")
    app = create_app(kg_path=args.kg_path, api_key=args.api_key, model_name=args.model_name)
    app.run(host=args.host, port=args.port, debug=args.debug)

def visualize_graph(args):
    """可视化知识图谱"""
    logger.info(f"可视化知识图谱：{args.kg_path}")
    
    # 创建可视化器
    visualizer = KnowledgeGraphVisualizer(
        graph_path=args.kg_path,
        output_dir=args.output_dir
    )
    
    # 检查是否只使用PIL可视化
    if args.pil_only:
        logger.info("仅使用PIL库可视化，避免字体问题")
        visualizer.visualize_with_pil(
            max_nodes=args.max_nodes
        )
        
        # 如果指定了实体，创建实体子图
        if args.entity:
            subgraph = visualizer.create_entity_subgraph(
                entity_name=args.entity,
                depth=args.depth,
                max_nodes=args.max_nodes
            )
            
            if subgraph:
                sub_visualizer = KnowledgeGraphVisualizer(
                    graph=subgraph,
                    output_dir=args.output_dir
                )
                
                sub_visualizer.visualize_with_pil(
                    title=f"医学知识图谱 - {args.entity}的相关实体",
                    output_file=f"entity_{args.entity}_pil.png"
                )
        
        # 生成统计信息
        if args.stats:
            visualizer.generate_statistics()
            
        # 导出为D3.js格式
        if args.export_d3:
            visualizer.export_to_d3_json(max_nodes=args.max_nodes)
            
        logger.info(f"知识图谱可视化完成，结果保存在 {args.output_dir} 目录")
        return
    
    # 根据可视化类型进行可视化
    if args.type == "matplotlib" or args.type == "all":
        visualizer.visualize_with_matplotlib(
            max_nodes=args.max_nodes,
            layout=args.layout
        )
        
    if args.type == "pyvis" or args.type == "all":
        visualizer.visualize_with_pyvis(
            max_nodes=args.max_nodes
        )
        
    if args.type == "plotly" or args.type == "all":
        visualizer.visualize_with_plotly(
            max_nodes=args.max_nodes
        )
        
    if args.type == "pil" or args.type == "all":
        visualizer.visualize_with_pil(
            max_nodes=args.max_nodes
        )
    
    # 生成统计信息
    if args.stats:
        visualizer.generate_statistics()
    
    # 导出为D3.js格式
    if args.export_d3:
        visualizer.export_to_d3_json(max_nodes=args.max_nodes)
    
    # 创建特定实体的子图
    if args.entity:
        subgraph = visualizer.create_entity_subgraph(
            entity_name=args.entity,
            depth=args.depth,
            max_nodes=args.max_nodes
        )
        
        if subgraph:
            # 使用子图重新创建可视化器
            sub_visualizer = KnowledgeGraphVisualizer(
                graph=subgraph,
                output_dir=args.output_dir
            )
            
            # 可视化子图
            if args.type == "matplotlib" or args.type == "all":
                sub_visualizer.visualize_with_matplotlib(
                    title=f"医学知识图谱 - {args.entity}的相关实体",
                    output_file=f"entity_{args.entity}_matplotlib.png",
                    layout=args.layout
                )
                
            if args.type == "pyvis" or args.type == "all":
                sub_visualizer.visualize_with_pyvis(
                    output_file=f"entity_{args.entity}_interactive.html"
                )
                
            if args.type == "pil" or args.type == "all":
                sub_visualizer.visualize_with_pil(
                    title=f"医学知识图谱 - {args.entity}的相关实体",
                    output_file=f"entity_{args.entity}_pil.png"
                )
    
    logger.info(f"知识图谱可视化完成，结果保存在 {args.output_dir} 目录")

def main():
    parser = argparse.ArgumentParser(description="医学领域知识图谱问答系统")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 构建知识图谱命令
    build_parser = subparsers.add_parser("build_kg", help="构建医学知识图谱")
    build_parser.add_argument("--data-dir", type=str, default="data/raw", help="原始数据目录")
    build_parser.add_argument("--output-dir", type=str, default="data/processed", help="处理后数据保存目录")
    build_parser.add_argument("--model-name", type=str, default="deepseek-chat", help="使用的模型名称")
    build_parser.add_argument("--api-key", type=str, help="DeepSeek API密钥，如不提供则从环境变量DEEPSEEK_API_KEY获取")
    
    # 启动API服务命令
    serve_parser = subparsers.add_parser("serve", help="启动知识图谱问答API服务")
    serve_parser.add_argument("--kg-path", type=str, default="data/processed/medical_kg.graphml", help="知识图谱文件路径")
    serve_parser.add_argument("--model-name", type=str, default="deepseek-chat", help="使用的模型名称")
    serve_parser.add_argument("--api-key", type=str, help="DeepSeek API密钥，如不提供则从环境变量DEEPSEEK_API_KEY获取")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="服务主机地址")
    serve_parser.add_argument("--port", type=int, default=5000, help="服务端口")
    serve_parser.add_argument("--debug", action="store_true", help="是否启用调试模式")
    
    # 可视化知识图谱命令
    vis_parser = subparsers.add_parser("visualize", help="可视化知识图谱")
    vis_parser.add_argument("--kg-path", type=str, default="data/processed/medical_kg.graphml", help="知识图谱文件路径")
    vis_parser.add_argument("--output-dir", type=str, default="data/visualization", help="可视化结果保存目录")
    vis_parser.add_argument("--type", type=str, default="all", choices=["matplotlib", "pyvis", "plotly", "pil", "all"], help="可视化类型")
    vis_parser.add_argument("--pil-only", action="store_true", help="仅使用PIL库可视化，解决中文显示问题")
    vis_parser.add_argument("--max-nodes", type=int, default=100, help="最大显示节点数")
    vis_parser.add_argument("--layout", type=str, default="spring", choices=["spring", "circular", "random", "shell", "kamada_kawai"], help="节点布局算法")
    vis_parser.add_argument("--stats", action="store_true", help="是否生成统计信息")
    vis_parser.add_argument("--export-d3", action="store_true", help="是否导出为D3.js格式")
    vis_parser.add_argument("--entity", type=str, help="用于创建子图的中心实体名称")
    vis_parser.add_argument("--depth", type=int, default=2, help="子图的深度")
    
    args = parser.parse_args()
    
    if args.command == "build_kg":
        build_knowledge_graph(args)
    elif args.command == "serve":
        serve_api(args)
    elif args.command == "visualize":
        visualize_graph(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
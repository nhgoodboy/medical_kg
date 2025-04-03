#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="medical_kg",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.31.0",
        "openai>=1.12.0",
        "flask>=2.3.2",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "networkx>=3.1",
        "tqdm>=4.65.0",
        "rdflib>=6.3.2",
        "matplotlib>=3.7.1",
    ],
    entry_points={
        "console_scripts": [
            "medical-kg=src.main:main",
        ],
    },
) 
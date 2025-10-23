#!/usr/bin/env python3
"""
Visual RAPTOR ColBERT Integration System
セットアップスクリプト
"""

from setuptools import setup, find_packages
import os

# README読み込み
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# requirements.txt読み込み
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="visual-raptor-colbert",
    version="1.0.0",
    author="LangChain Learning Project",
    author_email="contact@example.com",
    description="Visual RAPTOR ColBERT Integration System for Disaster Document Retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/langchain-ai/learning-langchain",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": [
            "torch>=2.0.0+cu118",
            "faiss-gpu>=1.7.4",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "web": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "streamlit>=1.25.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "visual-raptor=integrated_system:main",
            "generate-disaster-docs=disaster_dataset_generator:main",
            "setup-jina-vdr=jina_vdr_benchmark:main",
        ],
    },
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
    include_package_data=True,
    keywords=[
        "RAPTOR", "ColBERT", "Visual Document Retrieval", "OCR", 
        "Japanese", "Disaster Response", "Information Retrieval",
        "Multimodal Search", "LangChain", "RAG"
    ],
    project_urls={
        "Bug Reports": "https://github.com/langchain-ai/learning-langchain/issues",
        "Source": "https://github.com/langchain-ai/learning-langchain",
        "Documentation": "https://github.com/langchain-ai/learning-langchain/blob/main/visual-raptor-colvbert/README.md",
    },
)
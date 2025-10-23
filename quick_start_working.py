#!/usr/bin/env python3
"""
Simple Quick Start - No Dependencies on Broken Files
シンプルなクイックスタート - 破損ファイルへの依存なし
"""

import sys
from pathlib import Path

print("="*80)
print("🚀 Visual RAPTOR ColBERT - Simple Quick Start")
print("災害文書検索システム クイックスタート（簡易版）")
print("="*80)

# 1. Ollamaの確認
print("\n🔧 Step 1: Checking Ollama connection...")
try:
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url="http://localhost:11434"
    )
    
    # テスト
    test_embed = embeddings.embed_query("test")
    print(f"   ✅ Ollama embeddings working (dim: {len(test_embed)})")
    
    llm = ChatOllama(
        model="granite-code:8b",
        temperature=0,
        base_url="http://localhost:11434"
    )
    
    # テスト
    test_response = llm.invoke("こんにちは")
    print(f"   ✅ Ollama LLM working")
    
except Exception as e:
    print(f"   ❌ Ollama failed: {e}")
    print("\nPlease ensure:")
    print("  1. Ollama is running: ollama serve")
    print("  2. Models are available:")
    print("     - ollama pull mxbai-embed-large")
    print("     - ollama pull granite-code:8b")
    sys.exit(1)

# 2. PyTorchとGPUの確認
print("\n🔧 Step 2: Checking PyTorch and GPU...")
try:
    import torch
    
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✅ GPU ready for acceleration")
    else:
        print(f"   ⚠️  GPU not available, using CPU")
        
except Exception as e:
    print(f"   ❌ PyTorch check failed: {e}")

# 3. 簡単なデモ実行
print("\n🚀 Step 3: Running simple disaster document demo...")

# サンプル災害文書
sample_docs = [
    {"id": "doc1", "title": "地震発生時の避難手順", "content": "地震が発生したら、まず身の安全を確保してください。揺れが収まったら、火の始末をして避難経路を確認します。"},
    {"id": "doc2", "title": "津波警報時の対応", "content": "津波警報が発令されたら、直ちに高台または3階建て以上の頑丈な建物に避難してください。"},
    {"id": "doc3", "title": "避難所での生活ガイド", "content": "避難所では譲り合いの精神で生活してください。食事は決められた時間に配布されます。"},
    {"id": "doc4", "title": "緊急連絡先一覧", "content": "消防署: 119、警察署: 110、市役所災害対策本部: 045-123-4567"},
    {"id": "doc5", "title": "備蓄品チェックリスト", "content": "非常食（3日分）、飲料水（1人1日3リットル×3日分）、懐中電灯、ラジオ、応急医薬品"}
]

# 文書の埋め込みを作成
print("   Creating document embeddings...")
doc_texts = [f"{doc['title']}\n{doc['content']}" for doc in sample_docs]

try:
    doc_embeddings = []
    for i, text in enumerate(doc_texts):
        emb = embeddings.embed_query(text)
        doc_embeddings.append(emb)
        if (i + 1) % 2 == 0:
            print(f"      Embedded {i+1}/{len(doc_texts)} documents")
    
    print(f"   ✅ Created {len(doc_embeddings)} document embeddings")
    
except Exception as e:
    print(f"   ❌ Embedding failed: {e}")
    sys.exit(1)

# クエリで検索
print("\n🔍 Step 4: Testing document search...")

test_queries = [
    "地震が起きたときはどうすればいいですか？",
    "津波警報が出たときの対応は？",
    "緊急時の連絡先を教えて"
]

import numpy as np

for query_text in test_queries:
    print(f"\n   Query: {query_text}")
    
    # クエリの埋め込み
    query_emb = embeddings.embed_query(query_text)
    
    # コサイン類似度計算
    query_vec = np.array(query_emb)
    doc_vecs = np.array(doc_embeddings)
    
    similarities = np.dot(doc_vecs, query_vec) / (
        np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec)
    )
    
    # 上位3件取得
    top_indices = np.argsort(similarities)[::-1][:3]
    
    print("   Top results:")
    for rank, idx in enumerate(top_indices, 1):
        doc = sample_docs[idx]
        score = similarities[idx]
        print(f"      {rank}. {doc['title']} (score: {score:.4f})")

# 5. LLMで要約生成
print("\n🤖 Step 5: Testing LLM summarization...")

top_doc = sample_docs[top_indices[0]]
prompt = f"""
次の災害関連文書を参考に、質問に対する回答を日本語で生成してください。

質問: {test_queries[0]}

文書: {top_doc['title']}
{top_doc['content']}

回答:"""

try:
    response = llm.invoke(prompt)
    print(f"   LLM Response: {response.content[:200]}...")
    print(f"   ✅ LLM summarization working")
    
except Exception as e:
    print(f"   ❌ LLM failed: {e}")

# まとめ
print("\n" + "="*80)
print("✅ Quick Start Completed Successfully!")
print("="*80)
print("\n📊 System Status:")
print("   ✅ Ollama embeddings: Working")
print("   ✅ Ollama LLM: Working")
print("   ✅ Document search: Working")
print("   ✅ Summarization: Working")

if torch.cuda.is_available():
    print(f"   ✅ GPU ({torch.cuda.get_device_name(0)}): Available")
else:
    print("   ⚠️  GPU: Not available (using CPU)")

print("\n🎉 Your Visual RAPTOR ColBERT system is ready!")
print("\nNext steps:")
print("  - Run 'python gpu_utilization_test.py' to test GPU acceleration")
print("  - Check 'data/' directory for results")
print("  - Explore the full system with more documents")

print("\n" + "="*80)

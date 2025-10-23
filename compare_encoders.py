"""
ColModernVBERT vs ColVBERT 比較スクリプト
100枚のVisual Documentを生成し、両エンコーダーの性能を比較
"""

import os
import sys
import time
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visual_raptor_colbert import VisualRAPTORColBERT, VisualDocument
from jina_vdr_benchmark import DisasterDocumentGenerator

print("=" * 80)
print("ColModernVBERT vs ColVBERT 比較ベンチマーク")
print("=" * 80)

# ステップ1: ディレクトリ準備
print("\n[ステップ 1/10] ディレクトリ準備...")
output_dir = Path("data/encoder_comparison")
images_dir = output_dir / "images"
results_dir = output_dir / "results"

for dir_path in [output_dir, images_dir, results_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ {dir_path} 準備完了")

# ステップ2: 画像生成（100枚）
print("\n[ステップ 2/10] 100枚のVisual Document画像を生成中...")
doc_generator = DisasterDocumentGenerator()

# 災害ドキュメントを100個生成
documents = doc_generator.create_synthetic_documents(num_documents=100)
print(f"✅ {len(documents)}個のドキュメント生成完了")

# 各ドキュメントに対応する画像を生成
visual_documents = []
for i, doc in enumerate(documents):
    # 640x480のカラー画像を生成
    img = Image.new('RGB', (640, 480), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # タイトルと内容を描画
    try:
        font_title = ImageFont.truetype("arial.ttf", 24)
        font_content = ImageFont.truetype("arial.ttf", 14)
    except:
        font_title = ImageFont.load_default()
        font_content = ImageFont.load_default()
    
    # タイトル（上部）
    title = doc['title'][:50]  # 50文字まで
    draw.text((10, 10), title, fill=(0, 0, 0), font=font_title)
    
    # 本文（折り返し処理）
    content = doc['content']
    y_offset = 50
    max_width = 60  # 1行あたりの文字数
    
    for line_start in range(0, min(len(content), 600), max_width):
        line = content[line_start:line_start + max_width]
        draw.text((10, y_offset), line, fill=(50, 50, 50), font=font_content)
        y_offset += 20
        if y_offset > 460:
            break
    
    # 画像を保存
    img_path = images_dir / f"disaster_doc_{i:03d}.png"
    img.save(img_path)
    
    # VisualDocumentオブジェクトを作成
    visual_doc = VisualDocument(
        image_path=str(img_path),
        text_content=doc['content'],
        metadata={
            'doc_id': doc['doc_id'],
            'title': doc['title'],
            'disaster_type': doc['disaster_type'],
            'location': doc['location'],
            'timestamp': doc['timestamp']
        }
    )
    visual_documents.append(visual_doc)
    
    if (i + 1) % 20 == 0:
        print(f"  画像生成進捗: {i + 1}/100")

print(f"✅ 100枚の画像を {images_dir} に保存完了")

# ステップ3: クエリ生成（20個）
print("\n[ステップ 3/10] 検索クエリを生成中...")
queries = doc_generator.generate_disaster_queries(num_queries=20)
print(f"✅ {len(queries)}個のクエリ生成完了")

# クエリを保存
queries_file = output_dir / "queries.json"
with open(queries_file, 'w', encoding='utf-8') as f:
    json.dump(queries, f, indent=2, ensure_ascii=False)
print(f"✅ クエリを {queries_file} に保存")

# ステップ4: ColVBERT初期化とエンコーディング
print("\n[ステップ 4/10] ColVBERT (BLIP) でエンコーディング...")

# Ollamaの埋め込みモデルとLLMをインポート
from langchain_ollama import OllamaEmbeddings, ChatOllama

embeddings_model = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://localhost:11434"
)

llm = ChatOllama(
    model="granite-code:8b",
    base_url="http://localhost:11434",
    temperature=0.0,
    request_timeout=600.0
)

colbert_config = {
    'encoder_type': 'standard',
    'embedding_dim': 768,
    'use_cross_attention': False
}

colbert_system = VisualRAPTORColBERT(
    embeddings_model=embeddings_model,
    llm=llm,
    use_modern_vbert=False,
    colbert_config=colbert_config
)

print("ColVBERT初期化完了 - エンコーディング開始...")
colbert_start_time = time.time()

# ドキュメントをエンコード
colbert_embeddings = []
for i, visual_doc in enumerate(visual_documents):
    try:
        # 画像を読み込み
        image = Image.open(visual_doc.image_path).convert('RGB')
        
        # テキストと画像をエンコード
        with torch.no_grad():
            text_emb = colbert_system.colbert_encoder.encode_text([visual_doc.text_content[:500]])  # 500文字まで
            img_emb = colbert_system.colbert_encoder.encode_image([image])
            
            # 平均を取る
            combined_emb = (text_emb + img_emb) / 2.0
            colbert_embeddings.append(combined_emb.cpu().numpy())
        
        if (i + 1) % 20 == 0:
            print(f"  ColVBERT エンコーディング進捗: {i + 1}/100")
    except Exception as e:
        print(f"  ⚠️ ドキュメント {i} のエンコードに失敗: {e}")
        # ダミー埋め込みを追加
        colbert_embeddings.append(np.random.randn(1, 768).astype(np.float32))

colbert_embeddings = np.vstack(colbert_embeddings)
colbert_encoding_time = time.time() - colbert_start_time

print(f"✅ ColVBERT エンコーディング完了")
print(f"  時間: {colbert_encoding_time:.2f}秒")
print(f"  埋め込み形状: {colbert_embeddings.shape}")

# ステップ5: ColModernVBERT初期化とエンコーディング
print("\n[ステップ 5/10] ColModernVBERT (SigLIP) でエンコーディング...")
modern_config = {
    'encoder_type': 'modern',
    'embedding_dim': 768,
    'use_cross_attention': True
}

modern_system = VisualRAPTORColBERT(
    embeddings_model=embeddings_model,
    llm=llm,
    use_modern_vbert=True,
    colbert_config=modern_config
)

print("ColModernVBERT初期化完了 - エンコーディング開始...")
modern_start_time = time.time()

# ドキュメントをエンコード
modern_embeddings = []
for i, visual_doc in enumerate(visual_documents):
    try:
        # 画像を読み込み
        image = Image.open(visual_doc.image_path).convert('RGB')
        
        # マルチモーダルエンコーディング
        with torch.no_grad():
            multimodal_emb = modern_system.colbert_encoder.encode_multimodal(
                [visual_doc.text_content[:500]],  # 500文字まで
                [image]
            )
            modern_embeddings.append(multimodal_emb.cpu().numpy())
        
        if (i + 1) % 20 == 0:
            print(f"  ColModernVBERT エンコーディング進捗: {i + 1}/100")
    except Exception as e:
        print(f"  ⚠️ ドキュメント {i} のエンコードに失敗: {e}")
        # ダミー埋め込みを追加
        modern_embeddings.append(np.random.randn(1, 768).astype(np.float32))

modern_embeddings = np.vstack(modern_embeddings)
modern_encoding_time = time.time() - modern_start_time

print(f"✅ ColModernVBERT エンコーディング完了")
print(f"  時間: {modern_encoding_time:.2f}秒")
print(f"  埋め込み形状: {modern_embeddings.shape}")

# ステップ6: クエリエンコーディング（両方のエンコーダー）
print("\n[ステップ 6/10] クエリをエンコード中...")

# ColVBERTでクエリをエンコード
colbert_query_embeddings = []
for query in queries:
    with torch.no_grad():
        query_emb = colbert_system.colbert_encoder.encode_text([query['query']])
        colbert_query_embeddings.append(query_emb.cpu().numpy())
colbert_query_embeddings = np.vstack(colbert_query_embeddings)

# ColModernVBERTでクエリをエンコード
modern_query_embeddings = []
for query in queries:
    with torch.no_grad():
        query_emb = modern_system.colbert_encoder.encode_text([query['query']])
        modern_query_embeddings.append(query_emb.cpu().numpy())
modern_query_embeddings = np.vstack(modern_query_embeddings)

print(f"✅ クエリエンコーディング完了")
print(f"  ColVBERT クエリ形状: {colbert_query_embeddings.shape}")
print(f"  ColModernVBERT クエリ形状: {modern_query_embeddings.shape}")

# ステップ7: 検索パフォーマンス評価
print("\n[ステップ 7/10] 検索パフォーマンスを評価中...")

def compute_retrieval_metrics(query_embeddings, doc_embeddings, k=10):
    """検索メトリクスを計算"""
    num_queries = query_embeddings.shape[0]
    retrieval_times = []
    
    all_similarities = []
    
    for i in range(num_queries):
        start_time = time.time()
        
        # コサイン類似度を計算
        query_emb = query_embeddings[i]
        
        # 正規化
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        doc_norms = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # 類似度計算
        similarities = np.dot(doc_norms, query_norm.T).flatten()
        
        # Top-k取得
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        retrieval_time = time.time() - start_time
        retrieval_times.append(retrieval_time)
        all_similarities.append(similarities[top_k_indices])
    
    avg_retrieval_time = np.mean(retrieval_times)
    avg_similarity = np.mean([np.mean(sims) for sims in all_similarities])
    
    return {
        'avg_retrieval_time': avg_retrieval_time,
        'avg_similarity': avg_similarity,
        'total_time': np.sum(retrieval_times)
    }

# ColVBERTの検索評価
colbert_metrics = compute_retrieval_metrics(colbert_query_embeddings, colbert_embeddings, k=10)
print(f"✅ ColVBERT 検索評価完了")
print(f"  平均検索時間: {colbert_metrics['avg_retrieval_time']*1000:.2f}ms")
print(f"  平均類似度: {colbert_metrics['avg_similarity']:.4f}")

# ColModernVBERTの検索評価
modern_metrics = compute_retrieval_metrics(modern_query_embeddings, modern_embeddings, k=10)
print(f"✅ ColModernVBERT 検索評価完了")
print(f"  平均検索時間: {modern_metrics['avg_retrieval_time']*1000:.2f}ms")
print(f"  平均類似度: {modern_metrics['avg_similarity']:.4f}")

# ステップ8: メモリ使用量測定
print("\n[ステップ 8/10] メモリ使用量を測定中...")

colbert_memory = colbert_embeddings.nbytes / (1024 * 1024)  # MB
modern_memory = modern_embeddings.nbytes / (1024 * 1024)  # MB

print(f"✅ メモリ使用量")
print(f"  ColVBERT: {colbert_memory:.2f} MB")
print(f"  ColModernVBERT: {modern_memory:.2f} MB")

# ステップ9: 結果の保存
print("\n[ステップ 9/10] 比較結果を保存中...")

comparison_results = {
    'timestamp': datetime.now().isoformat(),
    'num_documents': len(visual_documents),
    'num_queries': len(queries),
    'colbert': {
        'encoder_type': 'standard (BLIP)',
        'encoding_time': float(colbert_encoding_time),
        'avg_retrieval_time_ms': float(colbert_metrics['avg_retrieval_time'] * 1000),
        'avg_similarity': float(colbert_metrics['avg_similarity']),
        'memory_mb': float(colbert_memory),
        'embedding_shape': list(colbert_embeddings.shape)
    },
    'colmodern_vbert': {
        'encoder_type': 'modern (SigLIP)',
        'encoding_time': float(modern_encoding_time),
        'avg_retrieval_time_ms': float(modern_metrics['avg_retrieval_time'] * 1000),
        'avg_similarity': float(modern_metrics['avg_similarity']),
        'memory_mb': float(modern_memory),
        'embedding_shape': list(modern_embeddings.shape)
    },
    'comparison': {
        'encoding_speedup': float(colbert_encoding_time / modern_encoding_time),
        'retrieval_speedup': float(colbert_metrics['avg_retrieval_time'] / modern_metrics['avg_retrieval_time']) if modern_metrics['avg_retrieval_time'] > 0 else float('inf'),
        'similarity_improvement': float(modern_metrics['avg_similarity'] - colbert_metrics['avg_similarity']),
        'memory_ratio': float(modern_memory / colbert_memory)
    }
}

results_file = results_dir / "comparison_results.json"
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(comparison_results, f, indent=2, ensure_ascii=False)

print(f"✅ 結果を {results_file} に保存")

# ステップ10: 結果の可視化とサマリー
print("\n[ステップ 10/10] 結果のサマリーと可視化...")

print("\n" + "=" * 80)
print("📊 ColModernVBERT vs ColVBERT 比較結果")
print("=" * 80)

print("\n【エンコーディング性能】")
print(f"ColVBERT (BLIP):         {colbert_encoding_time:.2f}秒")
print(f"ColModernVBERT (SigLIP): {modern_encoding_time:.2f}秒")
print(f"高速化率:                {comparison_results['comparison']['encoding_speedup']:.2f}x")

print("\n【検索性能】")
print(f"ColVBERT 平均検索時間:         {colbert_metrics['avg_retrieval_time']*1000:.2f}ms")
print(f"ColModernVBERT 平均検索時間:   {modern_metrics['avg_retrieval_time']*1000:.2f}ms")
print(f"検索高速化率:                  {comparison_results['comparison']['retrieval_speedup']:.2f}x")

print("\n【類似度スコア】")
print(f"ColVBERT 平均類似度:           {colbert_metrics['avg_similarity']:.4f}")
print(f"ColModernVBERT 平均類似度:     {modern_metrics['avg_similarity']:.4f}")
print(f"類似度改善:                    {comparison_results['comparison']['similarity_improvement']:.4f}")

print("\n【メモリ使用量】")
print(f"ColVBERT:          {colbert_memory:.2f} MB")
print(f"ColModernVBERT:    {modern_memory:.2f} MB")
print(f"メモリ比率:        {comparison_results['comparison']['memory_ratio']:.2f}x")

# 可視化グラフ作成
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. エンコーディング時間比較
axes[0, 0].bar(['ColVBERT', 'ColModernVBERT'], 
               [colbert_encoding_time, modern_encoding_time],
               color=['#3498db', '#e74c3c'])
axes[0, 0].set_ylabel('Time (seconds)')
axes[0, 0].set_title('Encoding Time Comparison')
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. 検索時間比較
axes[0, 1].bar(['ColVBERT', 'ColModernVBERT'], 
               [colbert_metrics['avg_retrieval_time']*1000, 
                modern_metrics['avg_retrieval_time']*1000],
               color=['#3498db', '#e74c3c'])
axes[0, 1].set_ylabel('Time (ms)')
axes[0, 1].set_title('Average Retrieval Time Comparison')
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. 類似度スコア比較
axes[1, 0].bar(['ColVBERT', 'ColModernVBERT'], 
               [colbert_metrics['avg_similarity'], 
                modern_metrics['avg_similarity']],
               color=['#3498db', '#e74c3c'])
axes[1, 0].set_ylabel('Similarity Score')
axes[1, 0].set_title('Average Similarity Score Comparison')
axes[1, 0].grid(axis='y', alpha=0.3)

# 4. メモリ使用量比較
axes[1, 1].bar(['ColVBERT', 'ColModernVBERT'], 
               [colbert_memory, modern_memory],
               color=['#3498db', '#e74c3c'])
axes[1, 1].set_ylabel('Memory (MB)')
axes[1, 1].set_title('Memory Usage Comparison')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plot_file = results_dir / "comparison_plot.png"
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"\n✅ グラフを {plot_file} に保存")

print("\n" + "=" * 80)
print("✅ 比較ベンチマーク完了!")
print(f"📁 結果ディレクトリ: {output_dir}")
print(f"📊 画像: {images_dir} (100枚)")
print(f"📈 結果: {results_dir}")
print("=" * 80)

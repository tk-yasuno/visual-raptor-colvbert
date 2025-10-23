"""
埋め込みの詳細分析スクリプト
ColVBERT vs ColModernVBERT の埋め込み特性を調査
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visual_raptor_colbert import VisualRAPTORColBERT
from langchain_ollama import OllamaEmbeddings, ChatOllama
from jina_vdr_benchmark import DisasterDocumentGenerator

print("=" * 80)
print("埋め込み特性分析")
print("=" * 80)

# Ollamaモデル初期化
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

# テストドキュメント生成
print("\n[1/5] テストデータ生成...")
doc_generator = DisasterDocumentGenerator()
documents = doc_generator.create_synthetic_documents(num_documents=10)

# 画像生成
images_dir = Path("data/embedding_analysis")
images_dir.mkdir(parents=True, exist_ok=True)

test_images = []
test_texts = []

for i, doc in enumerate(documents):
    img = Image.new('RGB', (640, 480), color=(255, 255, 255))
    img_path = images_dir / f"test_{i}.png"
    img.save(img_path)
    test_images.append(Image.open(img_path).convert('RGB'))
    test_texts.append(doc['content'][:500])

print(f"✅ {len(test_images)}個のテスト画像/テキスト生成完了")

# ColVBERT初期化
print("\n[2/5] ColVBERT初期化...")
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

# ColModernVBERT初期化
print("\n[3/5] ColModernVBERT初期化...")
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

# エンコーディング
print("\n[4/5] エンコーディング実行...")

with torch.no_grad():
    # ColVBERT
    colbert_text_emb = colbert_system.colbert_encoder.encode_text(test_texts)
    colbert_img_emb = colbert_system.colbert_encoder.encode_image(test_images)
    
    # ColModernVBERT
    modern_text_emb = modern_system.colbert_encoder.encode_text(test_texts)
    modern_img_emb = modern_system.colbert_encoder.encode_image(test_images)

print("✅ エンコーディング完了")

# 統計分析
print("\n[5/5] 統計分析...")

def analyze_embeddings(embeddings, name):
    """埋め込みの統計を計算"""
    embeddings_np = embeddings.cpu().numpy()
    
    print(f"\n{name}:")
    print(f"  形状: {embeddings_np.shape}")
    print(f"  平均: {embeddings_np.mean():.6f}")
    print(f"  標準偏差: {embeddings_np.std():.6f}")
    print(f"  最小値: {embeddings_np.min():.6f}")
    print(f"  最大値: {embeddings_np.max():.6f}")
    
    # ノルム計算
    norms = np.linalg.norm(embeddings_np, axis=1)
    print(f"  L2ノルム - 平均: {norms.mean():.6f}, 標準偏差: {norms.std():.6f}")
    
    # 自己類似度（同じエンコーダー内）
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings_np)
    # 対角線を除外
    sim_values = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    print(f"  自己類似度 - 平均: {sim_values.mean():.6f}, 標準偏差: {sim_values.std():.6f}")
    
    return embeddings_np, sim_matrix

print("\n" + "=" * 80)
print("📊 ColVBERT 埋め込み統計")
print("=" * 80)
colbert_text_np, colbert_text_sim = analyze_embeddings(colbert_text_emb, "ColVBERT テキスト")
colbert_img_np, colbert_img_sim = analyze_embeddings(colbert_img_emb, "ColVBERT 画像")

print("\n" + "=" * 80)
print("📊 ColModernVBERT 埋め込み統計")
print("=" * 80)
modern_text_np, modern_text_sim = analyze_embeddings(modern_text_emb, "ColModernVBERT テキスト")
modern_img_np, modern_img_sim = analyze_embeddings(modern_img_emb, "ColModernVBERT 画像")

# クロスモーダル類似度
print("\n" + "=" * 80)
print("📊 クロスモーダル類似度（テキスト vs 画像）")
print("=" * 80)

from sklearn.metrics.pairwise import cosine_similarity

colbert_cross_sim = cosine_similarity(colbert_text_np, colbert_img_np)
modern_cross_sim = cosine_similarity(modern_text_np, modern_img_np)

print(f"\nColVBERT クロスモーダル類似度:")
print(f"  平均: {colbert_cross_sim.mean():.6f}")
print(f"  標準偏差: {colbert_cross_sim.std():.6f}")
print(f"  対角要素（対応ペア）平均: {np.diag(colbert_cross_sim).mean():.6f}")

print(f"\nColModernVBERT クロスモーダル類似度:")
print(f"  平均: {modern_cross_sim.mean():.6f}")
print(f"  標準偏差: {modern_cross_sim.std():.6f}")
print(f"  対角要素（対応ペア）平均: {np.diag(modern_cross_sim).mean():.6f}")

# 可視化
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# ColVBERT
im1 = axes[0, 0].imshow(colbert_text_sim, cmap='viridis', vmin=0, vmax=1)
axes[0, 0].set_title('ColVBERT Text Self-Similarity')
plt.colorbar(im1, ax=axes[0, 0])

im2 = axes[0, 1].imshow(colbert_img_sim, cmap='viridis', vmin=0, vmax=1)
axes[0, 1].set_title('ColVBERT Image Self-Similarity')
plt.colorbar(im2, ax=axes[0, 1])

im3 = axes[0, 2].imshow(colbert_cross_sim, cmap='viridis', vmin=0, vmax=1)
axes[0, 2].set_title('ColVBERT Cross-Modal Similarity')
plt.colorbar(im3, ax=axes[0, 2])

# ColModernVBERT
im4 = axes[1, 0].imshow(modern_text_sim, cmap='viridis', vmin=0, vmax=1)
axes[1, 0].set_title('ColModernVBERT Text Self-Similarity')
plt.colorbar(im4, ax=axes[1, 0])

im5 = axes[1, 1].imshow(modern_img_sim, cmap='viridis', vmin=0, vmax=1)
axes[1, 1].set_title('ColModernVBERT Image Self-Similarity')
plt.colorbar(im5, ax=axes[1, 1])

im6 = axes[1, 2].imshow(modern_cross_sim, cmap='viridis', vmin=0, vmax=1)
axes[1, 2].set_title('ColModernVBERT Cross-Modal Similarity')
plt.colorbar(im6, ax=axes[1, 2])

plt.tight_layout()
output_path = images_dir / "embedding_similarity_analysis.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 可視化を {output_path} に保存")

# 分布の比較
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist(colbert_text_np.flatten(), bins=50, alpha=0.7, label='ColVBERT Text')
axes[0, 0].set_title('ColVBERT Text Embedding Distribution')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

axes[0, 1].hist(colbert_img_np.flatten(), bins=50, alpha=0.7, label='ColVBERT Image')
axes[0, 1].set_title('ColVBERT Image Embedding Distribution')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()

axes[1, 0].hist(modern_text_np.flatten(), bins=50, alpha=0.7, label='ColModernVBERT Text', color='orange')
axes[1, 0].set_title('ColModernVBERT Text Embedding Distribution')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()

axes[1, 1].hist(modern_img_np.flatten(), bins=50, alpha=0.7, label='ColModernVBERT Image', color='orange')
axes[1, 1].set_title('ColModernVBERT Image Embedding Distribution')
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

plt.tight_layout()
dist_output_path = images_dir / "embedding_distribution_analysis.png"
plt.savefig(dist_output_path, dpi=150, bbox_inches='tight')
print(f"✅ 分布グラフを {dist_output_path} に保存")

print("\n" + "=" * 80)
print("✅ 埋め込み分析完了")
print("=" * 80)

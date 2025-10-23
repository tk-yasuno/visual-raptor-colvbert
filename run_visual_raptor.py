"""
Visual RAPTOR ColBERT 完全実行デモ
ColModernVBERT (SigLIP) を使用した災害文書検索システム
"""

import os
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visual_raptor_colbert import VisualRAPTORColBERT, VisualDocument
from langchain_ollama import OllamaEmbeddings, ChatOllama
from jina_vdr_benchmark import DisasterDocumentGenerator

print("=" * 80)
print("🚀 Visual RAPTOR ColBERT - 完全実行デモ")
print("=" * 80)

# ステップ1: ディレクトリ準備
print("\n[ステップ 1/8] 出力ディレクトリ準備...")
output_dir = Path("data/visual_raptor_run")
images_dir = output_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)
print(f"✅ {output_dir} 準備完了")

# ステップ2: Ollamaモデル初期化
print("\n[ステップ 2/8] Ollamaモデル初期化...")
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
print("✅ Ollama mxbai-embed-large & granite-code:8b 初期化完了")

# ステップ3: ColModernVBERT初期化
print("\n[ステップ 3/8] ColModernVBERT (SigLIP) 初期化...")
config = {
    'encoder_type': 'modern',
    'embedding_dim': 768,
    'use_cross_attention': True
}

visual_raptor = VisualRAPTORColBERT(
    embeddings_model=embeddings_model,
    llm=llm,
    use_modern_vbert=True,
    colbert_config=config
)
print("✅ Visual RAPTOR ColBERT with SigLIP 初期化完了")

# ステップ4: テストドキュメント生成
print("\n[ステップ 4/8] 災害関連Visual Document生成...")
doc_generator = DisasterDocumentGenerator()
documents = doc_generator.create_synthetic_documents(num_documents=20)

visual_documents = []
for i, doc in enumerate(documents):
    # 画像生成
    img = Image.new('RGB', (640, 480), color=(240, 240, 255))
    draw = ImageDraw.Draw(img)
    
    try:
        font_title = ImageFont.truetype("arial.ttf", 20)
        font_body = ImageFont.truetype("arial.ttf", 12)
    except:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()
    
    # タイトル描画
    title = f"{doc['disaster_type'].upper()} - {doc['location']}"
    draw.text((20, 20), title[:60], fill=(0, 0, 100), font=font_title)
    
    # 内容描画（改行処理）
    content_lines = doc['content'][:400].split('。')
    y_pos = 60
    for line in content_lines[:8]:
        if line.strip():
            draw.text((20, y_pos), line[:80], fill=(40, 40, 40), font=font_body)
            y_pos += 25
    
    # 保存
    img_path = images_dir / f"disaster_{i:03d}.png"
    img.save(img_path)
    
    # VisualDocumentオブジェクト作成
    visual_doc = VisualDocument(
        image_path=str(img_path),
        text_content=doc['content'],
        metadata={
            'doc_id': doc['doc_id'],
            'title': doc['title'],
            'disaster_type': doc['disaster_type'],
            'location': doc['location']
        }
    )
    visual_documents.append(visual_doc)

print(f"✅ {len(visual_documents)}個のVisual Document生成完了")

# ステップ5: ドキュメントエンコーディング
print("\n[ステップ 5/8] Visual Documentエンコーディング...")
start_time = time.time()

encoded_docs = []
for i, vdoc in enumerate(visual_documents):
    try:
        image = Image.open(vdoc.image_path).convert('RGB')
        
        # マルチモーダルエンコーディング
        embedding = visual_raptor.colbert_encoder.encode_multimodal(
            [vdoc.text_content[:500]],
            [image]
        )
        encoded_docs.append({
            'doc': vdoc,
            'embedding': embedding
        })
        
        if (i + 1) % 5 == 0:
            print(f"  進捗: {i + 1}/{len(visual_documents)}")
    except Exception as e:
        print(f"  ⚠️ ドキュメント {i} のエンコードエラー: {e}")

encoding_time = time.time() - start_time
print(f"✅ {len(encoded_docs)}個エンコード完了 ({encoding_time:.2f}秒)")

# ステップ6: テストクエリ生成
print("\n[ステップ 6/8] 検索クエリ生成...")
queries = doc_generator.generate_disaster_queries(num_queries=5)
print(f"✅ {len(queries)}個のクエリ生成:")
for i, q in enumerate(queries, 1):
    print(f"  {i}. {q['query']}")

# ステップ7: クエリ検索実行
print("\n[ステップ 7/8] クエリ検索実行...")
import torch
import numpy as np

for query_data in queries:
    query_text = query_data['query']
    print(f"\n🔍 クエリ: {query_text}")
    
    # クエリエンコーディング
    query_embedding = visual_raptor.colbert_encoder.encode_text([query_text])
    query_np = query_embedding.detach().cpu().numpy()
    
    # 類似度計算
    similarities = []
    for i, encoded_doc in enumerate(encoded_docs):
        doc_np = encoded_doc['embedding'].detach().cpu().numpy()
        
        # コサイン類似度
        query_norm = query_np / (np.linalg.norm(query_np) + 1e-8)
        doc_norm = doc_np / (np.linalg.norm(doc_np) + 1e-8)
        sim = np.dot(query_norm.flatten(), doc_norm.flatten())
        
        similarities.append((i, sim))
    
    # Top-3取得
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_3 = similarities[:3]
    
    print("  📊 Top 3 結果:")
    for rank, (doc_idx, sim) in enumerate(top_3, 1):
        doc = encoded_docs[doc_idx]['doc']
        print(f"    {rank}. [類似度: {sim:.4f}] {doc.metadata.get('title', 'N/A')}")
        print(f"       タイプ: {doc.metadata.get('disaster_type', 'N/A')}")
        print(f"       場所: {doc.metadata.get('location', 'N/A')}")

# ステップ8: 統計サマリー
print("\n[ステップ 8/8] 実行サマリー")
print("=" * 80)
print(f"📊 Visual RAPTOR ColBERT 実行統計")
print("=" * 80)
print(f"エンコーダー: ColModernVBERT (SigLIP)")
print(f"生成ドキュメント数: {len(visual_documents)}")
print(f"エンコード成功数: {len(encoded_docs)}")
print(f"検索クエリ数: {len(queries)}")
print(f"エンコーディング時間: {encoding_time:.2f}秒")
print(f"平均エンコード時間: {encoding_time/len(visual_documents):.3f}秒/doc")
print(f"埋め込み次元: 768")
print(f"クロスアテンション: 有効")
print(f"使用GPU: {visual_raptor.colbert_encoder.device}")

print("\n📁 出力ファイル:")
print(f"  画像: {images_dir} ({len(list(images_dir.glob('*.png')))}枚)")
print(f"  ディレクトリ: {output_dir}")

print("\n" + "=" * 80)
print("✅ Visual RAPTOR ColBERT 実行完了！")
print("=" * 80)

print("\n💡 次のステップ:")
print("  1. data/visual_raptor_run/images/ で生成画像を確認")
print("  2. 検索精度を向上させるためにドキュメント数を増やす")
print("  3. ColVBERT (BLIP) との比較を実行")
print("  4. 実際の災害画像データでテスト")

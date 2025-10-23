"""
実際のPDF災害文書を処理するスクリプト
PDFを画像に変換してVisual RAPTOR ColBERTで検索可能にする
"""

import os
import sys
from pathlib import Path
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visual_raptor_colbert import VisualRAPTORColBERT, VisualDocument
from langchain_ollama import OllamaEmbeddings, ChatOllama

print("=" * 80)
print("📄 PDF災害文書処理 - Visual RAPTOR ColBERT")
print("=" * 80)

# PDFディレクトリ
pdf_dir = Path("data/disaster_visual_documents")
output_dir = Path("data/processed_pdfs")
images_dir = output_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

# PDFファイル確認
pdf_files = list(pdf_dir.glob("*.pdf"))
print(f"\n📁 検出されたPDFファイル: {len(pdf_files)}個")
for i, pdf_file in enumerate(pdf_files, 1):
    print(f"  {i}. {pdf_file.name}")

if not pdf_files:
    print("❌ PDFファイルが見つかりません")
    sys.exit(1)

# PDF to Image 変換
print("\n[ステップ 1/6] PDFを画像に変換中...")

try:
    # PyMuPDFを使用（poppler不要）
    import fitz  # PyMuPDF
    
    all_images = []
    doc_metadata = []
    
    for pdf_file in pdf_files:
        print(f"\n  📄 処理中: {pdf_file.name}")
        
        try:
            # PDFを開く
            pdf_document = fitz.open(str(pdf_file))
            num_pages = len(pdf_document)
            
            print(f"    ✅ {num_pages}ページを検出")
            
            # 各ページを画像に変換
            for page_num in range(num_pages):
                page = pdf_document[page_num]
                
                # ページを画像に変換（150 DPI）
                mat = fitz.Matrix(150/72, 150/72)  # 150 DPI
                pix = page.get_pixmap(matrix=mat)
                
                # ファイル名を生成
                safe_name = pdf_file.stem.replace(' ', '_')
                img_path = images_dir / f"{safe_name}_page{page_num+1:03d}.png"
                
                # 画像保存
                pix.save(str(img_path))
                
                # PIL Imageとして読み込み
                image = Image.open(img_path)
                all_images.append(image)
                
                # メタデータ保存
                doc_metadata.append({
                    'pdf_name': pdf_file.name,
                    'page_number': page_num + 1,
                    'total_pages': num_pages,
                    'image_path': str(img_path)
                })
                
                if page_num < 3:
                    print(f"      - ページ {page_num+1}: {img_path.name}")
            
            if num_pages > 3:
                print(f"      ... (残り {num_pages - 3} ページ)")
            
            pdf_document.close()
                
        except Exception as e:
            print(f"    ⚠️ エラー: {e}")
            continue
    
    print(f"\n✅ 合計 {len(all_images)} ページの画像を生成")
    
except ImportError:
    print("\n⚠️ PyMuPDFがインストールされていません")
    print("インストール方法:")
    print("  pip install PyMuPDF")
    sys.exit(1)

# Ollamaモデル初期化
print("\n[ステップ 2/6] Ollamaモデル初期化...")
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
print("✅ Ollama初期化完了")

# ColModernVBERT初期化
print("\n[ステップ 3/6] ColModernVBERT (SigLIP) 初期化...")
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
print("✅ Visual RAPTOR ColBERT初期化完了")

# OCR処理（オプション）
print("\n[ステップ 4/6] OCRテキスト抽出（オプション）...")
try:
    import pytesseract
    from PIL import Image as PILImage
    
    print("  Tesseract OCRでテキスト抽出中...")
    
    for i, metadata in enumerate(doc_metadata):
        try:
            image = PILImage.open(metadata['image_path'])
            text = pytesseract.image_to_string(
                image,
                lang='jpn+eng',
                config='--psm 6'
            )
            metadata['ocr_text'] = text[:1000]  # 最初の1000文字
            
            if (i + 1) % 5 == 0:
                print(f"    進捗: {i + 1}/{len(doc_metadata)}")
        except Exception as e:
            metadata['ocr_text'] = ""
            print(f"    ⚠️ ページ {i+1} OCRエラー: {e}")
    
    print(f"  ✅ {len(doc_metadata)}ページのOCR完了")
    
except ImportError:
    print("  ⚠️ Tesseract OCRが利用できません（スキップ）")
    for metadata in doc_metadata:
        metadata['ocr_text'] = f"{metadata['pdf_name']} - ページ {metadata['page_number']}"

# Visual Documentエンコーディング
print("\n[ステップ 5/6] Visual Documentエンコーディング...")
import time
start_time = time.time()

encoded_docs = []
for i, metadata in enumerate(doc_metadata):
    try:
        image = Image.open(metadata['image_path']).convert('RGB')
        text = metadata['ocr_text']
        
        # マルチモーダルエンコーディング
        with torch.no_grad():
            embedding = visual_raptor.colbert_encoder.encode_multimodal(
                [text],
                [image]
            )
        
        encoded_docs.append({
            'metadata': metadata,
            'embedding': embedding.detach().cpu()
        })
        
        if (i + 1) % 10 == 0:
            print(f"  進捗: {i + 1}/{len(doc_metadata)}")
    
    except Exception as e:
        print(f"  ⚠️ ページ {i+1} エンコードエラー: {e}")
        continue

encoding_time = time.time() - start_time
print(f"✅ {len(encoded_docs)}ページエンコード完了 ({encoding_time:.2f}秒)")
print(f"   平均: {encoding_time/len(encoded_docs):.3f}秒/ページ")

# テスト検索
print("\n[ステップ 6/6] テスト検索実行...")

test_queries = [
    "東日本大震災の教訓",
    "避難所の運営",
    "災害対応の課題",
    "復旧復興の取り組み",
    "防災対策"
]

print(f"\n検索クエリ: {len(test_queries)}個")

import numpy as np

for query_text in test_queries:
    print(f"\n🔍 クエリ: 「{query_text}」")
    
    # クエリエンコーディング
    with torch.no_grad():
        query_embedding = visual_raptor.colbert_encoder.encode_text([query_text])
        query_np = query_embedding.detach().cpu().numpy()
    
    # 類似度計算
    similarities = []
    for i, encoded_doc in enumerate(encoded_docs):
        doc_np = encoded_doc['embedding'].numpy()
        
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
        metadata = encoded_docs[doc_idx]['metadata']
        print(f"    {rank}. [類似度: {sim:.4f}]")
        print(f"       PDF: {metadata['pdf_name']}")
        print(f"       ページ: {metadata['page_number']}/{metadata['total_pages']}")
        print(f"       画像: {Path(metadata['image_path']).name}")
        
        # OCRテキストの一部を表示
        ocr_preview = metadata['ocr_text'][:100].replace('\n', ' ')
        if ocr_preview:
            print(f"       内容: {ocr_preview}...")

# サマリー
print("\n" + "=" * 80)
print("📊 処理サマリー")
print("=" * 80)
print(f"処理PDFファイル数: {len(pdf_files)}")
for pdf_file in pdf_files:
    pdf_pages = [m for m in doc_metadata if m['pdf_name'] == pdf_file.name]
    print(f"  - {pdf_file.name}: {len(pdf_pages)}ページ")

print(f"\n総ページ数: {len(doc_metadata)}")
print(f"エンコード成功: {len(encoded_docs)}ページ")
print(f"エンコーディング時間: {encoding_time:.2f}秒")
print(f"平均エンコード速度: {encoding_time/len(encoded_docs):.3f}秒/ページ")
print(f"\n使用エンコーダー: ColModernVBERT (SigLIP)")
print(f"埋め込み次元: 768")
print(f"GPU使用: {visual_raptor.colbert_encoder.device}")

print(f"\n📁 出力ディレクトリ:")
print(f"  画像: {images_dir}")
print(f"  総画像数: {len(list(images_dir.glob('*.png')))}枚")

print("\n" + "=" * 80)
print("✅ PDF災害文書処理完了！")
print("=" * 80)

print("\n💡 次のステップ:")
print("  1. data/processed_pdfs/images/ で生成画像を確認")
print("  2. 検索結果を確認して精度を評価")
print("  3. より多くのクエリでテスト")
print("  4. ファインチューニングや最適化を検討")

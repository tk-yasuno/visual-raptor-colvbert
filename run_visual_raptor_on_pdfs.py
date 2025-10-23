"""
Visual RAPTOR ColBERT - 実際のPDF画像（131枚）に対する実行
================================================

処理されたPDF画像に対してVisual RAPTORを実行し、
階層的クラスタリングとセマンティック検索を実行します。
"""

import os
import sys
import time
import json
from pathlib import Path
from PIL import Image
from datetime import datetime
from typing import List, Tuple, Dict
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI不要のバックエンド
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

# visual_raptor_colbert.pyをインポート
from visual_raptor_colbert import (
    VisualRAPTORColBERT,
    ColModernVBERTEncoder,
    ColVBERTEncoder
)

def load_pdf_images(images_dir: str):
    """PDF画像を読み込む"""
    images_path = Path(images_dir)
    
    if not images_path.exists():
        raise FileNotFoundError(f"画像ディレクトリが見つかりません: {images_dir}")
    
    # PNG画像を取得
    image_files = sorted(images_path.glob("*.png"))
    
    if len(image_files) == 0:
        raise FileNotFoundError(f"画像が見つかりません: {images_dir}")
    
    print(f"\n📁 画像ディレクトリ: {images_dir}")
    print(f"📊 検出された画像数: {len(image_files)}")
    
    # 画像を読み込み
    images = []
    metadata = []
    
    print("\n🖼️ 画像読み込み中...")
    for i, img_path in enumerate(image_files, 1):
        try:
            image = Image.open(img_path).convert('RGB')
            images.append(image)
            
            # メタデータを抽出（ファイル名から）
            filename = img_path.stem
            if "東日本大震災の教訓集" in filename:
                pdf_name = "東日本大震災の教訓集"
            elif "令和6年度の災害を中心とした事例集" in filename:
                pdf_name = "令和6年度の災害事例集"
            else:
                pdf_name = "不明"
            
            # ページ番号を抽出
            page_num = filename.split("_page")[-1]
            
            metadata.append({
                'pdf_name': pdf_name,
                'page_number': page_num,
                'filename': img_path.name,
                'path': str(img_path)
            })
            
            if i % 20 == 0:
                print(f"  進捗: {i}/{len(image_files)}")
                
        except Exception as e:
            print(f"  ⚠️ エラー ({img_path.name}): {e}")
            continue
    
    print(f"✅ {len(images)}枚の画像を読み込み完了\n")
    return images, metadata

def plot_siglip_metrics(all_results: Dict, output_dir: Path):
    """SigLIP評価指標をグラフ化して保存"""
    
    queries = [q['query'] for q in all_results['queries']]
    
    # 6つの指標を抽出
    variances = [q['siglip_metrics']['variance'] for q in all_results['queries']]
    entropies = [q['siglip_metrics']['normalized_entropy'] for q in all_results['queries']]
    confidences = [q['siglip_metrics']['confidence'] for q in all_results['queries']]
    dominances = [q['siglip_metrics']['relative_dominance'] for q in all_results['queries']]
    qualities = [q['siglip_metrics']['ranking_quality'] for q in all_results['queries']]
    decays = [q['siglip_metrics']['score_decay_rate'] for q in all_results['queries']]
    
    # 図のサイズと配置
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('SigLIP評価指標 - クエリ別比較', fontsize=16, fontweight='bold')
    
    # クエリ番号（X軸）
    x_pos = np.arange(len(queries))
    
    # 1. 分散 (Variance)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x_pos, variances, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_title('① 分散 (Variance)\n結果の多様性', fontsize=12, fontweight='bold')
    ax1.set_xlabel('クエリ番号', fontsize=10)
    ax1.set_ylabel('分散値', fontsize=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Q{i+1}' for i in range(len(queries))], rotation=0)
    ax1.grid(axis='y', alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 2. 正規化エントロピー (Normalized Entropy)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x_pos, entropies, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_title('② 正規化エントロピー (Entropy)\n結果の不確実性 (0:確信的, 1:不確実)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('クエリ番号', fontsize=10)
    ax2.set_ylabel('エントロピー', fontsize=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Q{i+1}' for i in range(len(queries))], rotation=0)
    ax2.set_ylim([0, 1.1])
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. 信頼度 (Confidence)
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x_pos, confidences, color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax3.set_title('③ 信頼度 (Confidence)\nTop1とTop2の差 (高いほど明確)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('クエリ番号', fontsize=10)
    ax3.set_ylabel('信頼度', fontsize=10)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'Q{i+1}' for i in range(len(queries))], rotation=0)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. 相対優位性 (Relative Dominance)
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x_pos, dominances, color='orange', alpha=0.7, edgecolor='black')
    ax4.set_title('④ 相対優位性 (Dominance)\nTop1が平均を超える度合い', fontsize=12, fontweight='bold')
    ax4.set_xlabel('クエリ番号', fontsize=10)
    ax4.set_ylabel('優位性 (標準偏差単位)', fontsize=10)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'Q{i+1}' for i in range(len(queries))], rotation=0)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. ランキング品質 (Ranking Quality)
    ax5 = axes[2, 0]
    bars5 = ax5.bar(x_pos, qualities, color='mediumpurple', alpha=0.7, edgecolor='black')
    ax5.set_title('⑤ ランキング品質 (Quality)\nDCG風スコア (高いほど良好)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('クエリ番号', fontsize=10)
    ax5.set_ylabel('品質スコア', fontsize=10)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([f'Q{i+1}' for i in range(len(queries))], rotation=0)
    ax5.grid(axis='y', alpha=0.3)
    ax5.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # 6. スコア減衰率 (Score Decay Rate)
    ax6 = axes[2, 1]
    bars6 = ax6.bar(x_pos, decays, color='crimson', alpha=0.7, edgecolor='black')
    ax6.set_title('⑥ スコア減衰率 (Decay)\nランキングの滑らかさ (低いほど滑らか)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('クエリ番号', fontsize=10)
    ax6.set_ylabel('減衰率', fontsize=10)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([f'Q{i+1}' for i in range(len(queries))], rotation=0)
    ax6.grid(axis='y', alpha=0.3)
    ax6.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # レイアウト調整
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"siglip_metrics_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 グラフを保存しました: {output_file}")
    
    # クエリリストも保存
    query_legend_file = output_dir / f"query_legend_{timestamp}.txt"
    with open(query_legend_file, 'w', encoding='utf-8') as f:
        f.write("クエリ凡例\n")
        f.write("=" * 50 + "\n")
        for i, query in enumerate(queries, 1):
            f.write(f"Q{i}: {query}\n")
    
    print(f"📝 クエリ凡例を保存: {query_legend_file}")

def run_visual_raptor_on_pdfs():
    """実際のPDF画像に対してVisual RAPTORを実行"""
    
    print("=" * 80)
    print("🎯 Visual RAPTOR ColBERT - 実際のPDF画像処理")
    print("=" * 80)
    
    # 画像ディレクトリ
    images_dir = "data/processed_pdfs/images"
    
    # ステップ1: 画像読み込み
    print("\n[ステップ 1/4] PDF画像読み込み")
    print("-" * 80)
    images, metadata = load_pdf_images(images_dir)
    
    # ステップ2: Visual RAPTOR初期化
    print("\n[ステップ 2/4] Visual RAPTOR初期化")
    print("-" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 デバイス: {device}")
    
    # Ollama初期化（テキスト埋め込みとLLM）
    from langchain_ollama import OllamaEmbeddings
    from langchain_ollama.llms import OllamaLLM
    
    print("🔧 Ollama初期化中...")
    embeddings_model = OllamaEmbeddings(model="mxbai-embed-large")
    llm = OllamaLLM(model="granite-code:8b")
    print("✅ Ollama初期化完了")
    
    # ColModernVBERT (SigLIP) を使用
    print("🚀 エンコーダー: ColModernVBERT (SigLIP)")
    
    # PDF元ファイルのディレクトリ（テキスト抽出用）
    pdf_source_dir = "data/disaster_visual_documents"
    
    raptor = VisualRAPTORColBERT(
        embeddings_model=embeddings_model,
        llm=llm,
        use_modern_vbert=True,  # ColModernVBERT (SigLIP)を使用
        pdf_source_dir=pdf_source_dir,  # PDFテキスト抽出用
        min_clusters=2,         # 最小クラスタ数
        max_clusters=10,        # 最大クラスタ数
        max_depth=3,            # 階層の最大深さ
    )
    
    print("✅ Visual RAPTOR初期化完了\n")
    
    # ステップ3: 画像をVisual Documentsとして読み込み
    print("\n[ステップ 3/4] Visual Documents読み込み & エンコーディング")
    print("-" * 80)
    
    start_time = time.time()
    
    # 画像ディレクトリから直接読み込み
    print(f"📝 画像ディレクトリから読み込み中...")
    visual_docs = raptor.load_visual_documents(
        image_directory=images_dir,
        supported_formats=['.png']
    )
    
    print(f"✅ {len(visual_docs)}枚のVisual Documents読み込み完了")
    
    # ビジュアルインデックス構築
    print("\n🔨 ビジュアルインデックス構築中...")
    index_start = time.time()
    raptor.build_visual_index()
    index_time = time.time() - index_start
    
    total_time = time.time() - start_time
    
    print(f"✅ ビジュアルインデックス構築完了 ({index_time:.2f}秒)")
    print(f"⏱️ 総処理時間: {total_time:.2f}秒")
    print(f"📊 平均処理速度: {total_time/len(visual_docs):.3f}秒/ページ")
    
    # ステップ4: テスト検索
    print("\n[ステップ 4/4] セマンティック検索テスト")
    print("-" * 80)
    
    # 災害関連のクエリ
    test_queries = [
        "東日本大震災の教訓",
        "避難所の運営方法",
        "災害対応の課題と改善点",
        "復旧復興の取り組み",
        "防災対策の重要性",
        "津波の被害状況",
        "地震発生時の対応",
        "被災者支援の実践例"
    ]
    
    print(f"\n🔍 テストクエリ: {len(test_queries)}個\n")
    
    # 評価結果を保存
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'encoder': 'ColModernVBERT (SigLIP)',
            'embedding_dim': 768,
            'device': device,
            'total_documents': len(visual_docs)
        },
        'queries': []
    }
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"🔍 クエリ {i}: 「{query}」")
        print('='*70)
        
        # 検索実行
        search_start = time.time()
        results = raptor.search_visual_documents(
            query=query,
            top_k=5
        )
        search_time = time.time() - search_start
        
        print(f"⏱️ 検索時間: {search_time*1000:.2f}ms")
        
        # SigLIP用評価指標計算
        scores = np.array([score for _, score in results])
        
        # 基本統計
        mean_score = np.mean(scores) if len(scores) > 0 else 0
        max_score = np.max(scores) if len(scores) > 0 else 0
        min_score = np.min(scores) if len(scores) > 0 else 0
        score_range = max_score - min_score
        std_score = np.std(scores) if len(scores) > 0 else 0
        
        # SigLIP特有の指標
        # 1. スコア分散 (Score Variance) - 結果の多様性
        variance = np.var(scores) if len(scores) > 0 else 0
        
        # 2. 正規化エントロピー (Normalized Entropy) - 結果の不確実性
        # スコアを確率分布に変換（softmax風）
        if len(scores) > 0 and score_range > 1e-10:
            score_probs = np.exp(scores - np.max(scores))
            score_probs = score_probs / np.sum(score_probs)
            entropy = -np.sum(score_probs * np.log(score_probs + 1e-10))
            normalized_entropy = entropy / np.log(len(scores))  # 正規化
        else:
            entropy = 0
            normalized_entropy = 0
        
        # 3. 信頼度スコア (Confidence Score) - Top1とTop2の差
        confidence = scores[0] - scores[1] if len(scores) > 1 else 0
        
        # 4. 相対的優位性 (Relative Dominance) - Top1 vs 平均
        relative_dominance = (scores[0] - mean_score) / (std_score + 1e-10) if std_score > 0 else 0
        
        # 5. ランキング品質 (Ranking Quality) - DCG風の指標
        # Top結果ほど高スコアであるべき
        ranking_quality = 0
        for idx, score in enumerate(scores):
            ranking_quality += score / np.log2(idx + 2)  # NDCG風
        
        # 6. スコア減衰率 (Score Decay Rate) - ランキングの一貫性
        if len(scores) > 1:
            score_diffs = np.diff(scores)  # 隣接スコア差
            decay_rate = np.mean(np.abs(score_diffs))
        else:
            decay_rate = 0
        
        print(f"\n📈 基本統計:")
        print(f"   平均スコア: {mean_score:.4f}")
        print(f"   最大スコア: {max_score:.4f}")
        print(f"   最小スコア: {min_score:.4f}")
        print(f"   スコア範囲: {score_range:.4f}")
        print(f"   標準偏差: {std_score:.4f}")
        
        print(f"\n🎯 SigLIP評価指標:")
        print(f"   分散 (Variance): {variance:.6f}")
        print(f"   正規化エントロピー (Entropy): {normalized_entropy:.4f}")
        print(f"   信頼度 (Confidence): {confidence:.4f}")
        print(f"   相対優位性 (Dominance): {relative_dominance:.4f}")
        print(f"   ランキング品質 (Quality): {ranking_quality:.4f}")
        print(f"   スコア減衰率 (Decay): {decay_rate:.6f}")
        
        print(f"\n📊 Top 5 結果:\n")
        
        query_results = {
            'query': query,
            'search_time_ms': search_time * 1000,
            'basic_statistics': {
                'mean_score': float(mean_score),
                'max_score': float(max_score),
                'min_score': float(min_score),
                'score_range': float(score_range),
                'std_score': float(std_score)
            },
            'siglip_metrics': {
                'variance': float(variance),
                'normalized_entropy': float(normalized_entropy),
                'confidence': float(confidence),
                'relative_dominance': float(relative_dominance),
                'ranking_quality': float(ranking_quality),
                'score_decay_rate': float(decay_rate)
            },
            'top_results': []
        }
        
        for rank, (visual_doc, score) in enumerate(results, 1):
            # ファイル名からメタデータを抽出
            filename = Path(visual_doc.image_path).name
            if "201205_EastJapanQuakeLesson" in filename:
                pdf_name = "東日本大震災の教訓集"
            elif "202505_Reiwa6DisasterExamples" in filename:
                pdf_name = "令和6年度の災害事例集"
            else:
                pdf_name = "不明"
            
            # ページ番号を抽出
            page_num = filename.split("_page")[-1].replace(".png", "")
            
            # テキスト内容のプレビュー(最初の200文字)
            text_preview = visual_doc.text_content[:200] if visual_doc.text_content else ""
            text_preview = text_preview.replace('\n', ' ').strip()
            
            print(f"  {rank}. [類似度: {score:.4f}]")
            print(f"     📄 PDF: {pdf_name}")
            print(f"     📖 ページ: {page_num}")
            print(f"     🖼️ ファイル: {filename}")
            print(f"     📝 テキスト長: {len(visual_doc.text_content)}文字")
            
            if text_preview:
                print(f"     💬 内容プレビュー: {text_preview}...")
            
            print()
            
            # 結果を保存
            query_results['top_results'].append({
                'rank': rank,
                'score': float(score),
                'pdf_name': pdf_name,
                'page_number': page_num,
                'filename': filename,
                'text_length': len(visual_doc.text_content),
                'text_preview': text_preview,
                'full_text': visual_doc.text_content
            })
        
        all_results['queries'].append(query_results)
    
    # 全クエリの集計指標を計算
    all_confidences = [q['siglip_metrics']['confidence'] for q in all_results['queries']]
    all_entropies = [q['siglip_metrics']['normalized_entropy'] for q in all_results['queries']]
    all_qualities = [q['siglip_metrics']['ranking_quality'] for q in all_results['queries']]
    all_decays = [q['siglip_metrics']['score_decay_rate'] for q in all_results['queries']]
    
    aggregate_metrics = {
        'average_confidence': float(np.mean(all_confidences)),
        'average_entropy': float(np.mean(all_entropies)),
        'average_ranking_quality': float(np.mean(all_qualities)),
        'average_decay_rate': float(np.mean(all_decays)),
        'confidence_std': float(np.std(all_confidences)),
        'entropy_std': float(np.std(all_entropies))
    }
    
    all_results['aggregate_metrics'] = aggregate_metrics
    
    # 結果をJSONファイルに保存
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"visual_raptor_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 検索結果を保存しました: {output_file}")
    print(f"   総クエリ数: {len(test_queries)}")
    print(f"   各クエリTop5結果 + 完全テキスト内容を含む")
    
    # SigLIP評価指標のグラフを生成
    print(f"\n📊 SigLIP評価指標のグラフを生成中...")
    plot_siglip_metrics(all_results, output_dir)
    
    print(f"\n📊 全体集計指標 (SigLIP):")
    print(f"   平均信頼度: {aggregate_metrics['average_confidence']:.4f} ± {aggregate_metrics['confidence_std']:.4f}")
    print(f"   平均エントロピー: {aggregate_metrics['average_entropy']:.4f} ± {aggregate_metrics['entropy_std']:.4f}")
    print(f"   平均ランキング品質: {aggregate_metrics['average_ranking_quality']:.4f}")
    print(f"   平均スコア減衰率: {aggregate_metrics['average_decay_rate']:.6f}")
    
    # 最終サマリー
    print("\n" + "=" * 80)
    print("📊 実行サマリー")
    print("=" * 80)
    print(f"処理画像数: {len(visual_docs)}枚")
    print(f"総処理時間: {total_time:.2f}秒")
    print(f"  - 画像読み込み & インデックス構築: {index_time:.2f}秒")
    print(f"平均処理速度: {total_time/len(visual_docs):.3f}秒/ページ")
    print(f"\nビジュアルインデックス:")
    print(f"  - 総ドキュメント数: {len(raptor.visual_documents)}")
    print(f"  - 埋め込み次元: {raptor.visual_embeddings.shape[1] if raptor.visual_embeddings is not None else '?'}")
    print(f"\n使用エンコーダー: ColModernVBERT (SigLIP)")
    print(f"埋め込み次元: 768")
    print(f"デバイス: cuda")
    
    print("\n" + "=" * 80)
    print("✅ Visual RAPTOR処理完了！")
    print("=" * 80)
    
    return raptor, images, metadata

if __name__ == "__main__":
    try:
        raptor, visual_docs, _ = run_visual_raptor_on_pdfs()
        
        print("\n💡 次のステップ:")
        print("  1. 検索結果の精度を評価")
        print("  2. より多様なクエリでテスト")
        print("  3. 特定のPDFページを詳細に調査")
        print("  4. OCRテキストを追加してマルチモーダル検索を強化")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

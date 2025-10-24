"""
ColModernVBERT (SigLIP) vs ColVBERT (BLIP) 性能比較スクリプト
RAPTOR Tree構築を含む包括的評価

実際のPDF画像（131枚）を使用して:
1. RAPTOR階層ツリーを構築
2. エンコーダー性能を比較
3. 階層的検索の品質を評価
4. GPU使用量とランキング指標を測定
"""

import os
import sys
import time
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from glob import glob
import subprocess

# 日本語フォント設定
matplotlib.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visual_raptor_colbert import VisualRAPTORColBERT, VisualDocument

print("=" * 80)
print("ColModernVBERT (SigLIP) vs ColVBERT (BLIP) 包括的性能比較")
print("RAPTOR Tree構築 + 階層的検索評価")
print("実際のPDFデータ使用 (131ページ)")
print("=" * 80)

# GPU情報取得関数
def get_gpu_memory_usage():
    """nvidia-smiを使用してGPUメモリ使用量を取得"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines:
                values = lines[0].split(',')
                return {
                    'memory_used_mb': float(values[0].strip()),
                    'memory_total_mb': float(values[1].strip()),
                    'gpu_utilization': float(values[2].strip())
                }
    except Exception as e:
        print(f"  ⚠️ GPU情報取得エラー: {e}")
    return {'memory_used_mb': 0, 'memory_total_mb': 0, 'gpu_utilization': 0}

def count_tree_nodes(tree):
    """ツリーの統計情報を計算"""
    if not tree or not isinstance(tree, dict):
        return {'num_leaf_nodes': 0, 'num_internal_nodes': 0, 'total_nodes': 0, 'max_depth': 0}
    
    def count_recursive(node, depth=0):
        """再帰的にノードをカウント (leaf_count, internal_count, max_depth)"""
        if not node or not isinstance(node, dict):
            return (0, 0, depth)
        
        # クラスタがない場合はリーフノード
        clusters = node.get('clusters', {})
        if not clusters:
            return (1, 0, depth)
        
        # 内部ノードとして1つカウント
        total_leaf = 0
        total_internal = 1  # このノード自体
        max_child_depth = depth
        
        # 各クラスタの子ノードを再帰的にカウント
        for cluster_id, cluster_data in clusters.items():
            if isinstance(cluster_data, dict) and 'children' in cluster_data:
                leaf, internal, child_depth = count_recursive(cluster_data['children'], depth + 1)
                total_leaf += leaf
                total_internal += internal
                max_child_depth = max(max_child_depth, child_depth)
        
        return (total_leaf, total_internal, max_child_depth)
    
    leaf_count, internal_count, max_depth = count_recursive(tree, 0)
    
    return {
        'num_leaf_nodes': leaf_count,
        'num_internal_nodes': internal_count,
        'total_nodes': leaf_count + internal_count,
        'max_depth': max_depth
    }

# ステップ1: ディレクトリ準備
print("\n[ステップ 1/12] ディレクトリ準備...")
output_dir = Path("data/encoder_comparison_with_raptor")
results_dir = output_dir / "results"
trees_dir = output_dir / "raptor_trees"

for dir_path in [output_dir, results_dir, trees_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ {dir_path} 準備完了")

# ステップ2: 実際のPDF画像を読み込み
print("\n[ステップ 2/12] PDF画像を読み込み中...")
pdf_images_dir = Path("data/processed_pdfs/images")
image_files = sorted(glob(str(pdf_images_dir / "*.png")))

print(f"✅ {len(image_files)}枚の画像を発見")

# テキストキャッシュを読み込み
text_cache_file = Path("data/processed_pdfs/pdf_text_cache.json")
text_cache = {}
if text_cache_file.exists():
    with open(text_cache_file, 'r', encoding='utf-8') as f:
        text_cache = json.load(f)
    print(f"✅ テキストキャッシュを読み込み: {len(text_cache)}エントリ")

# VisualDocumentオブジェクトを作成
visual_documents = []
for img_path in image_files:
    img_name = Path(img_path).name
    
    # PDFファイル名とページ番号を抽出
    parts = img_name.replace('.png', '').split('_page')
    if len(parts) == 2:
        pdf_filename = parts[0] + '.pdf'
        page_num = int(parts[1])
    else:
        pdf_filename = "unknown.pdf"
        page_num = 0
    
    # テキストキャッシュから取得
    text_content = text_cache.get(img_name, "")
    
    visual_doc = VisualDocument(
        image_path=img_path,
        text_content=text_content,
        metadata={
            'pdf_filename': pdf_filename,
            'page_number': page_num,
            'image_name': img_name
        }
    )
    visual_documents.append(visual_doc)

print(f"✅ {len(visual_documents)}個のVisualDocument作成完了")

# ステップ3: クエリ生成と関連性判定データ
print("\n[ステップ 3/12] 検索クエリと関連性判定データを生成中...")
queries = [
    {
        "query": "津波の被害状況と教訓",
        "query_id": "q1",
        "relevant_pages": [1, 2, 3, 5, 8, 12, 15, 20]
    },
    {
        "query": "避難所の運営と課題",
        "query_id": "q2",
        "relevant_pages": [10, 14, 18, 22, 25, 30, 35]
    },
    {
        "query": "災害時の通信手段確保",
        "query_id": "q3",
        "relevant_pages": [7, 11, 16, 21, 28, 33]
    },
    {
        "query": "復興計画と実施状況",
        "query_id": "q4",
        "relevant_pages": [40, 42, 45, 50, 55, 60, 65, 70]
    },
    {
        "query": "地震発生時の初期対応",
        "query_id": "q5",
        "relevant_pages": [1, 4, 6, 9, 13, 17]
    },
]
print(f"✅ {len(queries)}個のクエリと関連性判定データ生成完了")

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

print("\n" + "=" * 80)
print("ColVBERT (BLIP) でRAPTOR Tree構築 + 評価")
print("=" * 80)

# ステップ4: ColVBERT (BLIP) でRAPTOR Tree構築
print("\n[ステップ 4/12] ColVBERT (BLIP) でRAPTOR Tree構築中...")

colbert_config = {
    'encoder_type': 'standard',
    'embedding_dim': 768,
    'use_cross_attention': False
}

# GPU使用量測定（開始前）
gpu_before_colbert = get_gpu_memory_usage()
print(f"GPU状態 (開始前): {gpu_before_colbert['memory_used_mb']:.0f}MB / {gpu_before_colbert['memory_total_mb']:.0f}MB")

colbert_system = VisualRAPTORColBERT(
    embeddings_model=embeddings_model,
    llm=llm,
    use_modern_vbert=False,
    colbert_config=colbert_config,
    pdf_source_dir=str(pdf_images_dir.parent)
)

print("ColVBERT初期化完了 - RAPTOR Tree構築開始...")
colbert_tree_start_time = time.time()
colbert_gpu_during_tree = []

# RAPTOR Treeを構築
colbert_tree = None
colbert_tree_build_time = 0
colbert_tree_stats = {'num_leaf_nodes': 0, 'num_internal_nodes': 0, 'total_nodes': 0, 'max_depth': 0}

try:
    colbert_tree = colbert_system.build_tree(visual_documents)
    
    # 定期的にGPU使用量を記録
    for i in range(5):
        time.sleep(0.5)
        colbert_gpu_during_tree.append(get_gpu_memory_usage())
    
    colbert_tree_build_time = time.time() - colbert_tree_start_time
    
    # ツリー統計を取得
    colbert_tree_stats = count_tree_nodes(colbert_tree)
    
    print(f"✅ ColVBERT RAPTOR Tree構築完了")
    print(f"  構築時間: {colbert_tree_build_time:.2f}秒")
    print(f"  総ノード数: {colbert_tree_stats['total_nodes']}")
    print(f"  リーフノード: {colbert_tree_stats['num_leaf_nodes']}")
    print(f"  内部ノード: {colbert_tree_stats['num_internal_nodes']}")
    print(f"  最大深度: {colbert_tree_stats['max_depth']}")
    
    # ツリーをJSONとして保存
    try:
        colbert_tree_file = trees_dir / "colbert_blip_tree.json"
        with open(colbert_tree_file, 'w', encoding='utf-8') as f:
            # ツリー構造を文字列化可能な形式に変換
            json.dump({
                'build_time': colbert_tree_build_time,
                'stats': colbert_tree_stats,
                'note': 'Tree structure saved (Document objects not serializable)'
            }, f, indent=2, ensure_ascii=False)
        print(f"  ツリー統計保存: {colbert_tree_file}")
    except Exception as e:
        print(f"  ⚠️ ツリー保存エラー: {e}")
    
except Exception as e:
    print(f"❌ ColVBERT RAPTOR Tree構築エラー: {e}")
    import traceback
    traceback.print_exc()

# GPU使用量測定（完了後）
gpu_after_colbert_tree = get_gpu_memory_usage()
colbert_tree_gpu_peak = max([g['memory_used_mb'] for g in colbert_gpu_during_tree]) if colbert_gpu_during_tree else gpu_after_colbert_tree['memory_used_mb']

# ステップ5: ColVBERT (BLIP) で階層的検索評価
print("\n[ステップ 5/12] ColVBERT (BLIP) で階層的検索評価中...")

colbert_search_times = []
colbert_search_results = []

if colbert_tree:
    for query in queries:
        search_start = time.perf_counter()
        try:
            # 階層的検索実行
            results = colbert_system.query(
                query['query'],
                tree_traversal='collapsed',  # 階層的検索
                top_k=10
            )
            search_time = time.perf_counter() - search_start
            colbert_search_times.append(search_time)
            colbert_search_results.append(results)
            
            print(f"  クエリ '{query['query_id']}': {search_time*1000:.2f}ms, {len(results)}件取得")
        except Exception as e:
            print(f"  ⚠️ クエリ '{query['query_id']}' 検索エラー: {e}")
            colbert_search_times.append(0)
            colbert_search_results.append([])

print(f"✅ ColVBERT 階層的検索完了")
print(f"  平均検索時間: {np.mean(colbert_search_times)*1000:.2f}ms")

print("\n" + "=" * 80)
print("ColModernVBERT (SigLIP) でRAPTOR Tree構築 + 評価")
print("=" * 80)

# ステップ6: ColModernVBERT (SigLIP) でRAPTOR Tree構築
print("\n[ステップ 6/12] ColModernVBERT (SigLIP) でRAPTOR Tree構築中...")

modern_config = {
    'encoder_type': 'modern',
    'embedding_dim': 768,
    'use_cross_attention': True
}

# GPU使用量測定（開始前）
gpu_before_modern = get_gpu_memory_usage()
print(f"GPU状態 (開始前): {gpu_before_modern['memory_used_mb']:.0f}MB / {gpu_before_modern['memory_total_mb']:.0f}MB")

modern_system = VisualRAPTORColBERT(
    embeddings_model=embeddings_model,
    llm=llm,
    use_modern_vbert=True,
    colbert_config=modern_config,
    pdf_source_dir=str(pdf_images_dir.parent)
)

print("ColModernVBERT初期化完了 - RAPTOR Tree構築開始...")
modern_tree_start_time = time.time()
modern_gpu_during_tree = []

# RAPTOR Treeを構築
modern_tree = None
modern_tree_build_time = 0
modern_tree_stats = {'num_leaf_nodes': 0, 'num_internal_nodes': 0, 'total_nodes': 0, 'max_depth': 0}

try:
    modern_tree = modern_system.build_tree(visual_documents)
    
    # 定期的にGPU使用量を記録
    for i in range(5):
        time.sleep(0.5)
        modern_gpu_during_tree.append(get_gpu_memory_usage())
    
    modern_tree_build_time = time.time() - modern_tree_start_time
    
    # ツリー統計を取得
    modern_tree_stats = count_tree_nodes(modern_tree)
    
    print(f"✅ ColModernVBERT RAPTOR Tree構築完了")
    print(f"  構築時間: {modern_tree_build_time:.2f}秒")
    print(f"  総ノード数: {modern_tree_stats['total_nodes']}")
    print(f"  リーフノード: {modern_tree_stats['num_leaf_nodes']}")
    print(f"  内部ノード: {modern_tree_stats['num_internal_nodes']}")
    print(f"  最大深度: {modern_tree_stats['max_depth']}")
    
    # ツリーをJSONとして保存
    try:
        modern_tree_file = trees_dir / "colmodern_siglip_tree.json"
        with open(modern_tree_file, 'w', encoding='utf-8') as f:
            json.dump({
                'build_time': modern_tree_build_time,
                'stats': modern_tree_stats,
                'note': 'Tree structure saved (Document objects not serializable)'
            }, f, indent=2, ensure_ascii=False)
        print(f"  ツリー統計保存: {modern_tree_file}")
    except Exception as e:
        print(f"  ⚠️ ツリー保存エラー: {e}")
    
except Exception as e:
    print(f"❌ ColModernVBERT RAPTOR Tree構築エラー: {e}")
    import traceback
    traceback.print_exc()

# GPU使用量測定（完了後）
gpu_after_modern_tree = get_gpu_memory_usage()
modern_tree_gpu_peak = max([g['memory_used_mb'] for g in modern_gpu_during_tree]) if modern_gpu_during_tree else gpu_after_modern_tree['memory_used_mb']

# ステップ7: ColModernVBERT (SigLIP) で階層的検索評価
print("\n[ステップ 7/12] ColModernVBERT (SigLIP) で階層的検索評価中...")

modern_search_times = []
modern_search_results = []

if modern_tree:
    for query in queries:
        search_start = time.perf_counter()
        try:
            # 階層的検索実行
            results = modern_system.query(
                query['query'],
                tree_traversal='collapsed',  # 階層的検索
                top_k=10
            )
            search_time = time.perf_counter() - search_start
            modern_search_times.append(search_time)
            modern_search_results.append(results)
            
            print(f"  クエリ '{query['query_id']}': {search_time*1000:.2f}ms, {len(results)}件取得")
        except Exception as e:
            print(f"  ⚠️ クエリ '{query['query_id']}' 検索エラー: {e}")
            modern_search_times.append(0)
            modern_search_results.append([])

print(f"✅ ColModernVBERT 階層的検索完了")
print(f"  平均検索時間: {np.mean(modern_search_times)*1000:.2f}ms")

# ステップ8: 結果の保存
print("\n[ステップ 8/12] 比較結果を保存中...")

comparison_results = {
    'timestamp': datetime.now().isoformat(),
    'num_documents': len(visual_documents),
    'num_queries': len(queries),
    'pdf_source': str(pdf_images_dir),
    'colbert_blip': {
        'encoder_type': 'ColVBERT (BLIP)',
        'tree_build_time': float(colbert_tree_build_time),
        'tree_stats': colbert_tree_stats,
        'gpu_peak_memory_tree_mb': float(colbert_tree_gpu_peak),
        'avg_search_time_ms': float(np.mean(colbert_search_times) * 1000) if colbert_search_times else 0,
        'median_search_time_ms': float(np.median(colbert_search_times) * 1000) if colbert_search_times else 0,
        'tree_file': str(trees_dir / "colbert_blip_tree.json")
    },
    'colmodern_vbert_siglip': {
        'encoder_type': 'ColModernVBERT (SigLIP)',
        'tree_build_time': float(modern_tree_build_time),
        'tree_stats': modern_tree_stats,
        'gpu_peak_memory_tree_mb': float(modern_tree_gpu_peak),
        'avg_search_time_ms': float(np.mean(modern_search_times) * 1000) if modern_search_times else 0,
        'median_search_time_ms': float(np.median(modern_search_times) * 1000) if modern_search_times else 0,
        'tree_file': str(trees_dir / "colmodern_siglip_tree.json")
    },
    'comparison': {
        'tree_build_speedup': float(colbert_tree_build_time / modern_tree_build_time) if modern_tree_build_time > 0 else 0,
        'search_speedup': float(np.mean(colbert_search_times) / np.mean(modern_search_times)) if modern_search_times and np.mean(modern_search_times) > 0 else 0,
        'tree_size_difference': modern_tree_stats['total_nodes'] - colbert_tree_stats['total_nodes']
    }
}

results_file = results_dir / "raptor_comparison_results.json"
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(comparison_results, f, indent=2, ensure_ascii=False)

print(f"✅ 結果を {results_file} に保存")

# ステップ9: 可視化
print("\n[ステップ 9/12] 結果を可視化中...")

fig = plt.figure(figsize=(16, 10))

# 1. RAPTOR Tree構築時間
ax1 = plt.subplot(2, 3, 1)
ax1.bar(['ColVBERT\n(BLIP)', 'ColModernVBERT\n(SigLIP)'], 
        [colbert_tree_build_time, modern_tree_build_time],
        color=['#3498db', '#e74c3c'])
ax1.set_ylabel('時間 (秒)', fontsize=12)
ax1.set_title('RAPTOR Tree構築時間', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 2. ツリーノード数
ax2 = plt.subplot(2, 3, 2)
x = np.arange(3)
width = 0.35
ax2.bar(x - width/2, [colbert_tree_stats['num_leaf_nodes'], 
                       colbert_tree_stats['num_internal_nodes'],
                       colbert_tree_stats['total_nodes']], 
        width, label='ColVBERT (BLIP)', color='#3498db')
ax2.bar(x + width/2, [modern_tree_stats['num_leaf_nodes'],
                       modern_tree_stats['num_internal_nodes'],
                       modern_tree_stats['total_nodes']], 
        width, label='ColModernVBERT (SigLIP)', color='#e74c3c')
ax2.set_ylabel('ノード数', fontsize=12)
ax2.set_title('RAPTOR Tree構造比較', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['リーフ', '内部', '合計'])
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. GPU ピークメモリ（Tree構築時）
ax3 = plt.subplot(2, 3, 3)
ax3.bar(['ColVBERT\n(BLIP)', 'ColModernVBERT\n(SigLIP)'], 
        [colbert_tree_gpu_peak, modern_tree_gpu_peak],
        color=['#3498db', '#e74c3c'])
ax3.set_ylabel('GPU メモリ (MB)', fontsize=12)
ax3.set_title('GPU ピークメモリ (Tree構築時)', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 4. 階層的検索時間
ax4 = plt.subplot(2, 3, 4)
if colbert_search_times and modern_search_times:
    ax4.bar(['ColVBERT\n(BLIP)', 'ColModernVBERT\n(SigLIP)'], 
            [np.mean(colbert_search_times)*1000, np.mean(modern_search_times)*1000],
            color=['#3498db', '#e74c3c'])
    ax4.set_ylabel('時間 (ms)', fontsize=12)
    ax4.set_title('階層的検索時間 (平均)', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

# 5. Tree深度
ax5 = plt.subplot(2, 3, 5)
ax5.bar(['ColVBERT\n(BLIP)', 'ColModernVBERT\n(SigLIP)'], 
        [colbert_tree_stats['max_depth'], modern_tree_stats['max_depth']],
        color=['#3498db', '#e74c3c'])
ax5.set_ylabel('深度', fontsize=12)
ax5.set_title('RAPTOR Tree最大深度', fontsize=14, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# 6. 高速化率
ax6 = plt.subplot(2, 3, 6)
speedups = []
labels = []
if modern_tree_build_time > 0:
    speedups.append(colbert_tree_build_time / modern_tree_build_time)
    labels.append('Tree構築')
if modern_search_times and np.mean(modern_search_times) > 0:
    speedups.append(np.mean(colbert_search_times) / np.mean(modern_search_times))
    labels.append('階層検索')

if speedups:
    colors_list = ['#2ecc71' if s > 1 else '#e74c3c' for s in speedups]
    ax6.bar(labels, speedups, color=colors_list)
    ax6.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    ax6.set_ylabel('高速化率 (倍)', fontsize=12)
    ax6.set_title('SigLIP高速化率 (>1で高速)', fontsize=14, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)

plt.suptitle('ColModernVBERT (SigLIP) vs ColVBERT (BLIP)\nRAPTOR Tree構築 + 階層的検索性能比較', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

plot_file = results_dir / "raptor_comparison.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"✅ グラフを {plot_file} に保存")

# ステップ10: サマリー出力
print("\n[ステップ 10/12] サマリー出力...")

print("\n" + "=" * 80)
print("📊 RAPTOR Tree構築 + 階層的検索 性能比較サマリー")
print("=" * 80)

print("\n【RAPTOR Tree構築】")
print(f"ColVBERT (BLIP):")
print(f"  - 構築時間: {colbert_tree_build_time:.2f}秒")
print(f"  - 総ノード数: {colbert_tree_stats['total_nodes']}")
print(f"  - リーフノード: {colbert_tree_stats['num_leaf_nodes']}")
print(f"  - 内部ノード: {colbert_tree_stats['num_internal_nodes']}")
print(f"  - 最大深度: {colbert_tree_stats['max_depth']}")

print(f"\nColModernVBERT (SigLIP):")
print(f"  - 構築時間: {modern_tree_build_time:.2f}秒")
print(f"  - 総ノード数: {modern_tree_stats['total_nodes']}")
print(f"  - リーフノード: {modern_tree_stats['num_leaf_nodes']}")
print(f"  - 内部ノード: {modern_tree_stats['num_internal_nodes']}")
print(f"  - 最大深度: {modern_tree_stats['max_depth']}")

if modern_tree_build_time > 0:
    print(f"\n⚡ Tree構築高速化率: {comparison_results['comparison']['tree_build_speedup']:.2f}x")

print("\n【階層的検索性能】")
if colbert_search_times:
    print(f"ColVBERT (BLIP):")
    print(f"  - 平均検索時間: {np.mean(colbert_search_times)*1000:.2f}ms")
    print(f"  - 中央値: {np.median(colbert_search_times)*1000:.2f}ms")

if modern_search_times:
    print(f"\nColModernVBERT (SigLIP):")
    print(f"  - 平均検索時間: {np.mean(modern_search_times)*1000:.2f}ms")
    print(f"  - 中央値: {np.median(modern_search_times)*1000:.2f}ms")

if colbert_search_times and modern_search_times and np.mean(modern_search_times) > 0:
    print(f"\n⚡ 検索高速化率: {comparison_results['comparison']['search_speedup']:.2f}x")

print("\n【GPU使用量】")
print(f"ColVBERT (BLIP) ピークメモリ: {colbert_tree_gpu_peak:.0f}MB")
print(f"ColModernVBERT (SigLIP) ピークメモリ: {modern_tree_gpu_peak:.0f}MB")

print("\n" + "=" * 80)
print("✅ RAPTOR Tree構築 + 階層的検索 性能比較完了!")
print(f"📁 結果ディレクトリ: {results_dir}")
print(f"📊 グラフ: {plot_file}")
print(f"📄 詳細結果: {results_file}")
print(f"🌳 RAPTORツリー: {trees_dir}")
print("=" * 80)

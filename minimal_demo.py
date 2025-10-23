"""
Minimal Quick Start Demo
依存関係を最小限にしたクイックスタートデモ
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


def create_sample_data(output_dir: str = "data/minimal_demo") -> Dict[str, Any]:
    """サンプルデータを作成"""
    print("📊 Creating sample disaster document data...")
    
    # ディレクトリ作成
    data_path = Path(output_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # サンプル災害文書データ
    sample_documents = [
        {
            "doc_id": "evac_map_001",
            "title": "避難マップ - 中央地区",
            "content": "中央地区の避難所：中央小学校体育館（収容人数200名）、避難経路：国道1号線経由",
            "category": "evacuation_map",
            "metadata": {
                "area": "中央地区",
                "capacity": 200,
                "type": "evacuation"
            }
        },
        {
            "doc_id": "recovery_001",
            "title": "復旧計画書 - インフラ復旧",
            "content": "道路復旧：国道1号線の復旧完了（進捗90%）、水道復旧：配水管の修復作業中（進捗60%）",
            "category": "recovery_plan",
            "metadata": {
                "progress": {"road": 90, "water": 60},
                "type": "infrastructure"
            }
        },
        {
            "doc_id": "admin_notice_001",
            "title": "行政通知 - 避難指示",
            "content": "台風接近に伴い、低地地域に避難指示を発令。指定避難所：中央小学校、南部公民館",
            "category": "admin_notice",
            "metadata": {
                "urgency": "high",
                "affected_areas": ["低地地域"],
                "type": "evacuation_order"
            }
        },
        {
            "doc_id": "damage_report_001",
            "title": "被害報告書",
            "content": "人的被害：負傷者5名、建物被害：全壊3棟、半壊15棟、停電：1,200世帯",
            "category": "damage_report",
            "metadata": {
                "casualties": {"injured": 5},
                "buildings": {"destroyed": 3, "damaged": 15},
                "utilities": {"power_outage": 1200}
            }
        },
        {
            "doc_id": "support_info_001",
            "title": "支援物資配布情報",
            "content": "配布場所：中央小学校、配布時間：9:00-17:00、配布物資：水、食料、毛布",
            "category": "support_info",
            "metadata": {
                "location": "中央小学校",
                "hours": "9:00-17:00",
                "supplies": ["水", "食料", "毛布"]
            }
        }
    ]
    
    # サンプルクエリ
    sample_queries = [
        {
            "query_id": "q001",
            "text": "避難所の場所を教えてください",
            "category": "evacuation"
        },
        {
            "query_id": "q002", 
            "text": "復旧工事の進捗状況は？",
            "category": "recovery"
        },
        {
            "query_id": "q003",
            "text": "支援物資の配布について",
            "category": "support"
        }
    ]
    
    # データ保存
    documents_file = data_path / "sample_documents.json"
    with open(documents_file, 'w', encoding='utf-8') as f:
        json.dump(sample_documents, f, ensure_ascii=False, indent=2)
    
    queries_file = data_path / "sample_queries.json"
    with open(queries_file, 'w', encoding='utf-8') as f:
        json.dump(sample_queries, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ Created {len(sample_documents)} sample documents")
    print(f"   ✅ Created {len(sample_queries)} sample queries")
    print(f"   📁 Data saved to: {output_dir}")
    
    return {
        "documents": sample_documents,
        "queries": sample_queries,
        "output_dir": str(data_path)
    }


def simulate_search(documents: List[Dict], query: str) -> List[Dict]:
    """簡単な検索をシミュレート"""
    results = []
    query_lower = query.lower()
    
    for doc in documents:
        # 簡単なキーワードマッチング
        content_lower = doc["content"].lower()
        title_lower = doc["title"].lower()
        
        score = 0.0
        
        # タイトルでのマッチ
        for word in query_lower.split():
            if word in title_lower:
                score += 0.5
            if word in content_lower:
                score += 0.3
        
        # カテゴリマッチ
        if "避難" in query_lower and doc["category"] == "evacuation_map":
            score += 0.4
        elif "復旧" in query_lower and doc["category"] == "recovery_plan":
            score += 0.4
        elif "支援" in query_lower and doc["category"] == "support_info":
            score += 0.4
        elif "被害" in query_lower and doc["category"] == "damage_report":
            score += 0.4
        elif "通知" in query_lower and doc["category"] == "admin_notice":
            score += 0.4
        
        if score > 0:
            results.append({
                "document": doc,
                "score": score,
                "relevance": "high" if score > 0.7 else "medium" if score > 0.4 else "low"
            })
    
    # スコア順でソート
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:3]  # 上位3件


def run_search_demo(data: Dict[str, Any]) -> Dict[str, Any]:
    """検索デモを実行"""
    print("\n🔍 Running search demonstration...")
    
    documents = data["documents"]
    queries = data["queries"]
    
    search_results = {}
    
    for query_info in queries:
        query_id = query_info["query_id"]
        query_text = query_info["text"]
        
        print(f"\n   Query: {query_text}")
        
        # 検索実行
        results = simulate_search(documents, query_text)
        
        # 結果表示
        for i, result in enumerate(results, 1):
            doc = result["document"]
            score = result["score"]
            print(f"   [{i}] {doc['title']} (Score: {score:.2f})")
            print(f"       {doc['content'][:100]}...")
        
        search_results[query_id] = results
    
    return search_results


def calculate_demo_metrics(search_results: Dict[str, Any]) -> Dict[str, Any]:
    """デモ用のメトリクスを計算"""
    total_queries = len(search_results)
    total_results = sum(len(results) for results in search_results.values())
    avg_results_per_query = total_results / total_queries if total_queries > 0 else 0
    
    # 簡単な品質評価
    high_quality_results = 0
    for results in search_results.values():
        for result in results:
            if result["score"] > 0.7:
                high_quality_results += 1
    
    quality_ratio = high_quality_results / total_results if total_results > 0 else 0
    
    metrics = {
        "total_queries": total_queries,
        "total_results": total_results,
        "avg_results_per_query": avg_results_per_query,
        "high_quality_results": high_quality_results,
        "quality_ratio": quality_ratio,
        "demo_precision": quality_ratio  # 簡易的な精度指標
    }
    
    return metrics


def save_demo_results(data: Dict[str, Any], search_results: Dict[str, Any], metrics: Dict[str, Any]):
    """デモ結果を保存"""
    output_dir = Path(data["output_dir"])
    
    demo_summary = {
        "demo_info": {
            "title": "Visual RAPTOR ColBERT - Minimal Demo",
            "description": "災害文書検索システムの基本機能デモ",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0-minimal"
        },
        "data_summary": {
            "num_documents": len(data["documents"]),
            "num_queries": len(data["queries"]),
            "document_categories": list(set(doc["category"] for doc in data["documents"]))
        },
        "search_results": search_results,
        "metrics": metrics
    }
    
    results_file = output_dir / "demo_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(demo_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Demo results saved to: {results_file}")


def main():
    """メイン実行関数"""
    print("="*80)
    print("🚀 Visual RAPTOR ColBERT - Minimal Quick Start Demo")
    print("災害文書検索システム - 最小構成クイックスタート")
    print("="*80)
    
    print("\n📝 This is a minimal demonstration showing:")
    print("   - Basic disaster document data structure")
    print("   - Simple keyword-based search simulation")
    print("   - Performance metrics calculation")
    print("   - System component overview")
    
    try:
        # 1. サンプルデータ作成
        print("\n" + "="*60)
        print("Step 1: Creating Sample Data")
        print("="*60)
        data = create_sample_data()
        
        # 2. 検索デモ実行
        print("\n" + "="*60)
        print("Step 2: Search Demonstration")
        print("="*60)
        search_results = run_search_demo(data)
        
        # 3. メトリクス計算
        print("\n" + "="*60)
        print("Step 3: Performance Metrics")
        print("="*60)
        metrics = calculate_demo_metrics(search_results)
        
        print(f"   Total queries processed: {metrics['total_queries']}")
        print(f"   Average results per query: {metrics['avg_results_per_query']:.1f}")
        print(f"   High quality results: {metrics['high_quality_results']}")
        print(f"   Demo precision: {metrics['demo_precision']:.3f}")
        
        # 4. 結果保存
        save_demo_results(data, search_results, metrics)
        
        # 5. 次のステップ案内
        print("\n" + "="*80)
        print("✅ Minimal Demo Completed Successfully!")
        print("="*80)
        
        print("\n🎯 What this demo showed:")
        print("   ✓ Disaster document data structure")
        print("   ✓ Basic search functionality")
        print("   ✓ Performance evaluation")
        print("   ✓ Results storage and analysis")
        
        print("\n🚀 Next steps for full system:")
        print("   1. Install full dependencies: pip install -r requirements.txt")
        print("   2. Set up Ollama: ollama pull mxbai-embed-large")
        print("   3. Run full demo: python quick_start_demo.py")
        print("   4. Explore visual document processing")
        print("   5. Try JinaVDR benchmarking")
        
        print(f"\n📁 Generated files location: {data['output_dir']}")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("This minimal demo should work without external dependencies.")


if __name__ == "__main__":
    main()
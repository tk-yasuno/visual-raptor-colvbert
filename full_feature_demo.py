#!/usr/bin/env python3
"""
Visual RAPTOR ColBERT Full Feature Demo
完全版のVisual RAPTOR ColBERTシステムのデモンストレーション
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# LangChain imports
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.schema import Document

# Local imports
from visual_raptor_colbert import VisualRAPTORColBERT, VisualDocument
from jina_vdr_benchmark import JinaVDRBenchmark, JinaBenchmarkDocument, JinaBenchmarkQuery
from enhanced_visual_processing import EnhancedVisualProcessor


class FullFeatureDemo:
    """フル機能デモクラス"""
    
    def __init__(self):
        """デモシステムを初期化"""
        self.embeddings = None
        self.llm = None
        self.visual_raptor = None
        self.benchmark = None
        self.visual_processor = None
        
        # データディレクトリ
        self.data_dir = Path("data/full_feature_demo")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize_models(self) -> bool:
        """モデルを初期化"""
        try:
            print("🔧 Initializing Ollama models...")
            
            # 埋め込みモデル
            self.embeddings = OllamaEmbeddings(
                model="mxbai-embed-large",
                base_url="http://localhost:11434"
            )
            
            # LLMモデル
            self.llm = ChatOllama(
                model="granite-code:8b",
                temperature=0.1,
                base_url="http://localhost:11434",
                timeout=300  # 5分のタイムアウト
            )
            
            # モデルテスト
            print("   Testing embedding model...")
            test_embedding = self.embeddings.embed_query("test query")
            print(f"   Embedding dimension: {len(test_embedding)}")
            
            print("   Testing LLM model...")
            test_response = self.llm.invoke("Hello")
            print(f"   LLM response: {test_response.content[:50]}...")
            
            print("✅ Models initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize models: {e}")
            return False
    
    def initialize_system(self) -> bool:
        """システムコンポーネントを初期化"""
        try:
            print("🚀 Initializing system components...")
            
            # Visual RAPTOR初期化
            self.visual_raptor = VisualRAPTORColBERT(
                embeddings_model=self.embeddings,
                llm=self.llm,
                base_dir=str(self.data_dir / "raptor")
            )
            
            # ベンチマーク初期化
            self.benchmark = JinaVDRBenchmark(
                base_dir=str(self.data_dir / "benchmark")
            )
            
            # ビジュアルプロセッサ初期化（簡易版）
            self.visual_processor = EnhancedVisualProcessor()
            
            print("✅ System components initialized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize system: {e}")
            return False
    
    def create_sample_documents(self, num_docs: int = 10) -> List[JinaBenchmarkDocument]:
        """サンプル災害文書を作成"""
        print(f"📄 Creating {num_docs} sample disaster documents...")
        
        documents = []
        
        # 災害カテゴリと内容
        disaster_categories = [
            ("evacuation_map", "避難マップ", [
                "避難所の場所", "避難経路", "危険区域", "安全地帯", "集合場所"
            ]),
            ("emergency_manual", "緊急時マニュアル", [
                "緊急連絡先", "避難手順", "応急処置", "備蓄品リスト", "安全確認"
            ]),
            ("recovery_plan", "復旧計画書", [
                "復旧手順", "資源配分", "優先順位", "復旧スケジュール", "支援体制"
            ]),
            ("damage_report", "被害報告書", [
                "被害状況", "損失評価", "復旧見積もり", "影響範囲", "対応状況"
            ])
        ]
        
        for i in range(num_docs):
            # カテゴリをローテーション
            category_info = disaster_categories[i % len(disaster_categories)]
            category, category_jp, keywords = category_info
            
            # サンプル画像作成
            img_path = self.data_dir / "images" / f"document_{i+1:03d}.png"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 簡単な画像を作成
            img = Image.new('RGB', (800, 600), 'white')
            draw = ImageDraw.Draw(img)
            
            # タイトル
            title = f"{category_jp} #{i+1:03d}"
            draw.text((50, 50), title, fill='black')
            
            # キーワードをランダムに配置
            selected_keywords = np.random.choice(keywords, size=3, replace=False)
            for j, keyword in enumerate(selected_keywords):
                draw.text((50, 100 + j*50), f"• {keyword}", fill='blue')
            
            img.save(img_path)
            
            # 文書オブジェクト作成
            doc = JinaBenchmarkDocument(
                doc_id=f"doc_{i+1:03d}",
                category=category,
                subcategory=f"{category}_{i%3+1}",
                image_path=str(img_path),
                text_content=f"{title}\n" + "\n".join([f"• {kw}" for kw in selected_keywords]),
                metadata={
                    'title': title,
                    'keywords': list(selected_keywords),
                    'doc_number': i+1
                }
            )
            
            documents.append(doc)
        
        print(f"✅ Created {len(documents)} sample documents")
        return documents
    
    def create_sample_queries(self, num_queries: int = 5) -> List[JinaBenchmarkQuery]:
        """サンプルクエリを作成"""
        print(f"❓ Creating {num_queries} sample queries...")
        
        query_templates = [
            ("避難所への経路を教えて", "evacuation_route"),
            ("緊急時の連絡先は？", "emergency_contact"),
            ("復旧の優先順位は？", "recovery_priority"),
            ("被害状況の詳細は？", "damage_details"),
            ("安全確認の手順は？", "safety_check"),
            ("備蓄品のリストは？", "supply_list"),
            ("避難の手順を説明して", "evacuation_procedure"),
            ("復旧計画の内容は？", "recovery_plan_details")
        ]
        
        queries = []
        for i in range(num_queries):
            template = query_templates[i % len(query_templates)]
            query_text, query_type = template
            
            query = JinaBenchmarkQuery(
                query_id=f"query_{i+1:03d}",
                text=query_text,
                category=query_type,
                metadata={
                    'query_number': i+1,
                    'expected_doc_type': 'disaster_document'
                }
            )
            
            queries.append(query)
        
        print(f"✅ Created {len(queries)} sample queries")
        return queries
    
    def run_visual_search_demo(self) -> Dict[str, Any]:
        """ビジュアル検索デモを実行"""
        print("🔍 Running visual search demonstration...")
        
        # サンプルデータ作成
        documents = self.create_sample_documents(15)
        queries = self.create_sample_queries(8)
        
        # Visual Documentに変換
        visual_docs = []
        for doc in documents:
            visual_doc = VisualDocument(
                image_path=doc.image_path,
                text_content=doc.text_content,
                layout_elements=[],
                metadata=doc.metadata
            )
            visual_docs.append(visual_doc)
        
        # Visual RAPTORに文書追加
        self.visual_raptor.visual_documents = visual_docs
        
        # インデックス構築
        print("   Building visual index...")
        start_time = time.time()
        visual_index = self.visual_raptor.build_visual_index()
        build_time = time.time() - start_time
        
        print(f"   Index built in {build_time:.2f}s")
        
        # 検索実行
        search_results = {}
        search_times = []
        
        for query in queries:
            print(f"   Searching: {query.text}")
            
            start_time = time.time()
            results = self.visual_raptor.search_visual_documents(
                query.text,
                top_k=5
            )
            search_time = time.time() - start_time
            search_times.append(search_time)
            
            # 結果を整理
            formatted_results = []
            for visual_doc, score in results:
                result_info = {
                    'image_path': visual_doc.image_path,
                    'text_content': visual_doc.text_content[:100] + "...",
                    'score': float(score),
                    'metadata': visual_doc.metadata
                }
                formatted_results.append(result_info)
            
            search_results[query.query_id] = {
                'query': query.text,
                'results': formatted_results,
                'search_time': search_time
            }
        
        demo_stats = {
            'total_documents': len(documents),
            'total_queries': len(queries),
            'avg_search_time': np.mean(search_times),
            'total_search_time': sum(search_times),
            'index_build_time': build_time
        }
        
        print(f"✅ Visual search demo completed")
        print(f"   Documents: {demo_stats['total_documents']}")
        print(f"   Queries: {demo_stats['total_queries']}")
        print(f"   Avg search time: {demo_stats['avg_search_time']:.3f}s")
        
        return {
            'search_results': search_results,
            'statistics': demo_stats
        }
    
    def save_demo_results(self, results: Dict[str, Any]):
        """デモ結果を保存"""
        import json
        
        output_file = self.data_dir / "full_demo_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Demo results saved to {output_file}")
    
    def run_complete_demo(self) -> bool:
        """完全なデモを実行"""
        print("="*80)
        print("🚀 Visual RAPTOR ColBERT Full Feature Demo")
        print("災害文書検索システム 完全版デモンストレーション")
        print("="*80)
        
        try:
            # 1. モデル初期化
            if not self.initialize_models():
                return False
            
            # 2. システム初期化
            if not self.initialize_system():
                return False
            
            # 3. ビジュアル検索デモ
            demo_results = self.run_visual_search_demo()
            
            # 4. 結果保存
            self.save_demo_results(demo_results)
            
            # 5. サマリー表示
            print("\n" + "="*80)
            print("📊 Demo Summary")
            print("="*80)
            
            stats = demo_results['statistics']
            print(f"Documents processed: {stats['total_documents']}")
            print(f"Queries executed: {stats['total_queries']}")
            print(f"Index build time: {stats['index_build_time']:.2f}s")
            print(f"Average search time: {stats['avg_search_time']:.3f}s")
            print(f"Total search time: {stats['total_search_time']:.2f}s")
            
            print("\n🎉 Full feature demo completed successfully!")
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """メイン実行関数"""
    demo = FullFeatureDemo()
    success = demo.run_complete_demo()
    
    if success:
        print("\n✅ All demonstrations completed successfully!")
        print("Next steps:")
        print("1. Check 'data/full_feature_demo/' for generated files")
        print("2. Review 'full_demo_results.json' for detailed results")
        print("3. Try running with different parameters in the code")
    else:
        print("\n❌ Demo failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
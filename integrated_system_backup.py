"""
Visual RAPTOR ColBERT 統合システム

RAPTOR + ColVBERT + Visual Document処理の統合システム
- JinaVDRベンチマーク対応
- 震災関連文書検索
- 多言語・マルチモーダル検索
- 性能評価機能

Version: 1.0 - Complete Integration
"""

import os
import sys
import json
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import pandas as pd
from datetime import datetime

# 各コンポーネントをインポート
from visual_raptor_colbert import VisualRAPTORColBERT, VisualDocument
from jina_vdr_benchmark import JinaVDRBenchmark, VDRQuery, VDRDocument
try:
    from enhanced_visual_processing import EnhancedVisualProcessor, ProcessingResult
except:
    EnhancedVisualProcessor = None
    ProcessingResult = None

try:
    from disaster_dataset_generator import DisasterDocumentGenerator
except:
    DisasterDocumentGenerator = None

# LangChain関連
from langchain_core.documents import Document


class DisasterVDREvaluator:
    """災害VDR評価器"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_precision_recall(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str]
    ) -> Tuple[float, float, float]:
        """Precision, Recall, F1を計算"""
        if not retrieved_docs:
            return 0.0, 0.0, 0.0
        
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)
        
        true_positives = len(retrieved_set & relevant_set)
        
        precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
        recall = true_positives / len(relevant_set) if relevant_set else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def calculate_ndcg(
        self,
        retrieved_docs: List[Tuple[str, float]],
        relevance_judgments: Dict[str, int],
        k: int = 10
    ) -> float:
        """NDCG@kを計算"""
        if not retrieved_docs:
            return 0.0
        
        # DCG計算
        dcg = 0.0
        for i, (doc_id, score) in enumerate(retrieved_docs[:k]):
            rel = relevance_judgments.get(doc_id, 0)
            dcg += rel / np.log2(i + 2)
        
        # IDCG計算
        ideal_rels = sorted(relevance_judgments.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_system(
        self,
        system_results: Dict[str, List[Tuple[str, float]]],
        benchmark: JinaVDRBenchmark
    ) -> Dict[str, float]:
        """システム全体を評価"""
        all_precision = []
        all_recall = []
        all_f1 = []
        all_ndcg = []
        
        # 関連性判定を辞書に変換
        relevance_dict = {}
        for judgment in benchmark.relevance_judgments:
            if judgment.query_id not in relevance_dict:
                relevance_dict[judgment.query_id] = {}
            relevance_dict[judgment.query_id][judgment.doc_id] = judgment.relevance
        
        # 各クエリについて評価
        for query_id, results in system_results.items():
            if query_id in relevance_dict:
                # 関連文書IDのリスト
                relevant_docs = [
                    doc_id for doc_id, rel in relevance_dict[query_id].items()
                    if rel > 0
                ]
                
                # 検索結果のIDのリスト
                retrieved_docs = [doc_id for doc_id, score in results]
                
                # Precision, Recall, F1
                p, r, f1 = self.calculate_precision_recall(retrieved_docs, relevant_docs)
                all_precision.append(p)
                all_recall.append(r)
                all_f1.append(f1)
                
                # NDCG
                ndcg = self.calculate_ndcg(results, relevance_dict[query_id])
                all_ndcg.append(ndcg)
        
        # 平均値計算
        metrics = {
            'precision': np.mean(all_precision) if all_precision else 0.0,
            'recall': np.mean(all_recall) if all_recall else 0.0,
            'f1': np.mean(all_f1) if all_f1 else 0.0,
            'ndcg': np.mean(all_ndcg) if all_ndcg else 0.0,
            'num_queries': len(system_results)
        }
        
        return metrics


class IntegratedVisualRAPTORSystem:
    """
    統合Visual RAPTORシステム
    
    機能:
    - ベンチマークデータ生成
    - 文書処理・インデックス構築
    - マルチモーダル検索
    - 性能評価
    """
    
    def __init__(
        self,
        embeddings_model,
        llm,
        system_config: Dict[str, Any] = None
    ):
        self.embeddings_model = embeddings_model
        self.llm = llm
        
        # システム設定
        self.config = system_config or self._get_default_config()
        
        # コンポーネント初期化
        self.visual_raptor = None
        self.benchmark = None
        self.visual_processor = None
        self.document_generator = None
        self.evaluator = DisasterVDREvaluator()
        
        # 状態管理
        self.is_initialized = False
        self.processing_stats = {}
        
        print(f"🚀 Integrated Visual RAPTOR System initialized")
        print(f"   Configuration: {list(self.config.keys())}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を取得"""
        return {
            'data_dir': 'data/integrated_system',
            'benchmark_size': 'small',  # small: 500, medium: 1000, large: 2000+
            'num_queries': 50,
            'num_documents': 500,
            'colbert_config': {
                'text_model': 'intfloat/multilingual-e5-large',
                'vision_model': 'Salesforce/blip2-opt-2.7b',
                'embedding_dim': 768
            },
            'visual_config': {
                'ocr_engines': ['tesseract', 'easyocr'],
                'languages': ['ja', 'en'],
                'confidence_threshold': 0.5
            },
            'raptor_config': {
                'min_clusters': 2,
                'max_clusters': 5,
                'max_depth': 3,
                'chunk_size': 500,
                'chunk_overlap': 100,
                'selection_strategy': 'silhouette'
            }
        }
    
    def initialize_system(self):
        """システムを初期化"""
        print("🔧 Initializing system components...")
        
        # データディレクトリ作成
        data_dir = Path(self.config['data_dir'])
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Visual RAPTOR初期化
        print("   Initializing Visual RAPTOR...")
        self.visual_raptor = VisualRAPTORColBERT(
            embeddings_model=self.embeddings_model,
            llm=self.llm,
            colbert_config=self.config['colbert_config'],
            visual_config=self.config['visual_config'],
            **self.config['raptor_config']
        )
        
        # ベンチマーク初期化
        print("   Initializing JinaVDR benchmark...")
        self.benchmark = JinaVDRBenchmark(
            data_dir=str(data_dir / 'jina_vdr'),
            language='ja',
            dataset_size=self.config['benchmark_size']
        )
        
        # ビジュアルプロセッサ初期化
        print("   Initializing visual processor...")
        self.visual_processor = EnhancedVisualProcessor(
            ocr_engines=self.config['visual_config']['ocr_engines'],
            languages=self.config['visual_config']['languages'],
            confidence_threshold=self.config['visual_config']['confidence_threshold']
        )
        
        # 文書生成器初期化
        print("   Initializing document generator...")
        self.document_generator = DisasterDocumentGenerator(
            output_dir=str(data_dir / 'disaster_documents')
        )
        
        self.is_initialized = True
        print("✅ System initialization completed")
    
    def setup_benchmark_data(self) -> Dict[str, Any]:
        """ベンチマークデータをセットアップ"""
        if not self.is_initialized:
            self.initialize_system()
        
        print("📊 Setting up benchmark data...")
        
        # クエリ生成
        print(f"   Generating {self.config['num_queries']} queries...")
        queries = self.benchmark.generate_disaster_queries(self.config['num_queries'])
        
        # 災害文書生成
        print(f"   Generating {self.config['num_documents']} disaster documents...")
        disaster_docs = self.document_generator.generate_dataset(self.config['num_documents'])
        
        # ベンチマーク文書データを作成
        benchmark_docs = []
        for doc_info in disaster_docs:
            vdr_doc = VDRDocument(
                doc_id=doc_info['doc_id'],
                image_path=doc_info['image_path'],
                text_content="",  # OCRで後で抽出
                category=doc_info['metadata']['document_type'],
                subcategory=doc_info['metadata'].get('area_name', ''),
                metadata=doc_info['metadata']
            )
            benchmark_docs.append(vdr_doc)
        
        self.benchmark.documents = benchmark_docs
        
        # 関連性判定生成
        print("   Generating relevance judgments...")
        self.benchmark.generate_relevance_judgments()
        
        # データ保存
        self.benchmark.save_benchmark_data()
        
        setup_stats = {
            'num_queries': len(queries),
            'num_documents': len(benchmark_docs),
            'num_judgments': len(self.benchmark.relevance_judgments),
            'setup_time': time.time()
        }
        
        print(f"✅ Benchmark data setup completed")
        print(f"   Queries: {setup_stats['num_queries']}")
        print(f"   Documents: {setup_stats['num_documents']}")
        print(f"   Judgments: {setup_stats['num_judgments']}")
        
        return setup_stats
    
    def process_visual_documents(self) -> Dict[str, Any]:
        """ビジュアル文書を処理"""
        if not self.benchmark or not self.benchmark.documents:
            raise ValueError("ベンチマークデータが設定されていません")
        
        print("🖼️ Processing visual documents...")
        
        processed_docs = []
        processing_times = []
        
        for i, doc in enumerate(self.benchmark.documents):
            try:
                start_time = time.time()
                
                # OCRでテキスト抽出
                result = self.visual_processor.process_document(doc.image_path)
                
                # テキストコンテンツを更新
                doc.text_content = result.text_content
                doc.metadata.update({
                    'processing_confidence': result.confidence_scores['overall_confidence'],
                    'layout_elements': len(result.layout_elements),
                    'tables_detected': len(result.tables)
                })
                
                processed_docs.append(doc)
                processing_times.append(time.time() - start_time)
                
                if (i + 1) % 50 == 0:
                    print(f"   Processed {i + 1}/{len(self.benchmark.documents)} documents")
                    
            except Exception as e:
                print(f"   Error processing {doc.doc_id}: {e}")
        
        self.benchmark.documents = processed_docs
        
        processing_stats = {
            'total_processed': len(processed_docs),
            'avg_processing_time': np.mean(processing_times),
            'total_processing_time': sum(processing_times)
        }
        
        self.processing_stats = processing_stats
        
        print(f"✅ Visual document processing completed")
        print(f"   Processed: {processing_stats['total_processed']} documents")
        print(f"   Avg time: {processing_stats['avg_processing_time']:.2f}s per doc")
        
        return processing_stats
    
    def build_integrated_index(self) -> Dict[str, Any]:
        """統合インデックスを構築"""
        if not self.benchmark or not self.benchmark.documents:
            raise ValueError("文書が処理されていません")
        
        print("🔍 Building integrated index...")
        
        start_time = time.time()
        
        # ビジュアル文書をVisualDocumentに変換
        visual_docs = []
        for doc in self.benchmark.documents:
            visual_doc = VisualDocument(
                image_path=doc.image_path,
                text_content=doc.text_content,
                layout_elements=[],
                metadata=doc.metadata
            )
            visual_docs.append(visual_doc)
        
        # Visual RAPTOR に設定
        self.visual_raptor.visual_documents = visual_docs
        
        # ビジュアルインデックス構築
        visual_index = self.visual_raptor.build_visual_index()
        
        # テキスト文書準備
        text_docs = []
        for doc in self.benchmark.documents:
            langchain_doc = Document(
                page_content=doc.text_content,
                metadata={
                    'doc_id': doc.doc_id,
                    'category': doc.category,
                    'subcategory': doc.subcategory,
                    'image_path': doc.image_path
                }
            )
            text_docs.append(langchain_doc)
        
        # 統合ツリー構築
        tree = self.visual_raptor.build_hybrid_tree(
            text_documents=text_docs,
            visual_documents=visual_docs,
            save_dir=str(Path(self.config['data_dir']) / 'saved_models' / 'integrated')
        )
        
        build_time = time.time() - start_time
        
        index_stats = {
            'visual_documents': len(visual_docs),
            'text_documents': len(text_docs),
            'build_time': build_time,
            'tree_nodes': self.visual_raptor._count_nodes(tree),
            'tree_depth': self.visual_raptor._get_tree_depth(tree)
        }
        
        print(f"✅ Integrated index built successfully")
        print(f"   Visual docs: {index_stats['visual_documents']}")
        print(f"   Text docs: {index_stats['text_documents']}")
        print(f"   Tree nodes: {index_stats['tree_nodes']}")
        print(f"   Tree depth: {index_stats['tree_depth']}")
        print(f"   Build time: {build_time:.2f}s")
        
        return index_stats
    
    def run_evaluation(self, top_k: int = 10) -> Dict[str, Any]:
        """システム評価を実行"""
        if not self.visual_raptor or not self.benchmark:
            raise ValueError("システムが完全に初期化されていません")
        
        print(f"📈 Running system evaluation (top_k={top_k})...")
        
        # 各クエリで検索実行
        system_results = {}
        search_times = []
        
        for query in self.benchmark.queries:
            try:
                start_time = time.time()
                
                # ビジュアル検索実行
                visual_results = self.visual_raptor.search_visual_documents(
                    query.text,
                    top_k=top_k
                )
                
                # 結果を(doc_id, score)のタプルに変換
                results = []
                for visual_doc, score in visual_results:
                    # visual_docのmetadataからdoc_idを取得
                    doc_id = None
                    for doc in self.benchmark.documents:
                        if doc.image_path == visual_doc.image_path:
                            doc_id = doc.doc_id
                            break
                    
                    if doc_id:
                        results.append((doc_id, float(score)))
                
                system_results[query.query_id] = results
                search_times.append(time.time() - start_time)
                
            except Exception as e:
                print(f"   Error processing query {query.query_id}: {e}")
                system_results[query.query_id] = []
        
        # 評価メトリクス計算
        metrics = self.evaluator.evaluate_system(system_results, self.benchmark)
        
        # 追加統計
        metrics.update({
            'avg_search_time': np.mean(search_times) if search_times else 0.0,
            'total_search_time': sum(search_times),
            'queries_processed': len(system_results)
        })
        
        print(f"✅ Evaluation completed")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1: {metrics['f1']:.4f}")
        print(f"   NDCG: {metrics['ndcg']:.4f}")
        print(f"   Avg search time: {metrics['avg_search_time']:.3f}s")
        
        return metrics
    
    def save_results(self, results: Dict[str, Any]):
        """結果を保存"""
        results_dir = Path(self.config['data_dir']) / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 結果をJSONで保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f'evaluation_results_{timestamp}.json'
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"💾 Results saved to {results_file}")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """完全なパイプラインを実行"""
        print("="*80)
        print("🚀 Running Complete Visual RAPTOR Pipeline")
        print("="*80)
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'config': self.config
        }
        
        try:
            # 1. システム初期化
            if not self.is_initialized:
                self.initialize_system()
            
            # 2. ベンチマークデータセットアップ
            setup_stats = self.setup_benchmark_data()
            pipeline_results['setup_stats'] = setup_stats
            
            # 3. ビジュアル文書処理
            processing_stats = self.process_visual_documents()
            pipeline_results['processing_stats'] = processing_stats
            
            # 4. 統合インデックス構築
            index_stats = self.build_integrated_index()
            pipeline_results['index_stats'] = index_stats
            
            # 5. システム評価
            evaluation_metrics = self.run_evaluation()
            pipeline_results['evaluation_metrics'] = evaluation_metrics
            
            # 6. 結果保存
            pipeline_results['end_time'] = datetime.now().isoformat()
            self.save_results(pipeline_results)
            
            print("\
" + "="*80)
            print("✅ Complete pipeline execution finished successfully!")
            print("="*80)
            
            return pipeline_results
            
        except Exception as e:
            import traceback
            pipeline_results['error'] = str(e)
            pipeline_results['traceback'] = traceback.format_exc()
            pipeline_results['end_time'] = datetime.now().isoformat()
            
            print(f"\n❌ Pipeline execution failed: {e}")
            print("\nDetailed traceback:")
            traceback.print_exc()
            return pipeline_results


def create_integrated_system(
    embeddings_model,
    llm,
    config: Dict[str, Any] = None
) -> IntegratedVisualRAPTORSystem:
    """統合システムを作成"""
    system = IntegratedVisualRAPTORSystem(
        embeddings_model=embeddings_model,
        llm=llm,
        system_config=config
    )
    return system


def main():
    """メイン実行関数"""
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    
    print("="*80)
    print("Visual RAPTOR ColBERT Integration System")
    print("災害文書検索・教訓継承統合システム")
    print("="*80)
    
    # モデル初期化
    print("\
🔧 Initializing models...")
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url="http://localhost:11434"
    )
    llm = ChatOllama(
        model="granite-code:8b",
        temperature=0,
        base_url="http://localhost:11434",
        timeout=600  # 10分のタイムアウト
    )
    
    # 統合システム作成
    print("\
🚀 Creating integrated system...")
    config = {
        'data_dir': 'data/integrated_visual_raptor',
        'benchmark_size': 'small',
        'num_queries': 20,  # デモ用に少なめ
        'num_documents': 50  # デモ用に少なめ
    }
    
    system = create_integrated_system(
        embeddings_model=embeddings,
        llm=llm,
        config=config
    )
    
    # 完全なパイプライン実行
    results = system.run_complete_pipeline()
    
    print(f"\
📊 Final Results Summary:")
    if 'evaluation_metrics' in results:
        metrics = results['evaluation_metrics']
        print(f"   Precision: {metrics.get('precision', 0):.4f}")
        print(f"   Recall: {metrics.get('recall', 0):.4f}")
        print(f"   F1 Score: {metrics.get('f1', 0):.4f}")
        print(f"   NDCG: {metrics.get('ndcg', 0):.4f}")
    
    print("\
🎉 System demonstration completed!")


if __name__ == "__main__":
    main()
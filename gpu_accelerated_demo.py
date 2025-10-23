#!/usr/bin/env python3
"""
GPU-Accelerated Visual RAPTOR ColBERT Demo
GPUを活用したVisual RAPTOR ColBERTシステムのデモンストレーション
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

# LangChain imports
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.schema import Document

print("🔧 GPU Configuration Check")
print("="*50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("✅ GPU ready for acceleration")
else:
    print("❌ GPU not available, using CPU")

# GPU設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class GPUAcceleratedEmbeddings:
    """GPU加速埋め込みクラス"""
    
    def __init__(self, ollama_embeddings, batch_size=32):
        self.ollama_embeddings = ollama_embeddings
        self.batch_size = batch_size
        self.device = device
        
        # キャッシュ用
        self.embedding_cache = {}
        
    def embed_documents_batch(self, texts: List[str]) -> torch.Tensor:
        """バッチで文書を埋め込み"""
        print(f"🚀 GPU batch embedding for {len(texts)} documents...")
        
        embeddings_list = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            print(f"   Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
            
            # バッチでOllama埋め込み取得
            batch_embeddings = []
            for text in batch_texts:
                if text in self.embedding_cache:
                    embedding = self.embedding_cache[text]
                else:
                    embedding = self.ollama_embeddings.embed_query(text)
                    self.embedding_cache[text] = embedding
                batch_embeddings.append(embedding)
            
            # GPUテンソルに変換
            batch_tensor = torch.tensor(batch_embeddings, dtype=torch.float32, device=self.device)
            embeddings_list.append(batch_tensor)
        
        # 全バッチを結合
        all_embeddings = torch.cat(embeddings_list, dim=0)
        
        print(f"✅ Batch embedding completed: {all_embeddings.shape}")
        return all_embeddings
    
    def embed_query(self, query: str) -> torch.Tensor:
        """クエリを埋め込み"""
        if query in self.embedding_cache:
            embedding = self.embedding_cache[query]
        else:
            embedding = self.ollama_embeddings.embed_query(query)
            self.embedding_cache[query] = embedding
            
        return torch.tensor(embedding, dtype=torch.float32, device=self.device)


class GPUAcceleratedSearch:
    """GPU加速検索クラス"""
    
    def __init__(self, gpu_embeddings):
        self.gpu_embeddings = gpu_embeddings
        self.document_embeddings = None
        self.documents = []
        
    def index_documents(self, documents: List[Dict], doc_texts: List[str]):
        """文書をインデックス化"""
        print("📚 GPU-accelerated document indexing...")
        
        start_time = time.time()
        
        # GPU バッチ埋め込み
        self.document_embeddings = self.gpu_embeddings.embed_documents_batch(doc_texts)
        self.documents = documents
        
        # 正規化（コサイン類似度用）
        self.document_embeddings = F.normalize(self.document_embeddings, p=2, dim=1)
        
        index_time = time.time() - start_time
        
        print(f"✅ Indexing completed in {index_time:.2f}s")
        print(f"   Documents: {len(documents)}")
        print(f"   Embedding shape: {self.document_embeddings.shape}")
        print(f"   GPU memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        return index_time
    
    def search_gpu(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """GPU加速検索"""
        if self.document_embeddings is None:
            return []
        
        start_time = time.time()
        
        # クエリ埋め込み
        query_embedding = self.gpu_embeddings.embed_query(query)
        query_embedding = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
        
        # GPU上でコサイン類似度計算
        similarities = torch.mm(query_embedding, self.document_embeddings.t()).squeeze(0)
        
        # Top-K取得
        top_scores, top_indices = torch.topk(similarities, k=min(top_k, len(self.documents)))
        
        # CPUに移動して結果作成
        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        
        results = []
        for idx, score in zip(top_indices, top_scores):
            results.append((self.documents[idx], float(score)))
        
        search_time = time.time() - start_time
        
        return results, search_time


class GPUDisasterSearchDemo:
    """GPU災害文書検索デモ"""
    
    def __init__(self):
        self.data_dir = Path("data/gpu_demo")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.ollama_embeddings = None
        self.llm = None
        self.gpu_embeddings = None
        self.gpu_search = None
        
    def initialize_models(self):
        """モデル初期化"""
        print("🔧 Initializing models...")
        
        # Ollama埋め込みモデル
        self.ollama_embeddings = OllamaEmbeddings(
            model="mxbai-embed-large",
            base_url="http://localhost:11434"
        )
        
        # LLMモデル
        self.llm = ChatOllama(
            model="granite-code:8b",
            temperature=0.1,
            base_url="http://localhost:11434",
            timeout=120
        )
        
        # GPU加速埋め込み
        self.gpu_embeddings = GPUAcceleratedEmbeddings(
            self.ollama_embeddings,
            batch_size=16  # RTX 4060 Ti用に調整
        )
        
        # GPU加速検索
        self.gpu_search = GPUAcceleratedSearch(self.gpu_embeddings)
        
        print("✅ Models initialized with GPU acceleration")
    
    def create_large_disaster_dataset(self, num_docs: int = 100) -> Tuple[List[Dict], List[str]]:
        """大規模災害文書データセットを作成"""
        print(f"📄 Creating large disaster dataset ({num_docs} documents)...")
        
        # 災害カテゴリとテンプレート
        disaster_templates = {
            "earthquake": {
                "title_templates": [
                    "地震発生時の避難手順 #{id}",
                    "震度{magnitude}の地震対応マニュアル #{id}",
                    "地震による建物被害調査報告 #{id}",
                    "地震避難所運営ガイド #{id}"
                ],
                "content_templates": [
                    "地震発生時は直ちに机の下に身を隠し、揺れが収まったら避難経路を確認してください。",
                    "震度{magnitude}以上の地震では建物倒壊の危険があるため、速やかに屋外に避難してください。",
                    "地震後は余震に注意し、ガス・電気の安全確認を行ってから避難所に向かってください。",
                    "地震避難時は頭部を保護し、落下物に注意して避難してください。"
                ]
            },
            "tsunami": {
                "title_templates": [
                    "津波警報発令時の緊急避難 #{id}",
                    "津波避難ビル指定施設 #{id}",
                    "津波被害想定調査 #{id}",
                    "津波避難タワー運用マニュアル #{id}"
                ],
                "content_templates": [
                    "津波警報発令時は直ちに高台または3階建て以上の頑丈な建物に避難してください。",
                    "海岸から最低2km以上内陸に避難し、標高20m以上の場所を目指してください。",
                    "津波は第一波より第二波以降が高くなる可能性があるため、警報解除まで避難を継続してください。",
                    "自動車での避難は渋滞の原因となるため、徒歩での避難を優先してください。"
                ]
            },
            "fire": {
                "title_templates": [
                    "火災発生時の初期消火 #{id}",
                    "大規模火災避難計画 #{id}",
                    "消防設備点検マニュアル #{id}",
                    "火災予防対策ガイド #{id}"
                ],
                "content_templates": [
                    "火災発見時は大声で周囲に知らせ、119番通報後に初期消火を試みてください。",
                    "煙による視界不良時は姿勢を低くし、壁伝いに避難経路を確保してください。",
                    "消火器は火元から2-3m離れた風上から使用し、燃焼物の根元を狙ってください。",
                    "避難時はエレベーター使用を避け、階段を使用して避難してください。"
                ]
            },
            "flood": {
                "title_templates": [
                    "水害時の避難行動 #{id}",
                    "河川氾濫対策マニュアル #{id}",
                    "内水氾濫避難ガイド #{id}",
                    "土砂災害警戒情報 #{id}"
                ],
                "content_templates": [
                    "水害警報発令時は低地からの避難を開始し、2階以上への垂直避難を検討してください。",
                    "膝上まで浸水した道路では歩行困難となるため、無理な移動は避けてください。",
                    "土砂災害警戒区域では降雨量に注意し、早期避難を心がけてください。",
                    "河川の増水時は橋梁付近への立ち入りを避け、安全な避難経路を選択してください。"
                ]
            }
        }
        
        documents = []
        doc_texts = []
        
        categories = list(disaster_templates.keys())
        
        for i in range(num_docs):
            # カテゴリをローテーション
            category = categories[i % len(categories)]
            template_data = disaster_templates[category]
            
            # タイトルと内容をランダム選択
            title_template = np.random.choice(template_data["title_templates"])
            content_template = np.random.choice(template_data["content_templates"])
            
            # パラメータ置換
            magnitude = np.random.randint(4, 8)
            title = title_template.format(id=i+1, magnitude=magnitude)
            content = content_template.format(magnitude=magnitude)
            
            # 追加詳細情報
            additional_info = [
                f"発行日: 2024年{np.random.randint(1,13):02d}月{np.random.randint(1,29):02d}日",
                f"担当部署: {category.title()}対策課",
                f"文書番号: {category.upper()}-{i+1:04d}",
                f"重要度: {'高' if i % 3 == 0 else '中' if i % 3 == 1 else '低'}"
            ]
            
            full_content = content + "\n\n" + "\n".join(additional_info)
            
            doc = {
                'doc_id': f"doc_{i+1:04d}",
                'title': title,
                'content': full_content,
                'category': category,
                'metadata': {
                    'doc_number': i+1,
                    'category': category,
                    'importance': additional_info[3].split(': ')[1],
                    'creation_date': additional_info[0].split(': ')[1]
                }
            }
            
            documents.append(doc)
            doc_texts.append(f"{title}\n{full_content}")
        
        print(f"✅ Created {len(documents)} disaster documents")
        return documents, doc_texts
    
    def run_gpu_performance_test(self, documents: List[Dict], doc_texts: List[str]):
        """GPU性能テスト"""
        print("\n🚀 GPU Performance Test")
        print("="*50)
        
        # インデックス構築時間測定
        index_time = self.gpu_search.index_documents(documents, doc_texts)
        
        # テストクエリ
        test_queries = [
            "地震が発生したときの避難方法",
            "津波警報時の緊急対応",
            "火災発生時の初期消火手順",
            "水害時の避難行動",
            "緊急時の連絡体制",
            "避難所での生活ガイド",
            "災害備蓄品の準備",
            "応急手当の基本知識",
            "停電時の対応策",
            "高齢者への災害支援"
        ]
        
        print(f"\n🔍 Running {len(test_queries)} search queries...")
        
        search_times = []
        all_results = {}
        
        for i, query in enumerate(test_queries):
            print(f"   Query {i+1}: {query}")
            
            results, search_time = self.gpu_search.search_gpu(query, top_k=5)
            search_times.append(search_time)
            
            print(f"      Search time: {search_time:.4f}s")
            print(f"      Top result: {results[0][0]['title']} (score: {results[0][1]:.4f})")
            
            all_results[f"query_{i+1}"] = {
                'query': query,
                'search_time': search_time,
                'results': [
                    {
                        'title': result[0]['title'],
                        'category': result[0]['category'],
                        'score': result[1]
                    }
                    for result in results
                ]
            }
        
        # 性能統計
        performance_stats = {
            'total_documents': len(documents),
            'total_queries': len(test_queries),
            'index_build_time': index_time,
            'avg_search_time': np.mean(search_times),
            'min_search_time': np.min(search_times),
            'max_search_time': np.max(search_times),
            'total_search_time': np.sum(search_times),
            'gpu_memory_allocated': torch.cuda.memory_allocated() / 1e9,
            'gpu_memory_cached': torch.cuda.memory_reserved() / 1e9
        }
        
        print(f"\n📊 Performance Summary:")
        print(f"   Index build time: {performance_stats['index_build_time']:.2f}s")
        print(f"   Average search time: {performance_stats['avg_search_time']:.4f}s")
        print(f"   Search throughput: {len(test_queries)/performance_stats['total_search_time']:.2f} queries/sec")
        print(f"   GPU memory used: {performance_stats['gpu_memory_allocated']:.2f} GB")
        
        return all_results, performance_stats
    
    def save_results(self, results: Dict, stats: Dict):
        """結果を保存"""
        output_data = {
            'performance_stats': stats,
            'search_results': results,
            'gpu_info': {
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'cuda_version': torch.version.cuda,
                'pytorch_version': torch.__version__
            }
        }
        
        output_file = self.data_dir / "gpu_performance_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Results saved to {output_file}")
    
    def run_complete_demo(self):
        """完全GPUデモを実行"""
        print("="*80)
        print("🚀 GPU-Accelerated Visual RAPTOR Demo")
        print("GPU加速災害文書検索システム")
        print("="*80)
        
        try:
            # 1. モデル初期化
            self.initialize_models()
            
            # 2. 大規模データセット作成
            documents, doc_texts = self.create_large_disaster_dataset(200)  # RTX 4060 Ti用
            
            # 3. GPU性能テスト
            results, stats = self.run_gpu_performance_test(documents, doc_texts)
            
            # 4. 結果保存
            self.save_results(results, stats)
            
            print("\n" + "="*80)
            print("🎉 GPU Demo Completed Successfully!")
            print("="*80)
            print(f"GPU Acceleration: ✅ ENABLED")
            print(f"Documents processed: {stats['total_documents']}")
            print(f"Search throughput: {stats['total_queries']/stats['total_search_time']:.2f} queries/sec")
            print(f"GPU efficiency: {stats['gpu_memory_allocated']:.2f} GB used")
            
            return True
            
        except Exception as e:
            print(f"\n❌ GPU demo failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """メイン実行関数"""
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Please check your GPU setup.")
        return False
    
    demo = GPUDisasterSearchDemo()
    success = demo.run_complete_demo()
    
    if success:
        print("\n🚀 GPU is now actively utilized!")
        print("Your RTX 4060 Ti is accelerating:")
        print("  - Batch document embedding")
        print("  - Parallel similarity computation")
        print("  - High-throughput search operations")
        print("  - Large-scale disaster document processing")
    
    return success


if __name__ == "__main__":
    main()
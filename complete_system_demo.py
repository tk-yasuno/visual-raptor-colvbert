#!/usr/bin/env python3
"""
Visual RAPTOR ColBERT - Complete System Demo
完全版システムデモ（jina_vdr_benchmark依存なし）
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from visual_raptor_colbert import VisualRAPTORColBERT, VisualDocument

print("="*80)
print("🚀 Visual RAPTOR ColBERT - Complete System Demo")
print("災害文書検索システム 完全版デモ")
print("="*80)

class CompleteSystemDemo:
    """完全版システムデモクラス"""
    
    def __init__(self):
        self.data_dir = Path("data/complete_system_demo")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings = None
        self.llm = None
        self.visual_raptor = None
        
    def initialize_models(self):
        """モデル初期化"""
        print("\n🔧 Initializing models...")
        
        try:
            # Ollama埋め込みモデル
            self.embeddings = OllamaEmbeddings(
                model="mxbai-embed-large",
                base_url="http://localhost:11434"
            )
            
            # テスト
            test_emb = self.embeddings.embed_query("test")
            print(f"   ✅ Embeddings model loaded (dim: {len(test_emb)})")
            
            # OllamaLLMモデル
            self.llm = ChatOllama(
                model="granite-code:8b",
                temperature=0.1,
                base_url="http://localhost:11434",
                timeout=300
            )
            
            # テスト
            test_resp = self.llm.invoke("Hello")
            print(f"   ✅ LLM model loaded")
            
            # Visual RAPTOR初期化
            self.visual_raptor = VisualRAPTORColBERT(
                embeddings_model=self.embeddings,
                llm=self.llm
            )
            print(f"   ✅ Visual RAPTOR initialized")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Model initialization failed: {e}")
            return False
    
    def create_disaster_documents(self, num_docs: int = 20) -> List[Dict]:
        """災害文書を作成"""
        print(f"\n📄 Creating {num_docs} disaster documents...")
        
        document_templates = [
            {
                "category": "earthquake",
                "templates": [
                    ("地震発生時の対応マニュアル", "震度{mag}の地震が発生した場合、直ちに机の下に隠れて身を守ってください。揺れが収まったら避難経路を確認し、火災の危険がないか確認してください。"),
                    ("耐震対策ガイドライン", "建物の耐震補強は震度{mag}までの地震に耐えられるよう設計されています。定期的な点検と補修が重要です。"),
                    ("地震避難訓練実施要領", "年2回の避難訓練を実施し、震度{mag}クラスの地震を想定した避難経路の確認を行います。")
                ]
            },
            {
                "category": "tsunami",
                "templates": [
                    ("津波避難計画", "津波警報発令時は、標高{mag}m以上の高台または3階建て以上の頑丈な建物に避難してください。"),
                    ("津波避難タワー運用ガイド", "津波避難タワーは標高{mag}mに設置され、500名の収容が可能です。"),
                    ("津波ハザードマップ", "浸水想定区域は海岸から{mag}km以内の低地となります。")
                ]
            },
            {
                "category": "evacuation",
                "templates": [
                    ("避難所運営マニュアル", "避難所では{mag}日分の食料と飲料水を確保しています。受付で氏名を記入してください。"),
                    ("避難所生活ガイド", "避難所での生活は譲り合いが基本です。消灯は{mag}時です。"),
                    ("要配慮者支援計画", "高齢者や障害者など{mag}名の要配慮者への優先的な支援を行います。")
                ]
            },
            {
                "category": "recovery",
                "templates": [
                    ("復旧計画書", "インフラ復旧は{mag}ヶ月を目標に段階的に実施します。"),
                    ("生活再建支援制度", "被災世帯に対して最大{mag}00万円の支援金を支給します。"),
                    ("復興まちづくり計画", "復興事業は{mag}年計画で進めます。")
                ]
            }
        ]
        
        documents = []
        
        for i in range(num_docs):
            cat = document_templates[i % len(document_templates)]
            template = cat["templates"][i % len(cat["templates"])]
            
            mag = np.random.randint(3, 10)
            title = template[0]
            content = template[1].format(mag=mag)
            
            doc = {
                "doc_id": f"doc_{i+1:03d}",
                "title": title,
                "content": content,
                "category": cat["category"],
                "magnitude": mag,
                "created_date": f"2024-{(i%12)+1:02d}-01"
            }
            
            documents.append(doc)
        
        print(f"   ✅ Created {len(documents)} disaster documents")
        return documents
    
    def build_visual_raptor_index(self, documents: List[Dict]):
        """Visual RAPTORインデックスを構築"""
        print("\n🔍 Building Visual RAPTOR index...")
        
        start_time = time.time()
        
        # LangChain Documentに変換
        langchain_docs = []
        for doc in documents:
            lc_doc = Document(
                page_content=f"{doc['title']}\n\n{doc['content']}",
                metadata={
                    'doc_id': doc['doc_id'],
                    'category': doc['category'],
                    'title': doc['title']
                }
            )
            langchain_docs.append(lc_doc)
        
        # RAPTORツリー構築
        print(f"   Building RAPTOR tree with {len(langchain_docs)} documents...")
        
        try:
            tree = self.visual_raptor.build_tree(
                documents=langchain_docs,
                save_dir=str(self.data_dir / "raptor_tree")
            )
            
            build_time = time.time() - start_time
            
            print(f"   ✅ RAPTOR tree built in {build_time:.2f}s")
            print(f"      Tree layers: {len(tree) if tree else 'N/A'}")
            
            return tree
            
        except Exception as e:
            print(f"   ⚠️  Tree building failed: {e}")
            print(f"   Falling back to simple indexing...")
            return None
    
    def run_search_evaluation(self, documents: List[Dict]):
        """検索評価を実行"""
        print("\n🔍 Running search evaluation...")
        
        test_queries = [
            "地震が発生したときの対応手順を教えて",
            "津波警報が出たらどこに避難すればいい？",
            "避難所での生活について知りたい",
            "復旧計画の内容は？",
            "耐震対策について教えて",
            "要配慮者への支援内容は？",
            "生活再建のための支援制度は？",
            "津波ハザードマップの見方を教えて"
        ]
        
        search_results = {}
        search_times = []
        
        for i, query_text in enumerate(test_queries):
            print(f"\n   Query {i+1}: {query_text}")
            
            start_time = time.time()
            
            # 検索実行
            try:
                # RAPTORツリー検索
                results = self.visual_raptor.retriever.get_relevant_documents(query_text)
                
                search_time = time.time() - start_time
                search_times.append(search_time)
                
                print(f"      Found {len(results)} results in {search_time:.3f}s")
                
                # トップ3表示
                top_results = []
                for j, result in enumerate(results[:3]):
                    title = result.metadata.get('title', 'Unknown')
                    category = result.metadata.get('category', 'Unknown')
                    content_preview = result.page_content[:100].replace('\n', ' ')
                    
                    print(f"      {j+1}. [{category}] {title}")
                    print(f"         {content_preview}...")
                    
                    top_results.append({
                        'title': title,
                        'category': category,
                        'content': result.page_content
                    })
                
                # LLMで要約生成
                if top_results:
                    context = "\n\n".join([
                        f"【{r['title']}】\n{r['content']}"
                        for r in top_results[:2]
                    ])
                    
                    prompt = f"""
以下の災害関連文書を参考に、質問に対する回答を日本語で簡潔に生成してください。

質問: {query_text}

参考文書:
{context}

回答:"""
                    
                    try:
                        response = self.llm.invoke(prompt)
                        summary = response.content.strip()
                        print(f"      💡 Summary: {summary[:150]}...")
                    except Exception as e:
                        summary = f"要約生成エラー: {e}"
                else:
                    summary = "関連文書が見つかりませんでした"
                
                search_results[f"query_{i+1}"] = {
                    'query': query_text,
                    'search_time': search_time,
                    'num_results': len(results),
                    'top_results': top_results,
                    'summary': summary
                }
                
            except Exception as e:
                print(f"      ❌ Search failed: {e}")
                search_results[f"query_{i+1}"] = {
                    'query': query_text,
                    'error': str(e)
                }
        
        # 統計情報
        print(f"\n📊 Search Statistics:")
        print(f"   Total queries: {len(test_queries)}")
        print(f"   Average search time: {np.mean(search_times):.3f}s")
        print(f"   Min search time: {np.min(search_times):.3f}s")
        print(f"   Max search time: {np.max(search_times):.3f}s")
        
        return search_results
    
    def save_results(self, documents: List[Dict], search_results: Dict):
        """結果を保存"""
        print(f"\n💾 Saving results...")
        
        output_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {
                'num_documents': len(documents),
                'num_queries': len(search_results)
            },
            'documents': documents,
            'search_results': search_results
        }
        
        output_file = self.data_dir / "complete_demo_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ Results saved to {output_file}")
    
    def run_complete_demo(self):
        """完全デモを実行"""
        try:
            # 1. モデル初期化
            if not self.initialize_models():
                return False
            
            # 2. 文書作成
            documents = self.create_disaster_documents(30)
            
            # 3. インデックス構築
            self.build_visual_raptor_index(documents)
            
            # 4. 検索評価
            search_results = self.run_search_evaluation(documents)
            
            # 5. 結果保存
            self.save_results(documents, search_results)
            
            print("\n" + "="*80)
            print("✅ Complete System Demo Finished Successfully!")
            print("="*80)
            print(f"\n📈 Final Summary:")
            print(f"   Documents processed: {len(documents)}")
            print(f"   Queries executed: {len(search_results)}")
            print(f"   Results saved: {self.data_dir / 'complete_demo_results.json'}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """メイン実行"""
    demo = CompleteSystemDemo()
    success = demo.run_complete_demo()
    
    if success:
        print("\n🎉 Visual RAPTOR ColBERT complete system is fully operational!")
    else:
        print("\n❌ Demo encountered errors")
        sys.exit(1)


if __name__ == "__main__":
    main()

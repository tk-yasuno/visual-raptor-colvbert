#!/usr/bin/env python3
"""
Simple Visual RAPTOR ColBERT Demo
シンプルなVisual RAPTOR ColBERTシステムのデモンストレーション
- Ollama統合確認
- 基本的な検索機能テスト
- 災害文書検索デモ
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# LangChain imports
try:
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from langchain.schema import Document
    print("✅ LangChain Ollama imports successful")
except ImportError as e:
    print(f"❌ LangChain import error: {e}")
    sys.exit(1)


class SimpleDisasterDocument:
    """シンプルな災害文書クラス"""
    def __init__(self, doc_id: str, title: str, content: str, category: str, image_path: str = None):
        self.doc_id = doc_id
        self.title = title
        self.content = content
        self.category = category
        self.image_path = image_path
        self.metadata = {
            'title': title,
            'category': category,
            'doc_id': doc_id
        }


class SimpleVisualRAPTOR:
    """シンプルなVisual RAPTORクラス"""
    
    def __init__(self, embeddings_model, llm):
        self.embeddings = embeddings_model
        self.llm = llm
        self.documents = []
        self.document_embeddings = []
        
    def add_documents(self, documents: List[SimpleDisasterDocument]):
        """文書を追加"""
        print(f"📚 Adding {len(documents)} documents to index...")
        
        self.documents = documents
        
        # 各文書の埋め込みを計算
        texts = [f"{doc.title}\n{doc.content}" for doc in documents]
        
        start_time = time.time()
        embeddings = []
        
        for i, text in enumerate(texts):
            embedding = self.embeddings.embed_query(text)
            embeddings.append(embedding)
            
            if (i + 1) % 5 == 0:
                print(f"   Processed {i + 1}/{len(texts)} documents")
        
        self.document_embeddings = np.array(embeddings)
        embed_time = time.time() - start_time
        
        print(f"✅ Documents indexed in {embed_time:.2f}s")
        print(f"   Embedding dimension: {self.document_embeddings.shape[1]}")
        
    def search(self, query: str, top_k: int = 5) -> List[Tuple[SimpleDisasterDocument, float]]:
        """クエリで検索"""
        if not self.documents:
            return []
        
        # クエリの埋め込み
        query_embedding = np.array(self.embeddings.embed_query(query))
        
        # コサイン類似度計算
        similarities = np.dot(self.document_embeddings, query_embedding) / (
            np.linalg.norm(self.document_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # トップK取得
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            score = similarities[idx]
            results.append((doc, score))
        
        return results
    
    def generate_summary(self, query: str, top_documents: List[SimpleDisasterDocument]) -> str:
        """検索結果を要約"""
        if not top_documents:
            return "関連する文書が見つかりませんでした。"
        
        # 文書内容を結合
        context = "\n\n".join([
            f"【{doc.title}】\n{doc.content}"
            for doc in top_documents[:3]  # 上位3文書
        ])
        
        # LLMで要約生成
        prompt = f"""
以下の災害関連文書を参考に、質問「{query}」に対する回答を日本語で生成してください。

関連文書:
{context}

回答:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"要約生成中にエラーが発生しました: {e}"


class SimpleDemo:
    """シンプルデモクラス"""
    
    def __init__(self):
        self.data_dir = Path("data/simple_demo")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings = None
        self.llm = None
        self.raptor = None
        
    def initialize_ollama(self) -> bool:
        """Ollamaモデルを初期化"""
        print("🔧 Initializing Ollama models...")
        
        try:
            # 埋め込みモデル
            print("   Loading embedding model (mxbai-embed-large)...")
            self.embeddings = OllamaEmbeddings(
                model="mxbai-embed-large",
                base_url="http://localhost:11434"
            )
            
            # テスト
            test_embed = self.embeddings.embed_query("test")
            print(f"   Embedding model loaded (dim: {len(test_embed)})")
            
            # LLMモデル
            print("   Loading LLM model (granite-code:8b)...")
            self.llm = ChatOllama(
                model="granite-code:8b",
                temperature=0.1,
                base_url="http://localhost:11434",
                timeout=120
            )
            
            # テスト
            test_response = self.llm.invoke("Hello")
            print(f"   LLM model loaded (response: {test_response.content[:30]}...)")
            
            # Visual RAPTOR初期化
            self.raptor = SimpleVisualRAPTOR(self.embeddings, self.llm)
            
            print("✅ Ollama models initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Ollama: {e}")
            print("Please ensure:")
            print("1. Ollama is running (ollama serve)")
            print("2. Models are downloaded:")
            print("   - ollama pull mxbai-embed-large")
            print("   - ollama pull granite-code:8b")
            return False
    
    def create_sample_disaster_documents(self) -> List[SimpleDisasterDocument]:
        """サンプル災害文書を作成"""
        print("📄 Creating sample disaster documents...")
        
        sample_docs = [
            SimpleDisasterDocument(
                doc_id="doc_001",
                title="地震発生時の避難手順",
                content="""
1. まず身の安全を確保してください
2. 揺れが収まったら火の始末をしてください
3. 避難経路を確認し、落下物に注意してください
4. 指定避難所に向かってください
5. 家族の安否確認を行ってください
""",
                category="evacuation_procedure"
            ),
            SimpleDisasterDocument(
                doc_id="doc_002", 
                title="緊急時連絡先一覧",
                content="""
【緊急時連絡先】
• 消防署: 119
• 警察署: 110  
• 市役所災害対策本部: 045-123-4567
• 避難所（小学校）: 045-123-4568
• 医療機関（総合病院）: 045-123-4569
• ガス会社緊急連絡先: 045-123-4570
""",
                category="emergency_contact"
            ),
            SimpleDisasterDocument(
                doc_id="doc_003",
                title="避難所での生活ガイド",
                content="""
【避難所での過ごし方】
• 受付で氏名・連絡先を記入してください
• 指定された場所で生活してください
• 食事は決められた時間に配布されます
• 衛生管理に気をつけてください
• ペットは専用エリアでお世話ください
• 夜間は静粛にお過ごしください
""",
                category="shelter_guide"
            ),
            SimpleDisasterDocument(
                doc_id="doc_004",
                title="備蓄品チェックリスト",
                content="""
【必要な備蓄品】
食料品:
• 非常食（3日分）
• 飲料水（1人1日3リットル×3日分）
• 缶詰・レトルト食品

生活用品:
• 懐中電灯・電池
• ラジオ
• 応急医薬品
• 毛布・タオル
• 着替え・下着
""",
                category="emergency_supplies"
            ),
            SimpleDisasterDocument(
                doc_id="doc_005",
                title="津波警報発令時の対応",
                content="""
【津波警報時の行動】
1. 直ちに高台や頑丈な建物の3階以上に避難
2. 自動車での避難は渋滞の原因となるため徒歩で
3. 海岸・河川には絶対に近づかない
4. 警報解除まで絶対に低地に戻らない
5. 正確な情報収集に努める（ラジオ・防災無線）
""",
                category="tsunami_response"
            ),
            SimpleDisasterDocument(
                doc_id="doc_006",
                title="応急手当の基本",
                content="""
【基本的な応急手当】
止血:
• 清潔なガーゼで傷口を直接圧迫
• 出血が多い場合は心臓より高い位置に

骨折の疑い:
• 患部を固定し、動かさない
• 副木で患部を支える

やけど:
• 流水で十分に冷やす
• 水ぶくれは破らない
""",
                category="first_aid"
            ),
            SimpleDisasterDocument(
                doc_id="doc_007",
                title="停電時の対処法",
                content="""
【停電時の安全対策】
照明:
• 懐中電灯・ランタンを使用
• ろうそくは火災の危険があるため避ける

冷蔵庫:
• 扉の開閉を最小限に
• 保冷剤を活用

通信:
• 携帯電話の節電モード使用
• 乾電池式ラジオで情報収集
""",
                category="power_outage"
            ),
            SimpleDisasterDocument(
                doc_id="doc_008",
                title="高齢者・障害者支援ガイド",
                content="""
【要配慮者への支援】
高齢者:
• 移動の際は転倒に注意
• 薬の管理・服用確認
• 体調変化に注意

障害者:
• 視覚障害：手引きによる誘導
• 聴覚障害：筆談・手話での情報伝達
• 車椅子利用者：段差のない避難経路確保
""",
                category="special_needs"
            )
        ]
        
        print(f"✅ Created {len(sample_docs)} disaster documents")
        return sample_docs
    
    def create_sample_images(self, documents: List[SimpleDisasterDocument]):
        """サンプル画像を作成"""
        print("🖼️ Creating sample document images...")
        
        images_dir = self.data_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        for doc in documents:
            # 簡単な文書画像を作成
            img = Image.new('RGB', (800, 600), 'white')
            draw = ImageDraw.Draw(img)
            
            # タイトル
            draw.text((50, 50), doc.title, fill='black')
            
            # 内容（最初の数行）
            content_lines = doc.content.strip().split('\n')[:10]
            for i, line in enumerate(content_lines):
                if line.strip():
                    draw.text((50, 100 + i*25), line[:60], fill='blue')
            
            # カテゴリラベル
            draw.text((600, 550), f"分類: {doc.category}", fill='red')
            
            # 保存
            img_path = images_dir / f"{doc.doc_id}.png"
            img.save(img_path)
            doc.image_path = str(img_path)
        
        print(f"✅ Created images for {len(documents)} documents")
    
    def run_search_demo(self, documents: List[SimpleDisasterDocument]) -> Dict[str, Any]:
        """検索デモを実行"""
        print("🔍 Running search demonstration...")
        
        # 文書をインデックスに追加
        self.raptor.add_documents(documents)
        
        # テストクエリ
        test_queries = [
            "地震が起きたときはどうすればいいですか？",
            "緊急時の連絡先を教えて",
            "避難所での生活について知りたい",
            "津波警報が出たときの対応は？",
            "応急手当の方法を教えて",
            "停電したときの対処法は？"
        ]
        
        search_results = {}
        
        for i, query in enumerate(test_queries):
            print(f"\n📋 Query {i+1}: {query}")
            
            start_time = time.time()
            
            # 検索実行
            results = self.raptor.search(query, top_k=3)
            search_time = time.time() - start_time
            
            print(f"   Search time: {search_time:.3f}s")
            print("   Top results:")
            
            result_info = []
            for j, (doc, score) in enumerate(results):
                print(f"     {j+1}. {doc.title} (score: {score:.4f})")
                result_info.append({
                    'doc_id': doc.doc_id,
                    'title': doc.title,
                    'category': doc.category,
                    'score': float(score),
                    'content_preview': doc.content[:100] + "..."
                })
            
            # LLMで要約生成
            print("   Generating summary...")
            top_docs = [doc for doc, _ in results]
            summary = self.raptor.generate_summary(query, top_docs)
            print(f"   Summary: {summary[:100]}...")
            
            search_results[f"query_{i+1}"] = {
                'query': query,
                'search_time': search_time,
                'results': result_info,
                'summary': summary
            }
        
        return search_results
    
    def save_results(self, results: Dict[str, Any]):
        """結果を保存"""
        output_file = self.data_dir / "search_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Results saved to {output_file}")
    
    def run_complete_demo(self) -> bool:
        """完全なデモを実行"""
        print("="*80)
        print("🚀 Simple Visual RAPTOR Demo with Ollama")
        print("災害文書検索システム シンプルデモ")
        print("="*80)
        
        try:
            # 1. Ollama初期化
            if not self.initialize_ollama():
                return False
            
            # 2. サンプル文書作成
            documents = self.create_sample_disaster_documents()
            
            # 3. サンプル画像作成
            self.create_sample_images(documents)
            
            # 4. 検索デモ実行
            search_results = self.run_search_demo(documents)
            
            # 5. 結果保存
            self.save_results(search_results)
            
            # 6. サマリー表示
            print("\n" + "="*80)
            print("📊 Demo Summary")
            print("="*80)
            print(f"Documents processed: {len(documents)}")
            print(f"Queries executed: {len(search_results)}")
            
            avg_search_time = np.mean([
                result['search_time'] for result in search_results.values()
            ])
            print(f"Average search time: {avg_search_time:.3f}s")
            
            print("\n🎉 Simple demo completed successfully!")
            print(f"Results saved in: {self.data_dir}")
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """メイン実行関数"""
    demo = SimpleDemo()
    success = demo.run_complete_demo()
    
    if success:
        print("\n✅ Ollama integration successful!")
        print("Your Visual RAPTOR ColBERT system is working with:")
        print("  - Ollama mxbai-embed-large (embeddings)")
        print("  - Ollama granite-code:8b (LLM)")
        print("  - Disaster document search & summarization")
        print("\nNext: Try the full integrated system!")
    else:
        print("\n❌ Demo failed. Please check Ollama setup.")
        sys.exit(1)


if __name__ == "__main__":
    main()
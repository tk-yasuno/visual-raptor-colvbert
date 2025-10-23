"""
JinaVDR (Visual Document Retrieval) ベンチマーク統合

震災関連文書の検索性能評価用ベンチマーク
- 日本語データセット対応
- 複雑なレイアウト文書（グラフ、表、スキャン、スクリーンショット）
- OCRテキスト付き画像：500～1000枚規模
- 震災関連文書（避難所マップ、復旧計画、行政通知など）

Reference: jina-ai/jina-vdr
"""

import os
import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import urllib.request
import zipfile
from dataclasses import dataclass
import pandas as pd
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModel
import time

from visual_raptor_colbert import VisualDocument, VisualDocumentProcessor


@dataclass
class VDRQuery:
    """VDRクエリ"""
    query_id: str
    text: str
    language: str = "ja"
    category: str = "disaster"
    difficulty: str = "medium"


@dataclass
class VDRDocument:
    """VDR文書"""
    doc_id: str
    image_path: str
    text_content: str
    category: str
    subcategory: str
    metadata: Dict


@dataclass
class VDRRelevanceJudgment:
    """関連性判定"""
    query_id: str
    doc_id: str
    relevance: int  # 0: 無関連, 1: 関連, 2: 高関連


class JinaVDRBenchmark:
    """
    JinaVDR ベンチマーク管理クラス
    
    震災関連文書の検索性能評価
    """
    
    def __init__(
        self,
        data_dir: str = "data/jina_vdr",
        language: str = "ja",
        dataset_size: str = "small"  # small: 500, medium: 1000, large: 2000+
    ):
        self.data_dir = Path(data_dir)
        self.language = language
        self.dataset_size = dataset_size
        
        # ディレクトリ構造
        self.images_dir = self.data_dir / "images"
        self.queries_dir = self.data_dir / "queries"
        self.annotations_dir = self.data_dir / "annotations"
        
        # データストレージ
        self.queries: List[VDRQuery] = []
        self.documents: List[VDRDocument] = []
        self.relevance_judgments: List[VDRRelevanceJudgment] = []
        
        self._create_directories()
        print(f"📊 JinaVDR Benchmark initialized")
        print(f"   Language: {language}")
        print(f"   Dataset size: {dataset_size}")
        print(f"   Data directory: {data_dir}")
    
    def _create_directories(self):
        """必要なディレクトリを作成"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.queries_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)
    
    def generate_disaster_queries(self, num_queries: int = 50) -> List[VDRQuery]:
        """災害関連クエリを生成"""
        disaster_query_templates = [
            # 避難・安全
            ("避難所の場所を教えてください", "evacuation", "easy"),
            ("緊急避難経路はどこですか", "evacuation", "easy"),
            ("安全な避難場所はありますか", "evacuation", "easy"),
            ("津波避難ビルの位置", "evacuation", "medium"),
            ("避難所の収容人数", "evacuation", "medium"),
            
            # 復旧・復興
            ("復旧工事の進捗状況", "recovery", "medium"),
            ("インフラ復旧計画", "recovery", "hard"),
            ("仮設住宅の配置図", "recovery", "medium"),
            ("復興まちづくり計画", "recovery", "hard"),
            ("道路復旧スケジュール", "recovery", "medium"),
            
            # 行政・情報
            ("災害対策本部からの通知", "administration", "easy"),
            ("避難指示の詳細", "administration", "easy"),
            ("支援物資の配布場所", "administration", "medium"),
            ("被害状況の報告書", "administration", "medium"),
            ("復興予算の内訳", "administration", "hard"),
            
            # 防災教育
            ("防災訓練の手順", "education", "easy"),
            ("災害時の行動指針", "education", "easy"),
            ("津波の危険性について", "education", "medium"),
            ("地震発生時の対応", "education", "medium"),
            ("防災マップの見方", "education", "medium"),
            
            # 被害・統計
            ("被害の統計データ", "statistics", "medium"),
            ("人的被害の状況", "statistics", "hard"),
            ("建物被害の分析", "statistics", "hard"),
            ("経済的損失の評価", "statistics", "hard"),
            ("復興進捗の指標", "statistics", "hard"),
        ]
        
        queries = []
        for i in range(num_queries):
            template = random.choice(disaster_query_templates)
            query = VDRQuery(
                query_id=f"query_{i:03d}",
                text=template[0],
                language=self.language,
                category=template[1],
                difficulty=template[2]
            )
            queries.append(query)
        
        self.queries = queries
        return queries
    
    def create_synthetic_documents(
        self,
        num_documents: int = 500,
        base_text_dir: str = None
    ) -> List[VDRDocument]:
        """合成文書を作成"""
        if base_text_dir is None:
            base_text_dir = str(Path(__file__).parent / "0_base_tsunami-lesson-rag")
        
        # ベースとなるテキストデータを読み込み
        base_text = self._load_base_text_data(base_text_dir)
        
        # 文書カテゴリ
        doc_categories = {
            "evacuation_map": "避難マップ",
            "recovery_plan": "復旧計画書",
            "admin_notice": "行政通知",
            "damage_report": "被害報告書",
            "education_material": "防災教育資料",
            "statistics": "統計データ",
            "infrastructure": "インフラ情報",
            "support_info": "支援情報"
        }
        
        documents = []
        visual_processor = VisualDocumentProcessor()
        
        for i in range(num_documents):
            # カテゴリをランダム選択
            category = random.choice(list(doc_categories.keys()))
            subcategory = doc_categories[category]
            
            # 合成画像を生成（実際の実装では実際の画像を使用）
            image_path = self._create_synthetic_image(
                category, i, subcategory
            )
            
            # テキストコンテンツを生成
            text_content = self._generate_document_text(
                base_text, category, subcategory
            )
            
            # メタデータ
            metadata = {
                "created_date": "2024-01-01",
                "source": "synthetic",
                "format": "image_with_text",
                "processing_method": "ocr_extracted"
            }
            
            doc = VDRDocument(
                doc_id=f"doc_{i:04d}",
                image_path=str(image_path),
                text_content=text_content,
                category=category,
                subcategory=subcategory,
                metadata=metadata
            )
            documents.append(doc)
        
        self.documents = documents
        print(f"✅ Created {len(documents)} synthetic documents")
        return documents
    
    def _load_base_text_data(self, base_dir: str) -> str:
        """ベーステキストデータを読み込み"""
        try:
            with open(Path(base_dir) / "tohoku_earthquake_data.txt", 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            # フォールバック：基本的な災害関連テキスト
            return """\n東日本大震災における教訓と対応\n\n1. 避難行動\n津波警報発表時には、直ちに高台への避難が重要である。\n釜石市では「津波てんでんこ」の教えにより多くの命が救われた。\n\n2. 情報伝達\n防災行政無線、携帯電話の緊急速報メール等の活用\n停電時の情報収集手段の確保が課題\n\n3. 避難所運営\n避難所の運営には地域住民の協力が不可欠\n要配慮者への支援体制の構築\n\n4. 復旧・復興\nカウンターパート方式による支援\n創造的復興の理念に基づく取り組み
"""
    
    def _generate_document_text(self, base_text: str, category: str, subcategory: str) -> str:
        """文書テキストを生成"""
        # カテゴリに応じたテキスト生成
        category_texts = {
            "evacuation_map": f"""\n{subcategory}\n\n避難場所：○○小学校体育館\n収容人数：200名\n所在地：○○市○○町1-1-1\n連絡先：XXX-XXX-XXXX\n\n避難経路：\n- 国道○号線を北上\n- ○○橋を渡る\n- 信号を右折\n\n注意事項：\n- 徒歩での避難を推奨\n- ペット同伴可\n- 食料・水は3日分持参\n""",
            "recovery_plan": f"""\n{subcategory}\n\n復旧計画概要\n対象地域：○○地区\n期間：2024年4月～2025年3月\n\n主要項目：\n1. 道路復旧（○○道路他）\n2. 上下水道復旧\n3. 電力インフラ復旧\n4. 通信設備復旧\n\n予算：総額○○億円\n進捗：現在30%完了\n""",
            "admin_notice": f"""\n行政からのお知らせ\n\n件名：{subcategory}\n発行日：2024年1月15日\n発行者：○○市災害対策本部\n\n市民の皆様へ\n\n○○地区における復旧作業について下記の通りお知らせいたします。\n\n詳細は市ホームページをご確認ください。\n問い合わせ先：○○市役所 XXX-XXX-XXXX
"""
        }
        
        return category_texts.get(category, f"{subcategory}\
        \
        関連情報：\
        {base_text[:500]}...")
    
    def _create_synthetic_image(
        self, 
        category: str, 
        index: int, 
        subcategory: str
    ) -> Path:
        """合成画像を作成"""
        # 簡単な合成画像を作成（実際の実装では、より複雑な画像生成が必要）
        from PIL import Image, ImageDraw, ImageFont
        
        # 画像サイズとレイアウト
        width, height = 800, 600
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        # フォント（システムにない場合はデフォルト）
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            title_font = ImageFont.truetype("arial.ttf", 30)
        except:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # カテゴリに応じた画像生成
        if category == "evacuation_map":
            # 避難マップ風
            draw.rectangle([50, 50, width-50, height-50], outline='black', width=2)
            draw.text((60, 60), "避難マップ", fill='black', font=title_font)
            draw.text((60, 100), subcategory, fill='blue', font=font)
            
            # 簡単な地図要素
            draw.rectangle([100, 150, 200, 200], fill='green', outline='black')
            draw.text((105, 165), "避難所", fill='white', font=font)
            
            draw.line([(250, 175), (400, 175)], fill='gray', width=3)
            draw.text((300, 180), "避難経路", fill='black', font=font)
            
        elif category == "recovery_plan":
            # 復旧計画書風
            draw.rectangle([50, 50, width-50, height-50], outline='navy', width=3)
            draw.text((60, 60), "復旧計画書", fill='navy', font=title_font)
            draw.text((60, 100), subcategory, fill='black', font=font)
            
            # 表風の要素
            for i in range(4):
                y = 150 + i * 40
                draw.rectangle([100, y, 600, y+30], outline='black')
                draw.text((110, y+5), f"項目 {i+1}", fill='black', font=font)
        
        elif category == "admin_notice":
            # 行政通知風
            draw.rectangle([50, 50, width-50, height-50], outline='red', width=2)
            draw.text((60, 60), "行政通知", fill='red', font=title_font)
            draw.text((60, 100), subcategory, fill='black', font=font)
            
            # 重要マーク
            draw.ellipse([width-150, 60, width-60, 120], fill='red')
            draw.text((width-140, 80), "重要", fill='white', font=font)
        
        # ファイル保存
        image_path = self.images_dir / f"{category}_{index:04d}.png"
        image.save(image_path)
        
        return image_path
    
    def generate_relevance_judgments(
        self,
        num_judgments_per_query: int = 10
    ) -> List[VDRRelevanceJudgment]:
        """関連性判定を生成"""
        judgments = []
        
        for query in self.queries:
            # クエリカテゴリに基づいて関連文書を選択
            relevant_docs = [doc for doc in self.documents 
                           if self._is_relevant(query.category, doc.category)]
            
            # 関連文書から一部を選択
            selected_docs = random.sample(
                relevant_docs, 
                min(num_judgments_per_query, len(relevant_docs))
            )
            
            for doc in selected_docs:
                # 関連性スコアを決定
                relevance = self._calculate_relevance(query, doc)
                
                judgment = VDRRelevanceJudgment(
                    query_id=query.query_id,
                    doc_id=doc.doc_id,
                    relevance=relevance
                )
                judgments.append(judgment)
        
        self.relevance_judgments = judgments
        print(f"✅ Generated {len(judgments)} relevance judgments")
        return judgments
    
    def _is_relevant(self, query_category: str, doc_category: str) -> bool:
        """クエリと文書の関連性を判定"""
        # カテゴリマッピング
        category_mapping = {
            "evacuation": ["evacuation_map", "admin_notice", "education_material"],
            "recovery": ["recovery_plan", "infrastructure", "statistics"],
            "administration": ["admin_notice", "damage_report", "support_info"],
            "education": ["education_material", "evacuation_map"],
            "statistics": ["statistics", "damage_report"]
        }
        
        return doc_category in category_mapping.get(query_category, [])
    
    def _calculate_relevance(self, query: VDRQuery, doc: VDRDocument) -> int:
        """関連性スコアを計算"""
        # 簡単なヒューリスティック
        if query.category == doc.category:
            return random.choice([1, 2])  # 関連または高関連
        elif self._is_relevant(query.category, doc.category):
            return 1  # 関連
        else:
            return 0  # 無関連
    
    def save_benchmark_data(self):
        """ベンチマークデータを保存"""
        # クエリ保存
        queries_data = [
            {
                "query_id": q.query_id,
                "text": q.text,
                "language": q.language,
                "category": q.category,
                "difficulty": q.difficulty
            }
            for q in self.queries
        ]
        
        with open(self.queries_dir / "queries.json", 'w', encoding='utf-8') as f:
            json.dump(queries_data, f, ensure_ascii=False, indent=2)
        
        # 文書保存
        documents_data = [
            {
                "doc_id": d.doc_id,
                "image_path": d.image_path,
                "text_content": d.text_content,
                "category": d.category,
                "subcategory": d.subcategory,
                "metadata": d.metadata
            }
            for d in self.documents
        ]
        
        with open(self.annotations_dir / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(documents_data, f, ensure_ascii=False, indent=2)
        
        # 関連性判定保存
        judgments_data = [
            {
                "query_id": j.query_id,
                "doc_id": j.doc_id,
                "relevance": j.relevance
            }
            for j in self.relevance_judgments
        ]
        
        with open(self.annotations_dir / "relevance_judgments.json", 'w', encoding='utf-8') as f:
            json.dump(judgments_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Benchmark data saved to {self.data_dir}")
    
    def load_benchmark_data(self):
        """ベンチマークデータを読み込み"""
        # クエリ読み込み
        queries_file = self.queries_dir / "queries.json"
        if queries_file.exists():
            with open(queries_file, 'r', encoding='utf-8') as f:
                queries_data = json.load(f)
            
            self.queries = [
                VDRQuery(**q) for q in queries_data
            ]
        
        # 文書読み込み
        documents_file = self.annotations_dir / "documents.json"
        if documents_file.exists():
            with open(documents_file, 'r', encoding='utf-8') as f:
                documents_data = json.load(f)
            
            self.documents = [
                VDRDocument(**d) for d in documents_data
            ]
        
        # 関連性判定読み込み
        judgments_file = self.annotations_dir / "relevance_judgments.json"
        if judgments_file.exists():
            with open(judgments_file, 'r', encoding='utf-8') as f:
                judgments_data = json.load(f)
            
            self.relevance_judgments = [
                VDRRelevanceJudgment(**j) for j in judgments_data
            ]
        
        print(f"✅ Benchmark data loaded from {self.data_dir}")
        print(f"   Queries: {len(self.queries)}")
        print(f"   Documents: {len(self.documents)}")
        print(f"   Judgments: {len(self.relevance_judgments)}")
    
    def get_benchmark_statistics(self) -> Dict[str, Any]:
        """ベンチマーク統計を取得"""
        stats = {
            "num_queries": len(self.queries),
            "num_documents": len(self.documents),
            "num_judgments": len(self.relevance_judgments),
            "query_categories": {},
            "document_categories": {},
            "relevance_distribution": {0: 0, 1: 0, 2: 0}
        }
        
        # クエリカテゴリ分布
        for query in self.queries:
            stats["query_categories"][query.category] = \
                stats["query_categories"].get(query.category, 0) + 1
        
        # 文書カテゴリ分布
        for doc in self.documents:
            stats["document_categories"][doc.category] = \
                stats["document_categories"].get(doc.category, 0) + 1
        
        # 関連性分布
        for judgment in self.relevance_judgments:
            stats["relevance_distribution"][judgment.relevance] += 1
        
        return stats


def create_jina_vdr_benchmark(
    data_dir: str = "data/jina_vdr",
    num_queries: int = 50,
    num_documents: int = 500,
    dataset_size: str = "small"
    ) -> JinaVDRBenchmark:
    """JinaVDRベンチマークを作成"""
    print("="*80)
    print("Creating JinaVDR Benchmark for Disaster Document Retrieval")
    print("="*80)
    
    # ベンチマーク初期化
    benchmark = JinaVDRBenchmark(
        data_dir=data_dir,
        language="ja",
        dataset_size=dataset_size
    )
    
    # クエリ生成
    print(f"\
    🔍 Generating {num_queries} queries...")
    benchmark.generate_disaster_queries(num_queries)
    
    # 文書生成
    print(f"\
    📄 Creating {num_documents} synthetic documents...")
    benchmark.create_synthetic_documents(num_documents)
    
    # 関連性判定生成
    print(f"\
    📊 Generating relevance judgments...")
    benchmark.generate_relevance_judgments()
    
    # データ保存
    print(f"\
    💾 Saving benchmark data...")
    benchmark.save_benchmark_data()
    
    # 統計表示
    stats = benchmark.get_benchmark_statistics()
    print(f"\
    📈 Benchmark Statistics:")
    print(f"   Queries: {stats['num_queries']}")
    print(f"   Documents: {stats['num_documents']}")
    print(f"   Judgments: {stats['num_judgments']}")
    print(f"   Query categories: {list(stats['query_categories'].keys())}")
    print(f"   Document categories: {list(stats['document_categories'].keys())}")
    
    print(f"\n✅ JinaVDR benchmark created successfully!")
    print(f"   Data directory: {data_dir}")
    
    return benchmark


def main():
    """メイン実行関数"""
    # ベンチマーク作成
    benchmark = create_jina_vdr_benchmark(
        data_dir="data/jina_vdr_disaster",
        num_queries=50,
        num_documents=500,
        dataset_size="small"
    )
    
    print("\n" + "="*80)
    print("JinaVDR Benchmark Setup Complete")
    print("="*80)


class DisasterDocumentGenerator:
    """災害ドキュメント生成クラス（compare_encoders.py用）"""
    
    def __init__(self):
        self.disaster_types = [
            "earthquake", "tsunami", "flood", "typhoon", "landslide",
            "volcanic_eruption", "fire", "evacuation", "rescue", "recovery"
        ]
        self.locations = [
            "Tokyo", "Osaka", "Sendai", "Fukushima", "Kobe",
            "Nagoya", "Sapporo", "Hiroshima", "Kumamoto", "Iwate"
        ]
    
    def generate_disaster_queries(self, num_queries: int = 20) -> List[Dict]:
        """災害関連クエリを生成"""
        query_templates = [
            "避難所の場所を教えてください",
            "{}の被害状況について",
            "{}地域の復旧計画は？",
            "災害発生時の緊急連絡先",
            "{}での救援物資配布場所",
            "津波警報の発令状況",
            "地震の震度分布マップ",
            "避難経路の確認方法",
            "{}の安否情報確認",
            "災害支援ボランティアの募集"
        ]
        
        queries = []
        for i in range(num_queries):
            template = random.choice(query_templates)
            location = random.choice(self.locations)
            disaster_type = random.choice(self.disaster_types)
            
            if "{}" in template:
                query_text = template.format(location)
            else:
                query_text = template
            
            queries.append({
                'query_id': f"Q{i+1:03d}",
                'query': query_text,
                'disaster_type': disaster_type,
                'location': location
            })
        
        return queries
    
    def create_synthetic_documents(self, num_documents: int = 100) -> List[Dict]:
        """合成災害ドキュメントを生成"""
        document_templates = [
            "【緊急】{}で{}が発生しました。住民の皆様は直ちに指定避難所に避難してください。",
            "{}地域の復旧作業が進んでいます。現在、{}の被害が報告されています。",
            "避難所マップ：{}地区の避難所一覧。最寄りの避難所は{}です。",
            "{}における{}の被害状況：建物倒壊、道路寸断、ライフライン停止。",
            "救援物資配布のお知らせ：{}地域で{}時より配布を開始します。",
            "安否情報：{}地区の住民の皆様の安否確認を行っています。",
            "復旧計画：{}地域の{}復旧作業は来週から開始予定です。",
            "ボランティア募集：{}での災害支援活動にご協力ください。",
            "気象警報：{}地方に{}警報が発令されました。厳重に警戒してください。",
            "避難指示解除：{}地区の避難指示が解除されました。"
        ]
        
        documents = []
        for i in range(num_documents):
            template = random.choice(document_templates)
            location = random.choice(self.locations)
            disaster_type = random.choice(self.disaster_types)
            
            # ドキュメント内容を生成
            content = template.format(location, disaster_type)
            
            # 追加情報を付加
            additional_info = [
                f"発表時刻: 2024年{random.randint(1,12)}月{random.randint(1,28)}日 {random.randint(0,23)}時{random.randint(0,59)}分",
                f"対象地域: {location}および周辺地域",
                f"避難者数: 約{random.randint(100,10000)}名",
                f"被害状況: 詳細調査中",
                "問い合わせ: 災害対策本部 0120-XXX-XXX"
            ]
            
            content += "\n\n" + "\n".join(random.sample(additional_info, 3))
            
            documents.append({
                'doc_id': f"D{i+1:03d}",
                'title': f"{disaster_type.upper()} - {location}",
                'content': content,
                'disaster_type': disaster_type,
                'location': location,
                'timestamp': f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
            })
        
        return documents


if __name__ == "__main__":
    main()
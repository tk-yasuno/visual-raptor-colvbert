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
        visual_processor = VisualDocumentProcessor()\n        \n        for i in range(num_documents):\n            # カテゴリをランダム選択\n            category = random.choice(list(doc_categories.keys()))\n            subcategory = doc_categories[category]\n            \n            # 合成画像を生成（実際の実装では実際の画像を使用）\n            image_path = self._create_synthetic_image(\n                category, i, subcategory\n            )\n            \n            # テキストコンテンツを生成\n            text_content = self._generate_document_text(\n                base_text, category, subcategory\n            )\n            \n            # メタデータ\n            metadata = {\n                "created_date": "2024-01-01",\n                "source": "synthetic",\n                "format": "image_with_text",\n                "processing_method": "ocr_extracted"\n            }\n            \n            doc = VDRDocument(\n                doc_id=f"doc_{i:04d}",\n                image_path=str(image_path),\n                text_content=text_content,\n                category=category,\n                subcategory=subcategory,\n                metadata=metadata\n            )\n            documents.append(doc)\n        \n        self.documents = documents\n        print(f"✅ Created {len(documents)} synthetic documents")\n        return documents\n    \n    def _load_base_text_data(self, base_dir: str) -> str:\n        """ベーステキストデータを読み込み"""\n        try:\n            with open(Path(base_dir) / "tohoku_earthquake_data.txt", 'r', encoding='utf-8') as f:\n                return f.read()\n        except FileNotFoundError:\n            # フォールバック：基本的な災害関連テキスト\n            return """\n東日本大震災における教訓と対応\n\n1. 避難行動\n津波警報発表時には、直ちに高台への避難が重要である。\n釜石市では「津波てんでんこ」の教えにより多くの命が救われた。\n\n2. 情報伝達\n防災行政無線、携帯電話の緊急速報メール等の活用\n停電時の情報収集手段の確保が課題\n\n3. 避難所運営\n避難所の運営には地域住民の協力が不可欠\n要配慮者への支援体制の構築\n\n4. 復旧・復興\nカウンターパート方式による支援\n創造的復興の理念に基づく取り組み\n"""\n    \n    def _generate_document_text(self, base_text: str, category: str, subcategory: str) -> str:\n        """文書テキストを生成"""\n        # カテゴリに応じたテキスト生成\n        category_texts = {\n            "evacuation_map": f"""\n{subcategory}\n\n避難場所：○○小学校体育館\n収容人数：200名\n所在地：○○市○○町1-1-1\n連絡先：XXX-XXX-XXXX\n\n避難経路：\n- 国道○号線を北上\n- ○○橋を渡る\n- 信号を右折\n\n注意事項：\n- 徒歩での避難を推奨\n- ペット同伴可\n- 食料・水は3日分持参\n""",\n            "recovery_plan": f"""\n{subcategory}\n\n復旧計画概要\n対象地域：○○地区\n期間：2024年4月～2025年3月\n\n主要項目：\n1. 道路復旧（○○道路他）\n2. 上下水道復旧\n3. 電力インフラ復旧\n4. 通信設備復旧\n\n予算：総額○○億円\n進捗：現在30%完了\n""",\n            "admin_notice": f"""\n行政からのお知らせ\n\n件名：{subcategory}\n発行日：2024年1月15日\n発行者：○○市災害対策本部\n\n市民の皆様へ\n\n○○地区における復旧作業について下記の通りお知らせいたします。\n\n詳細は市ホームページをご確認ください。\n問い合わせ先：○○市役所 XXX-XXX-XXXX\n"""\n        }\n        \n        return category_texts.get(category, f"{subcategory}\\n\\n関連情報：\\n{base_text[:500]}...")\n    \n    def _create_synthetic_image(\n        self, \n        category: str, \n        index: int, \n        subcategory: str\n    ) -> Path:\n        """合成画像を作成"""\n        # 簡単な合成画像を作成（実際の実装では、より複雑な画像生成が必要）\n        from PIL import Image, ImageDraw, ImageFont\n        \n        # 画像サイズとレイアウト\n        width, height = 800, 600\n        image = Image.new('RGB', (width, height), 'white')\n        draw = ImageDraw.Draw(image)\n        \n        # フォント（システムにない場合はデフォルト）\n        try:\n            font = ImageFont.truetype("arial.ttf", 20)\n            title_font = ImageFont.truetype("arial.ttf", 30)\n        except:\n            font = ImageFont.load_default()\n            title_font = ImageFont.load_default()\n        \n        # カテゴリに応じた画像生成\n        if category == "evacuation_map":\n            # 避難マップ風\n            draw.rectangle([50, 50, width-50, height-50], outline='black', width=2)\n            draw.text((60, 60), "避難マップ", fill='black', font=title_font)\n            draw.text((60, 100), subcategory, fill='blue', font=font)\n            \n            # 簡単な地図要素\n            draw.rectangle([100, 150, 200, 200], fill='green', outline='black')\n            draw.text((105, 165), "避難所", fill='white', font=font)\n            \n            draw.line([(250, 175), (400, 175)], fill='gray', width=3)\n            draw.text((300, 180), "避難経路", fill='black', font=font)\n            \n        elif category == "recovery_plan":\n            # 復旧計画書風\n            draw.rectangle([50, 50, width-50, height-50], outline='navy', width=3)\n            draw.text((60, 60), "復旧計画書", fill='navy', font=title_font)\n            draw.text((60, 100), subcategory, fill='black', font=font)\n            \n            # 表風の要素\n            for i in range(4):\n                y = 150 + i * 40\n                draw.rectangle([100, y, 600, y+30], outline='black')\n                draw.text((110, y+5), f"項目 {i+1}", fill='black', font=font)\n        \n        elif category == "admin_notice":\n            # 行政通知風\n            draw.rectangle([50, 50, width-50, height-50], outline='red', width=2)\n            draw.text((60, 60), "行政通知", fill='red', font=title_font)\n            draw.text((60, 100), subcategory, fill='black', font=font)\n            \n            # 重要マーク\n            draw.ellipse([width-150, 60, width-60, 120], fill='red')\n            draw.text((width-140, 80), "重要", fill='white', font=font)\n        \n        # ファイル保存\n        image_path = self.images_dir / f"{category}_{index:04d}.png"\n        image.save(image_path)\n        \n        return image_path\n    \n    def generate_relevance_judgments(\n        self,\n        num_judgments_per_query: int = 10\n    ) -> List[VDRRelevanceJudgment]:\n        """関連性判定を生成"""\n        judgments = []\n        \n        for query in self.queries:\n            # クエリカテゴリに基づいて関連文書を選択\n            relevant_docs = [doc for doc in self.documents \n                           if self._is_relevant(query.category, doc.category)]\n            \n            # 関連文書から一部を選択\n            selected_docs = random.sample(\n                relevant_docs, \n                min(num_judgments_per_query, len(relevant_docs))\n            )\n            \n            for doc in selected_docs:\n                # 関連性スコアを決定\n                relevance = self._calculate_relevance(query, doc)\n                \n                judgment = VDRRelevanceJudgment(\n                    query_id=query.query_id,\n                    doc_id=doc.doc_id,\n                    relevance=relevance\n                )\n                judgments.append(judgment)\n        \n        self.relevance_judgments = judgments\n        print(f"✅ Generated {len(judgments)} relevance judgments")\n        return judgments\n    \n    def _is_relevant(self, query_category: str, doc_category: str) -> bool:\n        """クエリと文書の関連性を判定"""\n        # カテゴリマッピング\n        category_mapping = {\n            "evacuation": ["evacuation_map", "admin_notice", "education_material"],\n            "recovery": ["recovery_plan", "infrastructure", "statistics"],\n            "administration": ["admin_notice", "damage_report", "support_info"],\n            "education": ["education_material", "evacuation_map"],\n            "statistics": ["statistics", "damage_report"]\n        }\n        \n        return doc_category in category_mapping.get(query_category, [])\n    \n    def _calculate_relevance(self, query: VDRQuery, doc: VDRDocument) -> int:\n        """関連性スコアを計算"""\n        # 簡単なヒューリスティック\n        if query.category == doc.category:\n            return random.choice([1, 2])  # 関連または高関連\n        elif self._is_relevant(query.category, doc.category):\n            return 1  # 関連\n        else:\n            return 0  # 無関連\n    \n    def save_benchmark_data(self):\n        """ベンチマークデータを保存"""\n        # クエリ保存\n        queries_data = [\n            {\n                "query_id": q.query_id,\n                "text": q.text,\n                "language": q.language,\n                "category": q.category,\n                "difficulty": q.difficulty\n            }\n            for q in self.queries\n        ]\n        \n        with open(self.queries_dir / "queries.json", 'w', encoding='utf-8') as f:\n            json.dump(queries_data, f, ensure_ascii=False, indent=2)\n        \n        # 文書保存\n        documents_data = [\n            {\n                "doc_id": d.doc_id,\n                "image_path": d.image_path,\n                "text_content": d.text_content,\n                "category": d.category,\n                "subcategory": d.subcategory,\n                "metadata": d.metadata\n            }\n            for d in self.documents\n        ]\n        \n        with open(self.annotations_dir / "documents.json", 'w', encoding='utf-8') as f:\n            json.dump(documents_data, f, ensure_ascii=False, indent=2)\n        \n        # 関連性判定保存\n        judgments_data = [\n            {\n                "query_id": j.query_id,\n                "doc_id": j.doc_id,\n                "relevance": j.relevance\n            }\n            for j in self.relevance_judgments\n        ]\n        \n        with open(self.annotations_dir / "relevance_judgments.json", 'w', encoding='utf-8') as f:\n            json.dump(judgments_data, f, ensure_ascii=False, indent=2)\n        \n        print(f"✅ Benchmark data saved to {self.data_dir}")\n    \n    def load_benchmark_data(self):\n        """ベンチマークデータを読み込み"""\n        # クエリ読み込み\n        queries_file = self.queries_dir / "queries.json"\n        if queries_file.exists():\n            with open(queries_file, 'r', encoding='utf-8') as f:\n                queries_data = json.load(f)\n            \n            self.queries = [\n                VDRQuery(**q) for q in queries_data\n            ]\n        \n        # 文書読み込み\n        documents_file = self.annotations_dir / "documents.json"\n        if documents_file.exists():\n            with open(documents_file, 'r', encoding='utf-8') as f:\n                documents_data = json.load(f)\n            \n            self.documents = [\n                VDRDocument(**d) for d in documents_data\n            ]\n        \n        # 関連性判定読み込み\n        judgments_file = self.annotations_dir / "relevance_judgments.json"\n        if judgments_file.exists():\n            with open(judgments_file, 'r', encoding='utf-8') as f:\n                judgments_data = json.load(f)\n            \n            self.relevance_judgments = [\n                VDRRelevanceJudgment(**j) for j in judgments_data\n            ]\n        \n        print(f"✅ Benchmark data loaded from {self.data_dir}")\n        print(f"   Queries: {len(self.queries)}")\n        print(f"   Documents: {len(self.documents)}")\n        print(f"   Judgments: {len(self.relevance_judgments)}")\n    \n    def get_benchmark_statistics(self) -> Dict[str, Any]:\n        """ベンチマーク統計を取得"""\n        stats = {\n            "num_queries": len(self.queries),\n            "num_documents": len(self.documents),\n            "num_judgments": len(self.relevance_judgments),\n            "query_categories": {},\n            "document_categories": {},\n            "relevance_distribution": {0: 0, 1: 0, 2: 0}\n        }\n        \n        # クエリカテゴリ分布\n        for query in self.queries:\n            stats["query_categories"][query.category] = \\\n                stats["query_categories"].get(query.category, 0) + 1\n        \n        # 文書カテゴリ分布\n        for doc in self.documents:\n            stats["document_categories"][doc.category] = \\\n                stats["document_categories"].get(doc.category, 0) + 1\n        \n        # 関連性分布\n        for judgment in self.relevance_judgments:\n            stats["relevance_distribution"][judgment.relevance] += 1\n        \n        return stats\n\n\ndef create_jina_vdr_benchmark(\n    data_dir: str = "data/jina_vdr",\n    num_queries: int = 50,\n    num_documents: int = 500,\n    dataset_size: str = "small"\n) -> JinaVDRBenchmark:\n    """JinaVDRベンチマークを作成"""\n    print("="*80)\n    print("Creating JinaVDR Benchmark for Disaster Document Retrieval")\n    print("="*80)\n    \n    # ベンチマーク初期化\n    benchmark = JinaVDRBenchmark(\n        data_dir=data_dir,\n        language="ja",\n        dataset_size=dataset_size\n    )\n    \n    # クエリ生成\n    print(f"\\n🔍 Generating {num_queries} queries...")\n    benchmark.generate_disaster_queries(num_queries)\n    \n    # 文書生成\n    print(f"\\n📄 Creating {num_documents} synthetic documents...")\n    benchmark.create_synthetic_documents(num_documents)\n    \n    # 関連性判定生成\n    print(f"\\n📊 Generating relevance judgments...")\n    benchmark.generate_relevance_judgments()\n    \n    # データ保存\n    print(f"\\n💾 Saving benchmark data...")\n    benchmark.save_benchmark_data()\n    \n    # 統計表示\n    stats = benchmark.get_benchmark_statistics()\n    print(f"\\n📈 Benchmark Statistics:")\n    print(f"   Queries: {stats['num_queries']}")\n    print(f"   Documents: {stats['num_documents']}")\n    print(f"   Judgments: {stats['num_judgments']}")\n    print(f"   Query categories: {list(stats['query_categories'].keys())}")\n    print(f"   Document categories: {list(stats['document_categories'].keys())}")\n    \n    print(f"\\n✅ JinaVDR benchmark created successfully!")\n    print(f"   Data directory: {data_dir}")\n    \n    return benchmark\n\n\ndef main():\n    """メイン実行関数"""\n    # ベンチマーク作成\n    benchmark = create_jina_vdr_benchmark(\n        data_dir="data/jina_vdr_disaster",\n        num_queries=50,\n        num_documents=500,\n        dataset_size="small"\n    )\n    \n    print("\\n" + "="*80)\n    print("JinaVDR Benchmark Setup Complete")\n    print("="*80)\n\n\nif __name__ == "__main__":\n    main()
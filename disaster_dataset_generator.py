"""
災害関連文書データセット生成器

避難所マップ、復旧計画、行政通知などの震災関連文書を生成
- テンプレートベースの文書生成
- 多様なレイアウトパターン
- 日本語コンテンツ
- OCR処理に適した画像形式
"""

import os
import json
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import io
import base64


class DisasterDocumentGenerator:
    """災害関連文書生成器"""
    
    def __init__(self, output_dir: str = "disaster_documents"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # フォント設定
        self.fonts = self._setup_fonts()
        
        # カラーパレット
        self.colors = {
            'urgent': '#FF4444',
            'warning': '#FF8800',
            'info': '#0088FF',
            'safe': '#00AA44',
            'background': '#FFFFFF',
            'text': '#333333',
            'border': '#666666'
        }
        
        # データテンプレート
        self.disaster_data = self._load_disaster_data()
        
        print(f"📁 Disaster Document Generator initialized")
        print(f"   Output directory: {output_dir}")
    
    def _setup_fonts(self) -> Dict[str, Any]:
        """フォント設定"""
        fonts = {}
        try:
            # Windows標準フォント
            fonts['title'] = ImageFont.truetype("msgothic.ttc", 24)
            fonts['header'] = ImageFont.truetype("msgothic.ttc", 18)
            fonts['body'] = ImageFont.truetype("msgothic.ttc", 14)
            fonts['small'] = ImageFont.truetype("msgothic.ttc", 12)
        except:
            try:
                # フォールバック
                fonts['title'] = ImageFont.truetype("arial.ttf", 24)
                fonts['header'] = ImageFont.truetype("arial.ttf", 18)
                fonts['body'] = ImageFont.truetype("arial.ttf", 14)
                fonts['small'] = ImageFont.truetype("arial.ttf", 12)
            except:
                # デフォルトフォント
                fonts['title'] = ImageFont.load_default()
                fonts['header'] = ImageFont.load_default()
                fonts['body'] = ImageFont.load_default()
                fonts['small'] = ImageFont.load_default()
        
        return fonts
    
    def _load_disaster_data(self) -> Dict[str, List[str]]:
        """災害関連データを読み込み"""
        return {
            'evacuation_sites': [
                '○○小学校体育館', '△△中学校', '□□公民館', 
                '××市民センター', '◇◇総合体育館', '☆☆公園',
                '○○神社境内', '△△寺本堂', '□□コミュニティセンター'
            ],
            'roads': [
                '国道○号線', '県道△△号線', '市道□□線',
                '○○通り', '△△街道', '□□大橋', '××トンネル'
            ],
            'areas': [
                '○○地区', '△△町', '□□丁目', '××区',
                '◇◇団地', '☆☆ニュータウン', '○○工業団地'
            ],
            'facilities': [
                '市役所', '消防署', '警察署', '病院', '郵便局',
                '銀行', 'コンビニエンスストア', 'ガソリンスタンド'
            ],
            'disaster_types': [
                '地震', '津波', '洪水', '土砂災害', '台風',
                '大雨', '火災', '停電', '断水'
            ],
            'dates': [
                '2024年3月11日', '2024年9月1日', '2024年1月17日',
                '2024年7月15日', '2024年11月3日'
            ]
        }
    
    def generate_evacuation_map(
        self, 
        doc_id: str,
        area_name: str = None
    ) -> Tuple[str, Dict[str, Any]]:
        """避難マップを生成"""
        if area_name is None:
            area_name = random.choice(self.disaster_data['areas'])
        
        # 画像サイズ
        width, height = 800, 600
        image = Image.new('RGB', (width, height), self.colors['background'])
        draw = ImageDraw.Draw(image)
        
        # タイトル
        title = f"{area_name} 避難マップ"
        draw.rectangle([20, 20, width-20, 80], fill=self.colors['urgent'], outline=self.colors['border'])
        draw.text((30, 35), title, fill='white', font=self.fonts['title'])
        
        # 凡例\n        legend_y = 100\n        draw.text((30, legend_y), "凡例", fill=self.colors['text'], font=self.fonts['header'])\n        \n        # 避難所マーカー\n        draw.rectangle([30, legend_y + 25, 50, legend_y + 45], fill=self.colors['safe'])\n        draw.text((60, legend_y + 30), "避難所", fill=self.colors['text'], font=self.fonts['body'])\n        \n        # 危険区域マーカー\n        draw.rectangle([30, legend_y + 50, 50, legend_y + 70], fill=self.colors['urgent'])\n        draw.text((60, legend_y + 55), "危険区域", fill=self.colors['text'], font=self.fonts['body'])\n        \n        # 避難経路マーカー\n        draw.line([(30, legend_y + 80), (50, legend_y + 80)], fill=self.colors['info'], width=3)\n        draw.text((60, legend_y + 75), "避難経路", fill=self.colors['text'], font=self.fonts['body'])\n        \n        # 地図エリア\n        map_area = [200, 120, width-50, height-100]\n        draw.rectangle(map_area, outline=self.colors['border'], width=2)\n        \n        # 避難所をランダム配置\n        evacuation_sites = random.sample(self.disaster_data['evacuation_sites'], 3)\n        site_positions = []\n        \n        for i, site in enumerate(evacuation_sites):\n            x = random.randint(map_area[0] + 50, map_area[2] - 100)\n            y = random.randint(map_area[1] + 30, map_area[3] - 50)\n            \n            # 避難所マーカー\n            draw.rectangle([x-15, y-15, x+15, y+15], fill=self.colors['safe'])\n            draw.text((x+20, y-10), site, fill=self.colors['text'], font=self.fonts['small'])\n            \n            site_positions.append((x, y, site))\n        \n        # 危険区域を描画\n        danger_x = random.randint(map_area[0] + 20, map_area[2] - 120)\n        danger_y = random.randint(map_area[1] + 20, map_area[3] - 80)\n        draw.rectangle([danger_x, danger_y, danger_x + 80, danger_y + 60], \n                      fill=self.colors['urgent'], outline=self.colors['border'])\n        draw.text((danger_x + 10, danger_y + 25), "危険", fill='white', font=self.fonts['body'])\n        \n        # 避難経路を描画\n        for i in range(len(site_positions) - 1):\n            x1, y1, _ = site_positions[i]\n            x2, y2, _ = site_positions[i + 1]\n            draw.line([(x1, y1), (x2, y2)], fill=self.colors['info'], width=3)\n        \n        # 情報テキスト\n        info_y = height - 80\n        draw.text((30, info_y), "発行：○○市災害対策本部", fill=self.colors['text'], font=self.fonts['small'])\n        draw.text((30, info_y + 15), f"発行日：{random.choice(self.disaster_data['dates'])}", \n                 fill=self.colors['text'], font=self.fonts['small'])\n        draw.text((30, info_y + 30), "緊急時は指定避難所に避難してください", \n                 fill=self.colors['text'], font=self.fonts['small'])\n        \n        # 保存\n        image_path = self.output_dir / f"evacuation_map_{doc_id}.png"\n        image.save(image_path)\n        \n        # メタデータ\n        metadata = {\n            'document_type': 'evacuation_map',\n            'area_name': area_name,\n            'evacuation_sites': evacuation_sites,\n            'generated_date': datetime.now().isoformat(),\n            'image_size': (width, height)\n        }\n        \n        return str(image_path), metadata\n    \n    def generate_recovery_plan(self, doc_id: str) -> Tuple[str, Dict[str, Any]]:\n        """復旧計画書を生成"""\n        width, height = 800, 600\n        image = Image.new('RGB', (width, height), self.colors['background'])\n        draw = ImageDraw.Draw(image)\n        \n        # ヘッダー\n        area_name = random.choice(self.disaster_data['areas'])\n        title = f"{area_name} 復旧計画書"\n        draw.rectangle([20, 20, width-20, 70], fill=self.colors['info'], outline=self.colors['border'])\n        draw.text((30, 35), title, fill='white', font=self.fonts['title'])\n        \n        # 計画概要\n        y_pos = 100\n        draw.text((30, y_pos), "復旧計画概要", fill=self.colors['text'], font=self.fonts['header'])\n        y_pos += 30\n        \n        plan_items = [\n            "1. 道路インフラの復旧",\n            "2. 上下水道の復旧", \n            "3. 電力供給の復旧",\n            "4. 通信設備の復旧",\n            "5. 公共施設の復旧"\n        ]\n        \n        for item in plan_items:\n            draw.text((40, y_pos), item, fill=self.colors['text'], font=self.fonts['body'])\n            y_pos += 25\n        \n        # 進捗表\n        y_pos += 20\n        draw.text((30, y_pos), "進捗状況", fill=self.colors['text'], font=self.fonts['header'])\n        y_pos += 30\n        \n        # 表のヘッダー\n        table_headers = ["項目", "進捗率", "完了予定"]\n        col_widths = [200, 100, 150]\n        x_positions = [50, 250, 350]\n        \n        # ヘッダー行\n        for i, header in enumerate(table_headers):\n            draw.rectangle([x_positions[i], y_pos, x_positions[i] + col_widths[i], y_pos + 25],\n                          outline=self.colors['border'])\n            draw.text((x_positions[i] + 5, y_pos + 5), header, fill=self.colors['text'], font=self.fonts['body'])\n        \n        y_pos += 25\n        \n        # データ行\n        progress_data = [\n            ("道路復旧", "75%", "2024年6月"),\n            ("水道復旧", "60%", "2024年5月"),\n            ("電力復旧", "90%", "2024年4月"),\n            ("通信復旧", "45%", "2024年7月")\n        ]\n        \n        for row_data in progress_data:\n            for i, data in enumerate(row_data):\n                draw.rectangle([x_positions[i], y_pos, x_positions[i] + col_widths[i], y_pos + 25],\n                              outline=self.colors['border'])\n                draw.text((x_positions[i] + 5, y_pos + 5), data, fill=self.colors['text'], font=self.fonts['body'])\n            y_pos += 25\n        \n        # フッター\n        footer_y = height - 60\n        draw.text((30, footer_y), f"発行：{area_name}復興推進課", fill=self.colors['text'], font=self.fonts['small'])\n        draw.text((30, footer_y + 15), f"作成日：{random.choice(self.disaster_data['dates'])}", \n                 fill=self.colors['text'], font=self.fonts['small'])\n        draw.text((30, footer_y + 30), "問い合わせ：TEL 0XX-XXX-XXXX", \n                 fill=self.colors['text'], font=self.fonts['small'])\n        \n        # 保存\n        image_path = self.output_dir / f"recovery_plan_{doc_id}.png"\n        image.save(image_path)\n        \n        metadata = {\n            'document_type': 'recovery_plan',\n            'area_name': area_name,\n            'plan_items': plan_items,\n            'progress_data': progress_data,\n            'generated_date': datetime.now().isoformat()\n        }\n        \n        return str(image_path), metadata\n    \n    def generate_admin_notice(self, doc_id: str) -> Tuple[str, Dict[str, Any]]:\n        """行政通知を生成"""\n        width, height = 800, 600\n        image = Image.new('RGB', (width, height), self.colors['background'])\n        draw = ImageDraw.Draw(image)\n        \n        # 重要マーク\n        draw.ellipse([width-120, 20, width-20, 120], fill=self.colors['urgent'])\n        draw.text((width-90, 60), "重要", fill='white', font=self.fonts['header'])\n        \n        # タイトル\n        notice_type = random.choice(['避難指示', '避難勧告', '復旧作業通知', '支援物資配布'])\n        title = f"{notice_type}について"\n        draw.text((30, 30), "行政からのお知らせ", fill=self.colors['text'], font=self.fonts['header'])\n        draw.text((30, 60), title, fill=self.colors['urgent'], font=self.fonts['title'])\n        \n        # 発行情報\n        y_pos = 120\n        issue_date = random.choice(self.disaster_data['dates'])\n        draw.text((30, y_pos), f"発行日：{issue_date}", fill=self.colors['text'], font=self.fonts['body'])\n        draw.text((30, y_pos + 20), "発行者：○○市災害対策本部", fill=self.colors['text'], font=self.fonts['body'])\n        \n        # 本文\n        y_pos += 60\n        content_lines = [\n            "市民の皆様へ",\n            "",\n            f"{random.choice(self.disaster_data['disaster_types'])}の影響により、",\n            f"{random.choice(self.disaster_data['areas'])}において",\n            f"{notice_type}を発令いたします。",\n            "",\n            "【対象地域】",\n            f"・{random.choice(self.disaster_data['areas'])}",\n            f"・{random.choice(self.disaster_data['areas'])}",\n            "",\n            "【指定避難所】",\n            f"・{random.choice(self.disaster_data['evacuation_sites'])}",\n            f"・{random.choice(self.disaster_data['evacuation_sites'])}",\n            "",\n            "詳細は市ホームページをご確認ください。"\n        ]\n        \n        for line in content_lines:\n            draw.text((30, y_pos), line, fill=self.colors['text'], font=self.fonts['body'])\n            y_pos += 20\n        \n        # 連絡先\n        contact_y = height - 80\n        draw.rectangle([20, contact_y - 10, width - 20, height - 20], \n                      outline=self.colors['border'], width=2)\n        draw.text((30, contact_y), "問い合わせ先", fill=self.colors['text'], font=self.fonts['header'])\n        draw.text((30, contact_y + 25), "○○市役所 災害対策本部", fill=self.colors['text'], font=self.fonts['body'])\n        draw.text((30, contact_y + 45), "TEL: 0XX-XXX-XXXX（24時間対応）", \n                 fill=self.colors['text'], font=self.fonts['body'])\n        \n        # 保存\n        image_path = self.output_dir / f"admin_notice_{doc_id}.png"\n        image.save(image_path)\n        \n        metadata = {\n            'document_type': 'admin_notice',\n            'notice_type': notice_type,\n            'issue_date': issue_date,\n            'content_lines': content_lines,\n            'generated_date': datetime.now().isoformat()\n        }\n        \n        return str(image_path), metadata\n    \n    def generate_damage_report(self, doc_id: str) -> Tuple[str, Dict[str, Any]]:\n        """被害報告書を生成"""\n        width, height = 800, 600\n        image = Image.new('RGB', (width, height), self.colors['background'])\n        draw = ImageDraw.Draw(image)\n        \n        # ヘッダー\n        title = "災害被害状況報告書"\n        draw.rectangle([20, 20, width-20, 70], fill=self.colors['warning'], outline=self.colors['border'])\n        draw.text((30, 35), title, fill='white', font=self.fonts['title'])\n        \n        # 報告日時\n        y_pos = 90\n        report_date = random.choice(self.disaster_data['dates'])\n        draw.text((30, y_pos), f"報告日時：{report_date} 14:30", fill=self.colors['text'], font=self.fonts['body'])\n        draw.text((30, y_pos + 20), "報告者：○○市災害対策本部", fill=self.colors['text'], font=self.fonts['body'])\n        \n        # 被害概要\n        y_pos += 60\n        draw.text((30, y_pos), "被害概要", fill=self.colors['text'], font=self.fonts['header'])\n        y_pos += 30\n        \n        # 被害データ生成\n        damage_data = {\n            '人的被害': {\n                '死者': random.randint(0, 5),\n                '負傷者': random.randint(5, 50),\n                '行方不明者': random.randint(0, 3)\n            },\n            '建物被害': {\n                '全壊': random.randint(10, 100),\n                '半壊': random.randint(50, 200),\n                '一部損壊': random.randint(100, 500)\n            },\n            'インフラ被害': {\n                '停電戸数': random.randint(1000, 10000),\n                '断水戸数': random.randint(500, 5000),\n                '道路通行止め': random.randint(5, 20)\n            }\n        }\n        \n        # 被害データ表示\n        for category, data in damage_data.items():\n            draw.text((30, y_pos), f"【{category}】", fill=self.colors['urgent'], font=self.fonts['body'])\n            y_pos += 25\n            \n            for item, count in data.items():\n                if '戸数' in item or 'か所' in item:\n                    draw.text((50, y_pos), f"{item}：{count:,}", fill=self.colors['text'], font=self.fonts['body'])\n                else:\n                    draw.text((50, y_pos), f"{item}：{count}名", fill=self.colors['text'], font=self.fonts['body'])\n                y_pos += 20\n            y_pos += 10\n        \n        # フッター\n        footer_y = height - 60\n        draw.text((30, footer_y), "※この情報は速報値であり、今後変更される可能性があります", \n                 fill=self.colors['warning'], font=self.fonts['small'])\n        draw.text((30, footer_y + 15), f"次回報告予定：{report_date} 18:00", \n                 fill=self.colors['text'], font=self.fonts['small'])\n        \n        # 保存\n        image_path = self.output_dir / f"damage_report_{doc_id}.png"\n        image.save(image_path)\n        \n        metadata = {\n            'document_type': 'damage_report',\n            'report_date': report_date,\n            'damage_data': damage_data,\n            'generated_date': datetime.now().isoformat()\n        }\n        \n        return str(image_path), metadata\n    \n    def generate_support_info(self, doc_id: str) -> Tuple[str, Dict[str, Any]]:\n        """支援情報を生成"""\n        width, height = 800, 600\n        image = Image.new('RGB', (width, height), self.colors['background'])\n        draw = ImageDraw.Draw(image)\n        \n        # ヘッダー\n        title = "支援物資配布のお知らせ"\n        draw.rectangle([20, 20, width-20, 70], fill=self.colors['safe'], outline=self.colors['border'])\n        draw.text((30, 35), title, fill='white', font=self.fonts['title'])\n        \n        # 配布情報\n        y_pos = 90\n        distribution_date = random.choice(self.disaster_data['dates'])\n        draw.text((30, y_pos), f"配布日：{distribution_date}", fill=self.colors['text'], font=self.fonts['body'])\n        draw.text((30, y_pos + 20), "配布時間：9:00〜17:00", fill=self.colors['text'], font=self.fonts['body'])\n        \n        # 配布場所\n        y_pos += 60\n        draw.text((30, y_pos), "配布場所", fill=self.colors['text'], font=self.fonts['header'])\n        y_pos += 25\n        \n        distribution_sites = random.sample(self.disaster_data['evacuation_sites'], 3)\n        for site in distribution_sites:\n            draw.text((50, y_pos), f"• {site}", fill=self.colors['text'], font=self.fonts['body'])\n            y_pos += 20\n        \n        # 配布物資\n        y_pos += 20\n        draw.text((30, y_pos), "配布物資", fill=self.colors['text'], font=self.fonts['header'])\n        y_pos += 25\n        \n        supplies = [\n            "飲料水（1人1日3リットル）",\n            "非常食（アルファ米、パン）",\n            "毛布・タオル",\n            "衛生用品",\n            "乾電池・懐中電灯"\n        ]\n        \n        for supply in supplies:\n            draw.text((50, y_pos), f"• {supply}", fill=self.colors['text'], font=self.fonts['body'])\n            y_pos += 20\n        \n        # 注意事項\n        y_pos += 20\n        draw.text((30, y_pos), "注意事項", fill=self.colors['urgent'], font=self.fonts['header'])\n        y_pos += 25\n        \n        notes = [\n            "• 身分証明書をお持ちください",\n            "• 受け取りは世帯主または代理人1名",\n            "• 配布数には限りがあります",\n            "• マスク着用をお願いします"\n        ]\n        \n        for note in notes:\n            draw.text((50, y_pos), note, fill=self.colors['text'], font=self.fonts['body'])\n            y_pos += 20\n        \n        # フッター\n        footer_y = height - 40\n        draw.text((30, footer_y), "主催：○○市・○○社会福祉協議会", \n                 fill=self.colors['text'], font=self.fonts['small'])\n        \n        # 保存\n        image_path = self.output_dir / f"support_info_{doc_id}.png"\n        image.save(image_path)\n        \n        metadata = {\n            'document_type': 'support_info',\n            'distribution_date': distribution_date,\n            'distribution_sites': distribution_sites,\n            'supplies': supplies,\n            'generated_date': datetime.now().isoformat()\n        }\n        \n        return str(image_path), metadata\n    \n    def generate_dataset(\n        self,\n        num_documents: int = 100,\n        document_types: List[str] = None\n    ) -> List[Dict[str, Any]]:\n        """データセットを生成"""\n        if document_types is None:\n            document_types = [\n                'evacuation_map', 'recovery_plan', 'admin_notice',\n                'damage_report', 'support_info'\n            ]\n        \n        dataset = []\n        generators = {\n            'evacuation_map': self.generate_evacuation_map,\n            'recovery_plan': self.generate_recovery_plan,\n            'admin_notice': self.generate_admin_notice,\n            'damage_report': self.generate_damage_report,\n            'support_info': self.generate_support_info\n        }\n        \n        print(f"📄 Generating {num_documents} disaster documents...")\n        \n        for i in range(num_documents):\n            doc_type = random.choice(document_types)\n            doc_id = f"{doc_type}_{i:04d}"\n            \n            try:\n                image_path, metadata = generators[doc_type](doc_id)\n                \n                document_info = {\n                    'doc_id': doc_id,\n                    'image_path': image_path,\n                    'metadata': metadata\n                }\n                \n                dataset.append(document_info)\n                \n                if (i + 1) % 20 == 0:\n                    print(f"   Generated {i + 1}/{num_documents} documents")\n                    \n            except Exception as e:\n                print(f"Error generating {doc_id}: {e}")\n        \n        # データセット情報を保存\n        dataset_info = {\n            'total_documents': len(dataset),\n            'document_types': document_types,\n            'generation_date': datetime.now().isoformat(),\n            'documents': dataset\n        }\n        \n        info_path = self.output_dir / 'dataset_info.json'\n        with open(info_path, 'w', encoding='utf-8') as f:\n            json.dump(dataset_info, f, ensure_ascii=False, indent=2)\n        \n        print(f"✅ Generated {len(dataset)} disaster documents")\n        print(f"   Dataset info saved to: {info_path}")\n        \n        return dataset\n\n\ndef create_disaster_dataset(\n    output_dir: str = "disaster_documents",\n    num_documents: int = 100\n) -> List[Dict[str, Any]]:\n    """災害文書データセットを作成"""\n    generator = DisasterDocumentGenerator(output_dir)\n    dataset = generator.generate_dataset(num_documents)\n    return dataset\n\n\ndef main():\n    """メイン実行関数"""\n    print("="*80)\n    print("Disaster Document Dataset Generator")\n    print("="*80)\n    \n    # データセット生成\n    dataset = create_disaster_dataset(\n        output_dir="data/disaster_documents",\n        num_documents=100\n    )\n    \n    # 統計表示\n    doc_types = {}\n    for doc in dataset:\n        doc_type = doc['metadata']['document_type']\n        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1\n    \n    print(f"\\n📊 Dataset Statistics:")\n    print(f"   Total documents: {len(dataset)}")\n    for doc_type, count in doc_types.items():\n        print(f"   {doc_type}: {count}")\n    \n    print("\\n✅ Disaster document dataset created successfully!")\n\n\nif __name__ == "__main__":\n    main()
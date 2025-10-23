"""
Enhanced Visual Document Processing
強化されたビジュアル文書処理

複雑なレイアウト文書（グラフ、表、スキャン、スクリーンショット）への対応
- 高精度OCR処理
- レイアウト解析
- 表・グラフ検出
- テキスト構造化
- 多言語対応（日本語・英語）
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
import json
import re
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import easyocr
from paddleocr import PaddleOCR


@dataclass
class LayoutElement:
    """レイアウト要素"""
    element_type: str  # text, table, figure, header, footer
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    text_content: str = ""
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TableStructure:
    """表構造"""
    rows: int
    cols: int
    cells: List[List[str]]
    headers: List[str] = None
    bbox: Tuple[int, int, int, int] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """DataFrameに変換"""
        if self.headers and len(self.headers) == self.cols:
            return pd.DataFrame(self.cells, columns=self.headers)
        else:
            return pd.DataFrame(self.cells)


@dataclass
class ProcessingResult:
    """処理結果"""
    image_path: str
    text_content: str
    layout_elements: List[LayoutElement]
    tables: List[TableStructure]
    processing_metadata: Dict
    confidence_scores: Dict


class EnhancedVisualProcessor:
    """
    強化されたビジュアル文書プロセッサ
    
    機能:
    - 複数OCRエンジンの統合
    - レイアウト解析
    - 表・グラフ検出
    - テキスト後処理
    """
    
    def __init__(
        self,
        ocr_engines: List[str] = None,
        layout_model: str = "microsoft/layoutlmv3-base",
        languages: List[str] = None,
        confidence_threshold: float = 0.5
    ):
        self.ocr_engines = ocr_engines or ["tesseract", "easyocr", "paddleocr"]
        self.languages = languages or ["ja", "en"]
        self.confidence_threshold = confidence_threshold
        
        # ロギング設定\n        logging.basicConfig(level=logging.INFO)\n        self.logger = logging.getLogger(__name__)\n        \n        # OCRエンジン初期化\n        self._initialize_ocr_engines()\n        \n        # レイアウトモデル初期化\n        self._initialize_layout_model(layout_model)\n        \n        self.logger.info(f"Enhanced Visual Processor initialized")\n        self.logger.info(f"OCR engines: {self.ocr_engines}")\n        self.logger.info(f"Languages: {self.languages}")\n    \n    def _initialize_ocr_engines(self):\n        """OCRエンジンを初期化"""\n        self.ocr_instances = {}\n        \n        # EasyOCR\n        if "easyocr" in self.ocr_engines:\n            try:\n                self.ocr_instances["easyocr"] = easyocr.Reader(self.languages)\n                self.logger.info("EasyOCR initialized")\n            except Exception as e:\n                self.logger.warning(f"EasyOCR initialization failed: {e}")\n        \n        # PaddleOCR\n        if "paddleocr" in self.ocr_engines:\n            try:\n                self.ocr_instances["paddleocr"] = PaddleOCR(\n                    use_angle_cls=True,\n                    lang='japan' if 'ja' in self.languages else 'en'\n                )\n                self.logger.info("PaddleOCR initialized")\n            except Exception as e:\n                self.logger.warning(f"PaddleOCR initialization failed: {e}")\n    \n    def _initialize_layout_model(self, model_name: str):\n        """レイアウト解析モデルを初期化"""\n        try:\n            self.layout_processor = LayoutLMv3Processor.from_pretrained(model_name)\n            self.layout_model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)\n            self.logger.info(f"LayoutLM model initialized: {model_name}")\n        except Exception as e:\n            self.logger.warning(f"LayoutLM initialization failed: {e}")\n            self.layout_processor = None\n            self.layout_model = None\n    \n    def preprocess_image(self, image: np.ndarray) -> np.ndarray:\n        """画像前処理"""\n        # グレースケール変換\n        if len(image.shape) == 3:\n            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n        else:\n            gray = image.copy()\n        \n        # ノイズ除去\n        denoised = cv2.fastNlMeansDenoising(gray)\n        \n        # コントラスト向上\n        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))\n        enhanced = clahe.apply(denoised)\n        \n        # 二値化（Otsu's method）\n        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n        \n        return binary\n    \n    def extract_text_tesseract(\n        self, \n        image: np.ndarray,\n        config: str = "--psm 6 -l jpn+eng"\n    ) -> Tuple[str, float]:\n        """Tesseractでテキスト抽出"""\n        try:\n            # テキスト抽出\n            text = pytesseract.image_to_string(image, config=config)\n            \n            # 信頼度取得\n            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)\n            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]\n            avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0\n            \n            return text.strip(), avg_confidence\n        except Exception as e:\n            self.logger.error(f"Tesseract OCR failed: {e}")\n            return "", 0.0\n    \n    def extract_text_easyocr(self, image: np.ndarray) -> Tuple[str, float]:\n        """EasyOCRでテキスト抽出"""\n        if "easyocr" not in self.ocr_instances:\n            return "", 0.0\n        \n        try:\n            results = self.ocr_instances["easyocr"].readtext(image)\n            \n            texts = []\n            confidences = []\n            \n            for (bbox, text, confidence) in results:\n                if confidence > self.confidence_threshold:\n                    texts.append(text)\n                    confidences.append(confidence)\n            \n            full_text = " ".join(texts)\n            avg_confidence = np.mean(confidences) if confidences else 0.0\n            \n            return full_text, avg_confidence\n        except Exception as e:\n            self.logger.error(f"EasyOCR failed: {e}")\n            return "", 0.0\n    \n    def extract_text_paddleocr(self, image: np.ndarray) -> Tuple[str, float]:\n        """PaddleOCRでテキスト抽出"""\n        if "paddleocr" not in self.ocr_instances:\n            return "", 0.0\n        \n        try:\n            results = self.ocr_instances["paddleocr"].ocr(image, cls=True)\n            \n            texts = []\n            confidences = []\n            \n            for result in results:\n                if result:\n                    for line in result:\n                        bbox, (text, confidence) = line\n                        if confidence > self.confidence_threshold:\n                            texts.append(text)\n                            confidences.append(confidence)\n            \n            full_text = " ".join(texts)\n            avg_confidence = np.mean(confidences) if confidences else 0.0\n            \n            return full_text, avg_confidence\n        except Exception as e:\n            self.logger.error(f"PaddleOCR failed: {e}")\n            return "", 0.0\n    \n    def extract_text_ensemble(self, image: np.ndarray) -> Tuple[str, float, Dict]:\n        """複数OCRエンジンのアンサンブル"""\n        results = {}\n        \n        # 各OCRエンジンで実行\n        if "tesseract" in self.ocr_engines:\n            text, conf = self.extract_text_tesseract(image)\n            results["tesseract"] = {"text": text, "confidence": conf}\n        \n        if "easyocr" in self.ocr_engines:\n            text, conf = self.extract_text_easyocr(image)\n            results["easyocr"] = {"text": text, "confidence": conf}\n        \n        if "paddleocr" in self.ocr_engines:\n            text, conf = self.extract_text_paddleocr(image)\n            results["paddleocr"] = {"text": text, "confidence": conf}\n        \n        # 最高信頼度の結果を選択\n        best_engine = max(results.keys(), key=lambda k: results[k]["confidence"])\n        best_text = results[best_engine]["text"]\n        best_confidence = results[best_engine]["confidence"]\n        \n        return best_text, best_confidence, results\n    \n    def detect_layout_elements(self, image: np.ndarray) -> List[LayoutElement]:\n        """レイアウト要素を検出"""\n        elements = []\n        \n        # 輪郭検出によるレイアウト解析\n        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n        \n        for contour in contours:\n            x, y, w, h = cv2.boundingRect(contour)\n            area = cv2.contourArea(contour)\n            \n            # 最小サイズフィルタ\n            if area < 1000:\n                continue\n            \n            # アスペクト比による要素タイプ判定\n            aspect_ratio = w / h\n            \n            if aspect_ratio > 3.0:  # 横長 → ヘッダー/フッター\n                element_type = "header" if y < image.shape[0] * 0.2 else "footer"\n            elif aspect_ratio > 1.5:  # やや横長 → 表\n                element_type = "table"\n            elif aspect_ratio < 0.5:  # 縦長 → サイドバー\n                element_type = "sidebar"\n            else:  # 正方形に近い → テキストブロック\n                element_type = "text"\n            \n            # 信頼度（面積に基づく）\n            confidence = min(area / (image.shape[0] * image.shape[1]), 1.0)\n            \n            element = LayoutElement(\n                element_type=element_type,\n                bbox=(x, y, w, h),\n                confidence=confidence,\n                metadata={"area": area, "aspect_ratio": aspect_ratio}\n            )\n            \n            elements.append(element)\n        \n        return elements\n    \n    def detect_tables(self, image: np.ndarray) -> List[TableStructure]:\n        """表を検出"""\n        tables = []\n        \n        # 水平線・垂直線検出\n        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))\n        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))\n        \n        horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)\n        vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)\n        \n        # 表グリッド検出\n        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)\n        \n        # 輪郭検出\n        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n        \n        for contour in contours:\n            x, y, w, h = cv2.boundingRect(contour)\n            area = cv2.contourArea(contour)\n            \n            # 表として十分な大きさかチェック\n            if area > 5000 and w > 100 and h > 100:\n                # 簡易的な行・列数推定\n                rows = max(1, h // 30)  # 推定行数\n                cols = max(1, w // 100)  # 推定列数\n                \n                # セル内容の抽出（簡略版）\n                cells = [["" for _ in range(cols)] for _ in range(rows)]\n                \n                table = TableStructure(\n                    rows=rows,\n                    cols=cols,\n                    cells=cells,\n                    bbox=(x, y, w, h)\n                )\n                \n                tables.append(table)\n        \n        return tables\n    \n    def detect_figures(self, image: np.ndarray) -> List[LayoutElement]:\n        """図・グラフを検出"""\n        figures = []\n        \n        # エッジ検出\n        edges = cv2.Canny(image, 50, 150)\n        \n        # 輪郭検出\n        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n        \n        for contour in contours:\n            x, y, w, h = cv2.boundingRect(contour)\n            area = cv2.contourArea(contour)\n            \n            # 図として十分な大きさかチェック\n            if area > 2000 and w > 50 and h > 50:\n                # 複雑さによる図/グラフ判定\n                hull = cv2.convexHull(contour)\n                hull_area = cv2.contourArea(hull)\n                solidity = area / hull_area if hull_area > 0 else 0\n                \n                element_type = "figure" if solidity > 0.8 else "graph"\n                \n                figure = LayoutElement(\n                    element_type=element_type,\n                    bbox=(x, y, w, h),\n                    confidence=solidity,\n                    metadata={"area": area, "solidity": solidity}\n                )\n                \n                figures.append(figure)\n        \n        return figures\n    \n    def post_process_text(self, text: str) -> str:\n        """テキスト後処理"""\n        # 改行の正規化\n        text = re.sub(r'\\n+', '\\n', text)\n        \n        # 不要な空白の除去\n        text = re.sub(r' +', ' ', text)\n        \n        # 文字化けの修正（基本的なもの）\n        text = text.replace('\\x0c', '')  # フォームフィード文字\n        text = text.replace('\\ufeff', '')  # BOM\n        \n        # 日本語特有の処理\n        if any('\\u3040' <= char <= '\\u309f' or '\\u30a0' <= char <= '\\u30ff' for char in text):\n            # ひらがな・カタカナが含まれる場合\n            text = re.sub(r'(\\w)\\s+(\\w)', r'\\1\\2', text)  # 日本語間の不要な空白除去\n        \n        return text.strip()\n    \n    def process_document(\n        self, \n        image_path: str,\n        output_dir: str = None\n    ) -> ProcessingResult:\n        """文書を包括的に処理"""\n        self.logger.info(f"Processing document: {image_path}")\n        \n        # 画像読み込み\n        image = cv2.imread(image_path)\n        if image is None:\n            raise ValueError(f"Cannot load image: {image_path}")\n        \n        # 前処理\n        processed_image = self.preprocess_image(image)\n        \n        # テキスト抽出（アンサンブル）\n        text, confidence, ocr_results = self.extract_text_ensemble(processed_image)\n        text = self.post_process_text(text)\n        \n        # レイアウト解析\n        layout_elements = self.detect_layout_elements(processed_image)\n        \n        # 表検出\n        tables = self.detect_tables(processed_image)\n        \n        # 図・グラフ検出\n        figures = self.detect_figures(processed_image)\n        layout_elements.extend(figures)\n        \n        # メタデータ作成\n        processing_metadata = {\n            "image_path": image_path,\n            "image_size": image.shape[:2],\n            "preprocessing_applied": True,\n            "ocr_engines_used": list(ocr_results.keys()),\n            "layout_elements_count": len(layout_elements),\n            "tables_count": len(tables)\n        }\n        \n        # 信頼度スコア\n        confidence_scores = {\n            "overall_confidence": confidence,\n            "ocr_results": ocr_results\n        }\n        \n        result = ProcessingResult(\n            image_path=image_path,\n            text_content=text,\n            layout_elements=layout_elements,\n            tables=tables,\n            processing_metadata=processing_metadata,\n            confidence_scores=confidence_scores\n        )\n        \n        # 結果保存（オプション）\n        if output_dir:\n            self.save_processing_result(result, output_dir)\n        \n        self.logger.info(f"Document processing completed")\n        self.logger.info(f"  Text length: {len(text)} characters")\n        self.logger.info(f"  Layout elements: {len(layout_elements)}")\n        self.logger.info(f"  Tables: {len(tables)}")\n        self.logger.info(f"  Confidence: {confidence:.3f}")\n        \n        return result\n    \n    def save_processing_result(self, result: ProcessingResult, output_dir: str):\n        """処理結果を保存"""\n        output_path = Path(output_dir)\n        output_path.mkdir(parents=True, exist_ok=True)\n        \n        # ファイル名作成\n        base_name = Path(result.image_path).stem\n        \n        # JSON形式で保存\n        result_dict = {\n            "image_path": result.image_path,\n            "text_content": result.text_content,\n            "layout_elements": [asdict(elem) for elem in result.layout_elements],\n            "tables": [asdict(table) for table in result.tables],\n            "processing_metadata": result.processing_metadata,\n            "confidence_scores": result.confidence_scores\n        }\n        \n        json_path = output_path / f"{base_name}_processing_result.json"\n        with open(json_path, 'w', encoding='utf-8') as f:\n            json.dump(result_dict, f, ensure_ascii=False, indent=2)\n        \n        # テキストファイルとして保存\n        text_path = output_path / f"{base_name}_extracted_text.txt"\n        with open(text_path, 'w', encoding='utf-8') as f:\n            f.write(result.text_content)\n        \n        self.logger.info(f"Results saved to {output_dir}")\n\n\ndef create_enhanced_processor(\n    ocr_engines: List[str] = None,\n    languages: List[str] = None\n) -> EnhancedVisualProcessor:\n    """強化ビジュアルプロセッサを作成"""\n    processor = EnhancedVisualProcessor(\n        ocr_engines=ocr_engines or ["tesseract", "easyocr"],\n        languages=languages or ["ja", "en"],\n        confidence_threshold=0.5\n    )\n    \n    return processor\n\n\ndef batch_process_documents(\n    input_dir: str,\n    output_dir: str,\n    file_extensions: List[str] = None\n) -> List[ProcessingResult]:\n    """バッチで文書を処理"""\n    if file_extensions is None:\n        file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']\n    \n    processor = create_enhanced_processor()\n    \n    input_path = Path(input_dir)\n    image_files = []\n    \n    for ext in file_extensions:\n        image_files.extend(input_path.glob(f"*{ext}"))\n        image_files.extend(input_path.glob(f"*{ext.upper()}"))\n    \n    results = []\n    for image_file in image_files:\n        try:\n            result = processor.process_document(\n                str(image_file),\n                output_dir=output_dir\n            )\n            results.append(result)\n        except Exception as e:\n            logging.error(f"Failed to process {image_file}: {e}")\n    \n    print(f"✅ Processed {len(results)} documents")\n    return results\n\n\ndef main():\n    """メイン実行関数"""\n    print("="*80)\n    print("Enhanced Visual Document Processing")\n    print("="*80)\n    \n    # プロセッサ作成\n    processor = create_enhanced_processor()\n    \n    print("\\n✅ Enhanced Visual Processor created successfully")\n    print(f"   OCR engines: {processor.ocr_engines}")\n    print(f"   Languages: {processor.languages}")\n\n\nif __name__ == "__main__":\n    main()
"""
Quick Start Demo - Simple Version
Visual RAPTOR ColBERT システムの簡単なクイックスタートデモ
"""

import os
import sys
from pathlib import Path

# パスを追加
sys.path.append(str(Path(__file__).parent))

from integrated_system_simple import create_integrated_system
from langchain_ollama import OllamaEmbeddings, ChatOllama


def check_ollama_connection():
    """Ollamaサーバーの接続をチェック"""
    print("🔧 Checking Ollama connection...")
    
    try:
        embeddings = OllamaEmbeddings(
            model="mxbai-embed-large",
            base_url="http://localhost:11434"
        )
        test_result = embeddings.embed_query("test")
        print("   ✅ Ollama embeddings working")
        return True
    except Exception as e:
        print(f"   ❌ Ollama connection failed: {e}")
        print("   Please ensure:")
        print("   1. Ollama is running: ollama serve")
        print("   2. Model is available: ollama pull mxbai-embed-large")
        return False


def run_simple_demo():
    """簡単なデモを実行"""
    print("="*80)
    print("🚀 Visual RAPTOR ColBERT Quick Start Demo (Simple Version)")
    print("災害文書検索システム クイックスタート（簡易版）")
    print("="*80)
    
    # Ollama接続チェック
    if not check_ollama_connection():
        print("\n❌ Cannot connect to Ollama. Please check the connection.")
        return
    
    # モデル初期化
    print("\n📚 Initializing models...")
    try:
        embeddings = OllamaEmbeddings(
            model="mxbai-embed-large",
            base_url="http://localhost:11434"
        )
        llm = ChatOllama(
            model="granite-code:8b",
            temperature=0,
            base_url="http://localhost:11434",
            timeout=300
        )
        print("   ✅ Models initialized successfully")
    except Exception as e:
        print(f"   ❌ Model initialization failed: {e}")
        return
    
    # 軽量設定でシステム作成
    print("\n🔧 Creating system with lightweight configuration...")
    config = {
        'data_dir': 'data/simple_demo',
        'benchmark_size': 'small',
        'num_queries': 5,     # 非常に少数でテスト
        'num_documents': 10,  # 非常に少数でテスト
        'raptor_config': {
            'min_clusters': 2,
            'max_clusters': 3,
            'max_depth': 2,
            'chunk_size': 300,
            'chunk_overlap': 50
        }
    }
    
    system = create_integrated_system(
        embeddings_model=embeddings,
        llm=llm,
        config=config
    )
    
    # デモ実行
    print("\n🏃 Running demonstration...")
    try:
        results = system.run_simple_demo()
        
        # 結果表示
        print("\n" + "="*80)
        print("📊 Demo Results Summary")
        print("="*80)
        
        if results.get('status') == 'completed':
            if 'setup_stats' in results:
                setup = results['setup_stats']
                print(f"📄 Documents created: {setup.get('num_documents', 0)}")
                print(f"🔍 Queries generated: {setup.get('num_queries', 0)}")
                print(f"📊 Judgments generated: {setup.get('num_judgments', 0)}")
            
            print("\n🎉 Simple demo completed successfully!")
            print("✨ Basic system components are working.")
            print("\n💡 Next steps:")
            print("   - Check the generated data in data/simple_demo/")
            print("   - Try the full system with more documents")
            print("   - Explore the visual document processing features")
        else:
            print(f"❌ Demo failed with error: {results.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"\n❌ Demo execution failed: {e}")
        print("Please check the error messages above.")


def show_system_info():
    """システム情報を表示"""
    print("\n" + "="*80)
    print("📋 System Information")
    print("="*80)
    print("🚀 Visual RAPTOR ColBERT Integration System")
    print("   - 震災教訓継承のためのビジュアル文書検索システム")
    print("   - RAPTOR: 階層的文書検索・要約")
    print("   - ColVBERT: マルチモーダル検索")
    print("   - JinaVDR: Visual Document Retrievalベンチマーク")
    print("   - Enhanced OCR: 高精度日本語テキスト抽出")
    
    print("\n📁 Files created:")
    print("   - visual_raptor_colbert.py: メインシステム")
    print("   - jina_vdr_benchmark.py: ベンチマーク環境")
    print("   - enhanced_visual_processing.py: OCR処理")
    print("   - disaster_dataset_generator.py: データセット生成")
    print("   - integrated_system_simple.py: 統合システム（簡易版）")
    
    print("\n🔧 Requirements:")
    print("   - Python 3.8+")
    print("   - Ollama with mxbai-embed-large and granite-code:8b")
    print("   - pip install -r requirements.txt")


if __name__ == "__main__":
    show_system_info()
    run_simple_demo()
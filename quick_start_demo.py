"""
Quick Start Demo
Visual RAPTOR ColBERT システムのクイックスタートデモ
"""

import os
import sys
from pathlib import Path

# パスを追加
sys.path.append(str(Path(__file__).parent))

from integrated_system import create_integrated_system
from langchain_ollama import OllamaEmbeddings, ChatOllama


def check_prerequisites():
    """前提条件をチェック"""
    print("🔧 Checking prerequisites...")
    
    # Ollamaサーバーの確認
    try:
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        test_result = embeddings.embed_query("test")
        print("   ✅ Ollama embeddings working")
    except Exception as e:
        print(f"   ❌ Ollama embeddings failed: {e}")
        print("   Please ensure Ollama is running: ollama serve")
        return False
    
    try:
        llm = ChatOllama(model="granite-code:8b")
        test_result = llm.invoke("test")
        print("   ✅ Ollama LLM working")
    except Exception as e:
        print(f"   ❌ Ollama LLM failed: {e}")
        print("   Please ensure granite-code:8b model is available: ollama pull granite-code:8b")
        return False
    
    return True


def run_quick_demo():
    """クイックデモを実行"""
    print("="*80)
    print("🚀 Visual RAPTOR ColBERT Quick Start Demo")
    print("災害文書検索システム クイックスタート")
    print("="*80)
    
    # 前提条件チェック
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return
    
    # モデル初期化
    print("\n📚 Initializing models...")
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
    
    # 軽量設定でシステム作成
    print("\n🔧 Creating system with lightweight configuration...")
    config = {
        'data_dir': 'data/quick_demo',
        'benchmark_size': 'small',
        'num_queries': 10,    # 少数でテスト
        'num_documents': 20,  # 少数でテスト
        'raptor_config': {
            'min_clusters': 2,
            'max_clusters': 3,
            'max_depth': 2,     # 浅めのツリー
            'chunk_size': 300,
            'chunk_overlap': 50
        }
    }
    
    system = create_integrated_system(
        embeddings_model=embeddings,
        llm=llm,
        config=config
    )
    
    # パイプライン実行
    print("\n🏃 Running demonstration pipeline...")
    try:
        results = system.run_complete_pipeline()
        
        # 結果表示
        print("\n" + "="*80)
        print("📊 Demo Results Summary")
        print("="*80)
        
        if 'setup_stats' in results:
            setup = results['setup_stats']
            print(f"📄 Documents created: {setup.get('num_documents', 0)}")
            print(f"🔍 Queries generated: {setup.get('num_queries', 0)}")
        
        if 'processing_stats' in results:
            proc = results['processing_stats']
            print(f"⚡ Processing time: {proc.get('avg_processing_time', 0):.2f}s per doc")
        
        if 'index_stats' in results:
            idx = results['index_stats']
            print(f"🌲 Tree nodes: {idx.get('tree_nodes', 0)}")
            print(f"📊 Tree depth: {idx.get('tree_depth', 0)}")
        
        if 'evaluation_metrics' in results:
            metrics = results['evaluation_metrics']
            print(f"🎯 Precision: {metrics.get('precision', 0):.4f}")
            print(f"📈 Recall: {metrics.get('recall', 0):.4f}")
            print(f"⭐ F1 Score: {metrics.get('f1', 0):.4f}")
            print(f"🔢 NDCG: {metrics.get('ndcg', 0):.4f}")
        
        print("\n🎉 Quick demo completed successfully!")
        print("✨ You can now explore the full system capabilities.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Please check the error messages above and try again.")


def show_next_steps():
    """次のステップを表示"""
    print("\n" + "="*80)
    print("📚 Next Steps")
    print("="*80)
    print("1. 📖 Read the full README.md for detailed documentation")
    print("2. 🔧 Customize the configuration in integrated_system.py")
    print("3. 📊 Run with larger datasets for better evaluation")
    print("4. 🖼️ Add your own disaster documents for processing")
    print("5. 🚀 Deploy the system for production use")
    
    print("\n📁 Generated files location:")
    print("   - data/quick_demo/")
    print("   - disaster_documents/")
    print("   - results/")


if __name__ == "__main__":
    run_quick_demo()
    show_next_steps()
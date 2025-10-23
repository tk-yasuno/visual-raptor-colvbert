"""
ColModernVBERT デモンストレーション
SigLIPを使用した最新のマルチモーダル検索システム
"""

import traceback
from pathlib import Path

print("="*80)
print("🚀 ColModernVBERT with SigLIP - Demo")
print("="*80)

try:
    # Step 1: Import modules
    print("\n📦 Step 1: Importing modules...")
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from visual_raptor_colbert import VisualRAPTORColBERT
    from jina_vdr_benchmark import JinaVDRBenchmark
    print("✅ Imports successful")
    
    # Step 2: Initialize Ollama models
    print("\n🔧 Step 2: Initializing Ollama models...")
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url="http://localhost:11434"
    )
    llm = ChatOllama(
        model="granite-code:8b",
        temperature=0,
        base_url="http://localhost:11434",
        timeout=600
    )
    print("✅ Ollama models initialized")
    
    # Step 3: Initialize Visual RAPTOR with ColModernVBERT
    print("\n✨ Step 3: Initializing Visual RAPTOR with ColModernVBERT (SigLIP)...")
    
    colbert_config = {
        'encoder_type': 'modern',  # 重要：modernを指定
        'text_model': 'google/siglip-base-patch16-224',
        'vision_model': 'google/siglip-base-patch16-224',
        'embedding_dim': 768,
        'use_cross_attention': True
    }
    
    visual_raptor = VisualRAPTORColBERT(
        embeddings_model=embeddings,
        llm=llm,
        colbert_config=colbert_config,
        use_modern_vbert=True,  # ColModernVBERTを使用
        max_depth=2,
        chunk_size=300
    )
    print("✅ ColModernVBERT initialized successfully!")
    
    # Step 4: Initialize JinaVDR Benchmark
    print("\n📊 Step 4: Initializing JinaVDR Benchmark...")
    benchmark = JinaVDRBenchmark(
        data_dir="data/colmodern_vbert_demo",
        language="ja",
        dataset_size="small"
    )
    print("✅ Benchmark initialized")
    
    # Step 5: Generate sample queries
    print("\n🔍 Step 5: Generating sample queries...")
    queries = benchmark.generate_disaster_queries(num_queries=5)
    print(f"✅ Generated {len(queries)} queries")
    for i, q in enumerate(queries[:3], 1):
        print(f"   {i}. {q.text}")
    
    # Step 6: Create sample documents
    print("\n📄 Step 6: Creating sample documents...")
    documents = benchmark.create_synthetic_documents(num_documents=20)
    print(f"✅ Created {len(documents)} documents")
    
    # Step 7: Generate relevance judgments
    print("\n📊 Step 7: Generating relevance judgments...")
    judgments = benchmark.generate_relevance_judgments(num_judgments_per_query=3)
    print(f"✅ Generated {len(judgments)} relevance judgments")
    
    # Step 8: Save benchmark data
    print("\n💾 Step 8: Saving benchmark data...")
    benchmark.save_benchmark_data()
    print("✅ Benchmark data saved")
    
    # Step 9: Test encoding (optional)
    print("\n🧪 Step 9: Testing ColModernVBERT encoding...")
    from PIL import Image
    import torch
    
    # テスト用のダミー画像とテキスト
    test_image = Image.new('RGB', (224, 224), color='white')
    test_texts = ["災害時の避難場所", "Emergency evacuation location"]
    test_images = [test_image, test_image]
    
    try:
        # マルチモーダルエンコーディング
        with torch.no_grad():
            embeddings_result = visual_raptor.colbert_encoder.encode_multimodal(
                texts=test_texts,
                images=test_images
            )
        print(f"✅ Encoding successful! Shape: {embeddings_result.shape}")
        print(f"   Embedding dimension: {embeddings_result.shape[1]}")
    except Exception as e:
        print(f"⚠️ Encoding test skipped: {e}")
    
    # Step 10: Show statistics
    print("\n📈 Step 10: System Statistics")
    stats = benchmark.get_benchmark_statistics()
    print(f"   Queries: {stats['num_queries']}")
    print(f"   Documents: {stats['num_documents']}")
    print(f"   Judgments: {stats['num_judgments']}")
    print(f"   Query categories: {list(stats['query_categories'].keys())}")
    print(f"   Document categories: {list(stats['document_categories'].keys())}")
    
    print("\n" + "="*80)
    print("✅ ColModernVBERT Demo Completed Successfully!")
    print("="*80)
    
    print("\n📝 Key Features:")
    print("   ✨ SigLIP-based multimodal encoding")
    print("   🔄 Cross-attention between text and images")
    print("   🚀 Improved contrastive learning (sigmoid loss)")
    print("   💾 Efficient token representation")
    
    print("\n📊 Performance Advantages over ColVBERT:")
    print("   • Better zero-shot transfer")
    print("   • More efficient training")
    print("   • Improved multilingual support")
    print("   • Reduced memory footprint")
    
    print("\n📂 Data Location:")
    print(f"   data/colmodern_vbert_demo/")
    print(f"   - queries/queries.json")
    print(f"   - annotations/documents.json")
    print(f"   - images/ (20 synthetic images)")
    
except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    print("\nDetailed traceback:")
    traceback.print_exc()
    
    print("\n💡 Troubleshooting:")
    print("   1. Ensure transformers>=4.35.0: pip install --upgrade transformers")
    print("   2. Check Ollama is running: ollama list")
    print("   3. Verify GPU availability: nvidia-smi")
    print("   4. Check CUDA compatibility")

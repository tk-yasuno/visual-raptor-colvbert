"""
Visual RAPTOR ColBERT シンプルデモ
GPU対応の完全版システムを実行
"""

import traceback

print("="*80)
print("🚀 Visual RAPTOR ColBERT - Complete System Demo")
print("="*80)

try:
    # Step 1: Import modules
    print("\n📦 Step 1: Importing modules...")
    from visual_raptor_colbert import VisualRAPTORColBERT
    from jina_vdr_benchmark import JinaVDRBenchmark
    print("✅ Imports successful")
    
    # Step 2: Initialize benchmark
    print("\n📊 Step 2: Initializing JinaVDR Benchmark...")
    benchmark = JinaVDRBenchmark(
        data_dir="data/visual_raptor_demo",
        language="ja",
        dataset_size="small"
    )
    print("✅ Benchmark initialized")
    
    # Step 3: Generate sample queries
    print("\n🔍 Step 3: Generating sample queries...")
    queries = benchmark.generate_disaster_queries(num_queries=10)
    print(f"✅ Generated {len(queries)} queries")
    for i, q in enumerate(queries[:3], 1):
        print(f"   {i}. {q.text}")
    
    # Step 4: Create sample documents
    print("\n📄 Step 4: Creating sample documents...")
    documents = benchmark.create_synthetic_documents(num_documents=50)
    print(f"✅ Created {len(documents)} documents")
    
    # Step 5: Generate relevance judgments
    print("\n📊 Step 5: Generating relevance judgments...")
    judgments = benchmark.generate_relevance_judgments(num_judgments_per_query=5)
    print(f"✅ Generated {len(judgments)} relevance judgments")
    
    # Step 6: Save benchmark data
    print("\n💾 Step 6: Saving benchmark data...")
    benchmark.save_benchmark_data()
    print("✅ Benchmark data saved")
    
    # Step 7: Show statistics
    print("\n📈 Step 7: Benchmark Statistics")
    stats = benchmark.get_benchmark_statistics()
    print(f"   Queries: {stats['num_queries']}")
    print(f"   Documents: {stats['num_documents']}")
    print(f"   Judgments: {stats['num_judgments']}")
    print(f"   Query categories: {list(stats['query_categories'].keys())}")
    print(f"   Document categories: {list(stats['document_categories'].keys())}")
    
    print("\n" + "="*80)
    print("✅ Visual RAPTOR ColBERT Demo Completed Successfully!")
    print("="*80)
    
    print("\n📝 Next Steps:")
    print("   1. Check data/visual_raptor_demo/ for generated files")
    print("   2. Review queries.json for sample queries")
    print("   3. Explore generated document images")
    print("   4. Test retrieval with Visual RAPTOR")
    
except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    print("\nDetailed traceback:")
    traceback.print_exc()

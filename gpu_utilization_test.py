#!/usr/bin/env python3
"""
Quick GPU Utilization Test
GPUの稼働を確認するクイックテスト
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from datetime import datetime

def gpu_warmup_test():
    """GPU稼働確認用ウォームアップテスト"""
    print("🔥 GPU Warmup Test")
    print("="*50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    device = torch.device('cuda')
    
    # GPU情報表示
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # GPUウォームアップ
    print("\n🚀 Starting GPU warmup...")
    
    # 大きな行列を作成してGPU上で計算
    sizes = [1000, 2000, 4000, 8000]
    
    for size in sizes:
        print(f"   Matrix size: {size}x{size}")
        
        # ランダム行列をGPU上に作成
        start_time = time.time()
        a = torch.randn(size, size, device=device, dtype=torch.float32)
        b = torch.randn(size, size, device=device, dtype=torch.float32)
        
        # 行列積計算（GPU集約的）
        c = torch.mm(a, b)
        
        # 追加計算でGPU使用率を上げる
        for _ in range(5):
            c = torch.mm(c, a)
            c = F.relu(c)
            c = c / c.norm()
        
        torch.cuda.synchronize()  # GPU同期
        elapsed = time.time() - start_time
        
        # メモリ使用量確認
        memory_used = torch.cuda.memory_allocated() / 1e9
        
        print(f"      Time: {elapsed:.3f}s, GPU Memory: {memory_used:.2f} GB")
    
    print("✅ GPU warmup completed")
    return True

def gpu_embedding_simulation():
    """埋め込み処理のGPUシミュレーション"""
    print("\n🧠 GPU Embedding Simulation")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 大規模埋め込みシミュレーション
    vocab_size = 50000
    embedding_dim = 1024
    batch_size = 128
    num_batches = 20
    
    print(f"Vocab size: {vocab_size}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Batch size: {batch_size}")
    
    # 埋め込み層をGPU上に作成
    embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim).to(device)
    
    print(f"\n🚀 Processing {num_batches} batches...")
    
    total_time = 0
    
    for batch_idx in range(num_batches):
        start_time = time.time()
        
        # ランダムなトークンIDを生成
        token_ids = torch.randint(0, vocab_size, (batch_size, 512), device=device)
        
        # 埋め込み計算
        embeddings = embedding_layer(token_ids)
        
        # 追加処理でGPU使用率を上げる
        # Attention-like computation
        query = embeddings
        key = embeddings
        value = embeddings
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, value)
        
        # Layer normalization
        layer_norm = torch.nn.LayerNorm(embedding_dim).to(device)
        output = layer_norm(attention_output + embeddings)
        
        # Feed forward network
        ffn = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 4 * embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * embedding_dim, embedding_dim)
        ).to(device)
        
        final_output = ffn(output)
        
        torch.cuda.synchronize()
        batch_time = time.time() - start_time
        total_time += batch_time
        
        memory_used = torch.cuda.memory_allocated() / 1e9
        
        print(f"   Batch {batch_idx+1:2d}: {batch_time:.3f}s, Memory: {memory_used:.2f} GB")
    
    print(f"\n✅ Embedding simulation completed")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Avg batch time: {total_time/num_batches:.3f}s")
    print(f"   Throughput: {batch_size * num_batches / total_time:.2f} samples/sec")

def gpu_search_simulation():
    """検索処理のGPUシミュレーション"""
    print("\n🔍 GPU Search Simulation")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 大規模検索シミュレーション
    num_documents = 10000
    embedding_dim = 1024
    num_queries = 100
    top_k = 10
    
    print(f"Documents: {num_documents}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Queries: {num_queries}")
    
    # 文書埋め込みを生成
    print("\n📚 Generating document embeddings...")
    doc_embeddings = torch.randn(num_documents, embedding_dim, device=device, dtype=torch.float32)
    doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
    
    print(f"✅ Document embeddings ready: {doc_embeddings.shape}")
    
    # 検索実行
    print(f"\n🚀 Running {num_queries} search queries...")
    
    search_times = []
    
    for query_idx in range(num_queries):
        start_time = time.time()
        
        # クエリ埋め込み生成
        query_embedding = torch.randn(1, embedding_dim, device=device, dtype=torch.float32)
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        
        # 類似度計算（GPU上で並列実行）
        similarities = torch.mm(query_embedding, doc_embeddings.t()).squeeze(0)
        
        # Top-K取得
        top_scores, top_indices = torch.topk(similarities, k=top_k)
        
        torch.cuda.synchronize()
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        if (query_idx + 1) % 20 == 0:
            avg_time = np.mean(search_times[-20:])
            memory_used = torch.cuda.memory_allocated() / 1e9
            print(f"   Query {query_idx+1:3d}: {search_time:.4f}s (avg: {avg_time:.4f}s), Memory: {memory_used:.2f} GB")
    
    print(f"\n✅ Search simulation completed")
    print(f"   Total queries: {num_queries}")
    print(f"   Avg search time: {np.mean(search_times):.4f}s")
    print(f"   Search throughput: {num_queries / np.sum(search_times):.2f} queries/sec")
    print(f"   GPU memory peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

def monitor_gpu_usage():
    """GPU使用状況をモニタリング"""
    print("\n📊 GPU Usage Monitoring")
    print("="*50)
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(', ')
            
            print(f"GPU Utilization: {gpu_util}%")
            print(f"Memory Used: {mem_used} MB / {mem_total} MB ({int(mem_used)/int(mem_total)*100:.1f}%)")
            print(f"Temperature: {temp}°C")
            
            if int(gpu_util) > 50:
                print("✅ GPU is actively utilized!")
            else:
                print("⚠️ GPU utilization is low")
        else:
            print("❌ Failed to get GPU status")
            
    except Exception as e:
        print(f"❌ Error monitoring GPU: {e}")

def main():
    """メイン実行関数"""
    print("="*80)
    print("🚀 GPU Utilization Test Suite")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Cannot run GPU tests.")
        return False
    
    try:
        # 1. GPU情報とウォームアップ
        gpu_warmup_test()
        
        # 2. 埋め込み処理シミュレーション
        gpu_embedding_simulation()
        
        # 3. 検索処理シミュレーション
        gpu_search_simulation()
        
        # 4. GPU使用状況確認
        monitor_gpu_usage()
        
        print("\n" + "="*80)
        print("🎉 GPU Test Suite Completed!")
        print("="*80)
        print("Your RTX 4060 Ti should now be actively utilized.")
        print("Check nvidia-smi to confirm GPU usage.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
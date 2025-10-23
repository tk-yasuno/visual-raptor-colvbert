#!/usr/bin/env python3
"""
Immediate GPU Activation Script
GPUを即座に稼働させるスクリプト
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import threading
from datetime import datetime

def activate_gpu_immediately():
    """GPUを即座に稼働させる"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    device = torch.device('cuda')
    print(f"🚀 Activating GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # GPU稼働確認用の重い処理を実行
    print("\n🔥 Starting intensive GPU workload...")
    
    # 大きな行列を作成してGPU上で計算
    start_time = time.time()
    
    # メモリを大量に使用する処理
    large_tensors = []
    for i in range(10):
        size = 4000 + i * 500  # サイズを徐々に増やす
        print(f"   Creating {size}x{size} tensor on GPU...")
        
        tensor = torch.randn(size, size, device=device, dtype=torch.float32)
        
        # 集約的な計算
        for _ in range(5):
            tensor = torch.mm(tensor, tensor.t())
            tensor = F.relu(tensor)
            tensor = tensor / (tensor.norm() + 1e-8)
        
        large_tensors.append(tensor)
        
        # GPU使用状況表示
        memory_used = torch.cuda.memory_allocated() / 1e9
        print(f"      GPU Memory used: {memory_used:.2f} GB")
        
        # 短時間待機してGPU使用率を維持
        time.sleep(0.1)
    
    # さらに集約的な処理
    print("\n🧮 Running matrix multiplication chain...")
    result = large_tensors[0]
    
    for i, tensor in enumerate(large_tensors[1:], 1):
        print(f"   Chain operation {i}...")
        # 大きな行列の乗算
        result = torch.mm(result[:2000, :2000], tensor[:2000, :2000])
        result = F.tanh(result)
        
        # GPU同期
        torch.cuda.synchronize()
        
        memory_used = torch.cuda.memory_allocated() / 1e9
        print(f"      Memory: {memory_used:.2f} GB")
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ GPU activation completed in {elapsed:.2f}s")
    print(f"🎯 Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    return True

def continuous_gpu_load():
    """継続的なGPU負荷を生成"""
    device = torch.device('cuda')
    
    print("\n🔄 Starting continuous GPU load...")
    
    # 継続的な処理用のモデル
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1024),
    ).to(device)
    
    batch_count = 0
    
    try:
        while True:
            # ランダムバッチ生成
            batch_size = np.random.randint(256, 1024)
            x = torch.randn(batch_size, 1024, device=device)
            
            # 前向き計算
            for _ in range(10):  # 複数回実行で負荷増大
                y = model(x)
                x = F.normalize(y, p=2, dim=1)
            
            batch_count += 1
            
            # 進捗表示
            if batch_count % 50 == 0:
                memory_used = torch.cuda.memory_allocated() / 1e9
                print(f"   Processed {batch_count} batches, Memory: {memory_used:.2f} GB")
            
            # 短時間待機（完全に連続的ではなく、制御可能に）
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n🛑 Continuous load stopped by user")

def monitor_gpu_status():
    """GPU状況をモニタリング"""
    import subprocess
    
    print("\n📊 GPU Status Monitor started")
    
    try:
        while True:
            try:
                result = subprocess.run([
                    'nvidia-smi', 
                    '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    values = result.stdout.strip().split(', ')
                    if len(values) >= 4:
                        gpu_util = values[0]
                        mem_used = values[1]
                        mem_total = values[2]
                        temp = values[3]
                        power = values[4] if len(values) > 4 else "N/A"
                        
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        print(f"🎯 [{timestamp}] GPU: {gpu_util}%, Memory: {mem_used}/{mem_total}MB, Temp: {temp}°C, Power: {power}W")
                        
                        # GPU使用率が高い場合に通知
                        if int(gpu_util) > 80:
                            print("🔥 HIGH GPU UTILIZATION DETECTED!")
                
                time.sleep(5)  # 5秒間隔
                
            except Exception as e:
                print(f"Error in monitoring: {e}")
                time.sleep(5)
                
    except KeyboardInterrupt:
        print("\n🛑 GPU monitoring stopped")

def main():
    """メイン実行"""
    print("="*80)
    print("🚀 GPU Immediate Activation Script")
    print("RTX 4060 Ti GPU稼働強制実行")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot activate GPU.")
        return
    
    # 1. 即座にGPUを稼働させる
    if activate_gpu_immediately():
        print("\n✅ GPU is now active!")
        
        # 2. 継続的な負荷をバックグラウンドで開始
        monitor_thread = threading.Thread(target=monitor_gpu_status, daemon=True)
        monitor_thread.start()
        
        print("\n🔄 Starting continuous GPU workload...")
        print("Press Ctrl+C to stop")
        
        # 3. 継続的なGPU負荷
        continuous_gpu_load()
    
    print("\n👋 GPU activation script finished")

if __name__ == "__main__":
    main()
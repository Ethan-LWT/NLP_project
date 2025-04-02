"""
Utility functions for GPU acceleration in the NLP project.
This module provides tools for checking GPU availability, memory usage,
and performance comparison between CPU and GPU implementations.
"""

import os
import time
import numpy as np
import sys

# Try to import PyTorch for GPU support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def check_gpu_availability():
    """Check if GPU is available and return information about it"""
    if not TORCH_AVAILABLE:
        return {
            'available': False,
            'message': "PyTorch not installed. Install with: pip install torch torchvision"
        }
    
    if torch.cuda.is_available():
        return {
            'available': True,
            'device_count': torch.cuda.device_count(),
            'device_name': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda,
            'message': f"GPU available: {torch.cuda.get_device_name(0)}"
        }
    else:
        return {
            'available': False,
            'message': "CUDA not available. Check NVIDIA drivers and PyTorch installation."
        }

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB"""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return None
    
    try:
        # Get memory usage information
        allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(0) / 1024**2    # MB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'total_mb': total,
            'free_mb': total - allocated,
            'utilization_percent': (allocated / total) * 100
        }
    except Exception as e:
        return {'error': str(e)}

def compare_cpu_gpu_performance(matrix_size=1000, iterations=10):
    """Compare performance of CPU vs GPU for matrix multiplication"""
    if not TORCH_AVAILABLE:
        return "PyTorch not available. Cannot compare performance."
    
    # Generate random matrices
    A_np = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    B_np = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    
    # CPU timing
    start_time = time.time()
    for _ in range(iterations):
        result_np = np.dot(A_np, B_np)
    cpu_time = time.time() - start_time
    
    # Check if GPU is available
    if not torch.cuda.is_available():
        return f"CPU time: {cpu_time:.4f}s (GPU not available for comparison)"
    
    # GPU timing
    A_torch = torch.tensor(A_np, device='cuda')
    B_torch = torch.tensor(B_np, device='cuda')
    
    # Warm-up
    _ = torch.matmul(A_torch, B_torch)
    torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(iterations):
        result_torch = torch.matmul(A_torch, B_torch)
        torch.cuda.synchronize()  # Wait for GPU to finish
    gpu_time = time.time() - start_time
    
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
    
    return {
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'speedup': speedup,
        'matrix_size': matrix_size,
        'iterations': iterations,
        'message': f"Matrix multiplication ({matrix_size}x{matrix_size}, {iterations} iterations):\n"
                  f"CPU: {cpu_time:.4f}s, GPU: {gpu_time:.4f}s, Speedup: {speedup:.2f}x"
    }

def benchmark_embedding_performance(vocab_size=10000, embedding_dim=100, batch_size=32, seq_length=100, iterations=10):
    """Benchmark embedding layer performance on CPU vs GPU"""
    if not TORCH_AVAILABLE:
        return "PyTorch not available. Cannot benchmark embeddings."
    
    # Create random word indices
    indices_np = np.random.randint(0, vocab_size, (batch_size, seq_length)).astype(np.int64)
    
    # CPU embedding
    weights_np = np.random.randn(embedding_dim, vocab_size).astype(np.float32) * 0.01
    
    start_time = time.time()
    for _ in range(iterations):
        # Simple embedding lookup
        embedded_np = np.zeros((embedding_dim * seq_length, batch_size), dtype=np.float32)
        for i in range(batch_size):
            for j in range(seq_length):
                word_idx = indices_np[i, j]
                embedded_np[j*embedding_dim:(j+1)*embedding_dim, i] = weights_np[:, word_idx]
    cpu_time = time.time() - start_time
    
    # Check if GPU is available
    if not torch.cuda.is_available():
        return f"CPU embedding time: {cpu_time:.4f}s (GPU not available for comparison)"
    
    # GPU embedding
    indices_torch = torch.tensor(indices_np, device='cuda')
    weights_torch = torch.tensor(weights_np, device='cuda')
    
    # Warm-up
    embedded_torch = torch.zeros((embedding_dim * seq_length, batch_size), device='cuda')
    for i in range(batch_size):
        for j in range(seq_length):
            word_idx = indices_torch[i, j]
            embedded_torch[j*embedding_dim:(j+1)*embedding_dim, i] = weights_torch[:, word_idx]
    torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(iterations):
        embedded_torch = torch.zeros((embedding_dim * seq_length, batch_size), device='cuda')
        for i in range(batch_size):
            for j in range(seq_length):
                word_idx = indices_torch[i, j]
                embedded_torch[j*embedding_dim:(j+1)*embedding_dim, i] = weights_torch[:, word_idx]
        torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
    
    return {
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'speedup': speedup,
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'iterations': iterations,
        'message': f"Embedding operation (vocab={vocab_size}, dim={embedding_dim}, batch={batch_size}, {iterations} iterations):\n"
                  f"CPU: {cpu_time:.4f}s, GPU: {gpu_time:.4f}s, Speedup: {speedup:.2f}x"
    }

if __name__ == "__main__":
    # Print GPU information when run directly
    gpu_info = check_gpu_availability()
    print("GPU Information:")
    for key, value in gpu_info.items():
        if key != 'message':
            print(f"  {key}: {value}")
    print("\n" + gpu_info['message'])
    
    if gpu_info['available']:
        # Print memory usage
        mem_info = get_gpu_memory_usage()
        print("\nGPU Memory Usage:")
        print(f"  Total: {mem_info['total_mb']:.1f} MB")
        print(f"  Used: {mem_info['allocated_mb']:.1f} MB")
        print(f"  Free: {mem_info['free_mb']:.1f} MB")
        print(f"  Utilization: {mem_info['utilization_percent']:.1f}%")
        
        # Run performance comparisons
        print("\nPerformance Comparison:")
        matrix_result = compare_cpu_gpu_performance(matrix_size=1000, iterations=5)
        print(matrix_result['message'])
        
        embed_result = benchmark_embedding_performance(iterations=5)
        print("\n" + embed_result['message'])
    
    print("\nNote: Install PyTorch with CUDA support for best performance:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

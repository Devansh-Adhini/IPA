import torch
import gc
import sys

def clear_cuda():
    """ Forcefully clear CUDA cache and collect garbage. """
    print("[CUDA Clean] Collecting garbage...")
    gc.collect()
    if torch.cuda.is_available():
        print("[CUDA Clean] Emptying CUDA cache...")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # Reset peak stats
        torch.cuda.reset_peak_memory_stats()
        
        mem_free, mem_total = torch.cuda.mem_get_info()
        print(f"[CUDA Clean] Result: {mem_free / 1024**2:.1f} MB free / {mem_total / 1024**2:.1f} MB total")
    else:
        print("[CUDA Clean] CUDA not available.")

if __name__ == "__main__":
    clear_cuda()
    print("[CUDA Clean] Done.")

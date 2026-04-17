import torch
import cv2
import numpy as np
import os
import argparse
import tqdm
import sys
import re
import csv
import gc

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from test_accuracy import detect_architecture, evaluate_accuracy

def get_iteration(filename):
    """Extracts iteration number from filename (e.g. model_ts-fl_072000.pth -> 72000)."""
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 9999999 # Push weird names to the end

def benchmark_logdir(log_dir, dataset_dir, limit=50, start_iter=0):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking models in {log_dir} on {dataset_dir} (limit: {limit})...")

    # 1. Discover and sort models
    models = [f for f in os.listdir(log_dir) if f.endswith('.pth')]
    models.sort(key=get_iteration)
    
    if not models:
        print(f"No .pth models found in {log_dir}")
        return

    results_file = os.path.join(log_dir, "benchmark_report.csv")
    
    # 2. Check for resume
    completed_models = set()
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None) # skip header
            for row in reader:
                if row: completed_models.add(row[0])

    print(f"Found {len(models)} models. {len(completed_models)} already processed.")

    # 3. Iterate and evaluate
    header = ["Model", "Iteration", "Accuracy (%)"]
    
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    print("\n" + "="*60)
    print(f"{'ITERATION':<15} | {'MODEL':<30} | {'ACCURACY':<10}")
    print("-"*60)

    for model_name in models:
        iteration = get_iteration(model_name)
        if iteration < start_iter:
            continue
            
        if model_name in completed_models:
            continue
            
        model_path = os.path.join(log_dir, model_name)
        iteration = get_iteration(model_name)
        
        try:
            # Silence the "Detected Architecture" print for the loop
            # save original stdout
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            # Initialize model
            net = detect_architecture(model_path, dev)
            
            # Evaluate
            acc = evaluate_accuracy(net, dataset_dir, limit=limit)
            
            # cleanup
            del net
            torch.cuda.empty_cache()
            
            # restore stdout
            sys.stdout.close()
            sys.stdout = original_stdout
            
            # Save and Log
            with open(results_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([model_name, iteration, f"{acc:.2f}"])
            
            print(f"{iteration:<15} | {model_name:<30} | {acc:>8.2f} %")
            
        except Exception as e:
            sys.stdout = original_stdout
            print(f"Failed to evaluate {model_name}: {e}")
            continue
        finally:
            gc.collect()

    print("="*60 + "\n")
    print(f"Benchmark complete. Results saved to {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk benchmark DALF models")
    parser.add_argument("--log_dir", type=str, default="./logdir", help="Directory with .pth models")
    parser.add_argument("--dataset", type=str, required=True, help="Path to image directory")
    parser.add_argument("--limit", type=int, default=50, help="Images per model")
    parser.add_argument("--start_iter", type=int, default=0, help="Iteration to start from")
    args = parser.parse_args()
    
    benchmark_logdir(args.log_dir, args.dataset, args.limit, args.start_iter)

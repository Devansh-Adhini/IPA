import os
import re
import argparse
from evaluate_metrics import evaluate
import tqdm

def benchmark(logdir, datapath, batches):
    print(f"Scanning {logdir} for models...")
    
    # regex to match model_MODE_ITERATION... filenames
    # Examples: 
    # model_ts-fl_000000.pth
    # model_ts-fl_119997_interrupted.pth
    # model_end2end-backbone_final.pth
    pattern = re.compile(r"model_([a-zA-Z0-9-]+)_(\d+|final)(?:_.*)?\.pth")
    
    model_files = []
    for f in os.listdir(logdir):
        match = pattern.match(f)
        if match:
            mode = match.group(1)
            iteration = match.group(2)
            # handle 'final' as a very large number for sorting
            sort_val = float('inf') if iteration == 'final' else int(iteration)
            model_files.append({
                "name": f,
                "path": os.path.join(logdir, f),
                "mode": mode,
                "iteration": iteration,
                "sort_val": sort_val
            })
            
    # Sort by iteration
    model_files.sort(key=lambda x: x["sort_val"])
    
    if not model_files:
        print("No models found in logdir!")
        return

    print(f"Found {len(model_files)} models. Starting benchmark (this may take a while)...")
    
    results = []
    
    # Progress bar for the whole benchmark
    for m in tqdm.tqdm(model_files):
        try:
            res = evaluate(m["path"], datapath, m["mode"], batches, verbose=False)
            res["name"] = m["name"]
            res["iteration"] = m["iteration"]
            results.append(res)
        except Exception as e:
            print(f"Error evaluating {m['name']}: {e}")

    # Print summary table
    print("\n" + "="*100)
    print(f"{'Model Name':<40} | {'Acc (%)':<8} | {'HardL':<8} | {'APL':<8} | {'SSIM':<8} | {'LogP':<8}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['name']:<40} | {r['acc']*100:>7.2f}% | {r['hard_loss']:>8.4f} | {r['ap_loss']:>8.4f} | {r['ssim']:>8.4f} | {r['logprob']:>8.4f}")
    
    print("="*100 + "\n")

    # Suggest the best model based on accuracy
    if results:
        best_acc = max(results, key=lambda x: x["acc"])
        print(f"Best model by Accuracy: {best_acc['name']} ({best_acc['acc']*100:.2f}%)")
        
        # Suggest based on health (low ssim loss usually means better geometry)
        best_health = min(results, key=lambda x: x["ssim"])
        print(f"Best model by Health (SSIM): {best_health['name']} ({best_health['ssim']:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark all models in a directory")
    parser.add_argument("--logdir", type=str, default="./logdir", help="Directory containing .pth files")
    parser.add_argument("--datapath", type=str, default="./dataset/*/images/*.jpg", help="Path to evaluation dataset")
    parser.add_argument("--batches", type=int, default=5, help="Number of batches per model (default: 5 for speed)")
    args = parser.parse_args()
    
    benchmark(args.logdir, args.datapath, args.batches)

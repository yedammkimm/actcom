"""Parse Qwen Standard 500-step result + display grad_norm baseline."""
import json, sys, glob

# Latest standard result
files = sorted(glob.glob('/home/yedam/HMA/HMA_Project/results/qwen3b_k_sweep/accuracy__standard*.json'))
if not files:
    print("No Standard result yet.")
    sys.exit(0)
p = files[-1]
d = json.load(open(p))
r = d['results']
print(f"=== Qwen Standard 500-step baseline ({p.split('/')[-1]}) ===")
print(f"  gsm8k_accuracy       : {r.get('gsm8k_accuracy', 'n/a')}%")
print(f"  n_correct            : {r.get('n_correct')}/{r.get('n_eval_samples')}")
fl = r.get('final_loss')
print(f"  final_loss           : {fl:.4f}" if fl is not None else "  final_loss           : n/a")
print(f"  n_nonfinite_grad_steps: {r.get('n_nonfinite_grad_steps', 'MISSING')}")
sl = r.get('step_losses', [])
if sl:
    stride = max(1, len(sl)//20)
    print(f"  loss (every {stride}): ", ", ".join(f"{sl[i]:.3f}" for i in range(0, len(sl), stride)))
gn = r.get('grad_norm_trace', [])
if gn:
    print(f"\n  grad_norm_trace ({len(gn)} samples):")
    print(f"    {'step':>6}  {'norm':>12}  {'finite':>6}")
    for entry in gn:
        step, val, finite = entry
        print(f"    {step:>6d}  {val:>12.4g}  {finite!s:>6}")
    # Summary stats on finite values
    finite_norms = [v for _, v, f in gn if f]
    if finite_norms:
        import statistics
        print(f"\n  finite grad_norm summary:")
        print(f"    min    = {min(finite_norms):.4g}")
        print(f"    max    = {max(finite_norms):.4g}")
        print(f"    median = {statistics.median(finite_norms):.4g}")
        print(f"    mean   = {statistics.mean(finite_norms):.4g}")
print(f"\n  train_wall_s  : {d['throughput'].get('train_wall_s'):.1f}s")
print(f"  eval_wall_s   : {d['throughput'].get('eval_wall_s'):.1f}s")

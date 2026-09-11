"""Run legacy benchmark_native_packing_vram.py on ONE config to compare against
the new pipeline. Same code path, same measurement, one (B=4, L=4096) Standard.
"""
import os, sys
os.environ.setdefault("HF_HOME", "/app/hf_cache")
sys.path.insert(0, '/app/HMA_Project/legacy')
sys.path.insert(0, '/app/HMA_Project/legacy/benchmarks')

# Import legacy module. Uses its own load_model + run_training.
import benchmark_native_packing_vram as leg

print(f"legacy NUM_STEPS = {leg.NUM_STEPS}")
print(f"legacy WARMUP_STEPS = {leg.WARMUP_STEPS}")

model, tokenizer = leg.load_model(bf16_rmsnorm=False)
leg.cleanup()

result = leg.run_training(
    model, tokenizer, B=4, L=4096,
    method_name='Standard-LoRA',
    hooks_ctx=None,
    gc_enabled=False,
)

print()
print("=== legacy single-config result ===")
print(f"  status:          {result['status']}")
print(f"  peak_vram_gb:    {result['peak_vram_gb']:.2f}")
print(f"  avg_step_ms:     {result.get('avg_step_ms', 'n/a')}")
print(f"  final_loss:      {result.get('final_loss', 'n/a')}")
print(f"  throughput_tok:  {result.get('throughput_tok_per_sec', 'n/a')}")

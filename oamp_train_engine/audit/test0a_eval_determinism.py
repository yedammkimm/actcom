#!/usr/bin/env python3
"""Test 0a — eval determinism on GSM8K (500 samples, 8-shot CoT, greedy).

Loads Llama-3.2-3B-Instruct (bf16, no LoRA), runs the exact evaluate_gsm8k
loop used by benchmark_multiseed.py twice back-to-back in the same process,
and reports per-sample agreement.

Purpose: separate eval-time non-determinism from training-time non-determinism.
Setup mirrors the paper's eval: greedy, batch=1, max_new_tokens=256.
"""
import os
os.environ["HF_HOME"] = "/app/hf_cache"

import sys, gc, json, random, re, time
from datetime import datetime

import numpy as np
import torch

# Reuse eval helpers from the actual benchmark script so we test the same code path.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))
sys.path.insert(0, _HERE)

# Prevent benchmark_multiseed's file-logging side-effects from creating clutter:
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler()], force=True)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# Import helpers *by string exec* to avoid triggering the module's logger setup.
# Cheaper: just copy the constants and helpers we need.
from datasets import load_dataset as load_hf_dataset  # noqa: E402

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
CACHE_DIR = "/app/hf_cache"
RESULTS_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SAMPLES = int(os.environ.get("N_SAMPLES", "500"))
MAX_NEW_TOKENS = 256

GSM8K_FEWSHOT = [
    {"q": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
     "a": "Janet sells 16 - 3 - 4 = 9 duck eggs a day. She makes 9 * 2 = $18 every day. The answer is 18."},
    {"q": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
     "a": "It takes 2/2 = 1 bolt of white fiber. So the total is 2 + 1 = 3. The answer is 3."},
    {"q": "Josh decides to try flipping a house. He buys a house for $80,000 and puts $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
     "a": "The cost was 80000 + 50000 = $130,000. The house value increased by 80000 * 150/100 = $120,000. So the house is worth 80000 + 120000 = $200,000. The profit is 200000 - 130000 = $70,000. The answer is 70000."},
    {"q": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
     "a": "He runs 3 * 3 = 9 sprints a week. So he runs 9 * 60 = 540 meters. The answer is 540."},
    {"q": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
     "a": "Total feed = 20 * 3 = 60 cups. Already given = 15 + 25 = 40 cups. Final meal = 60 - 40 = 20 cups. The answer is 20."},
    {"q": "Kylar went to the store to get water and some apples. A gallon of water costs $2, and each apple costs $1.50. If Kylar bought 3 gallons of water and 5 apples, how much did he spend?",
     "a": "Water cost = 3 * 2 = $6. Apple cost = 5 * 1.5 = $7.5. Total = 6 + 7.5 = $13.5. The answer is 13.5."},
    {"q": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
     "a": "Charleston = 4 * 20 = 80. Toulouse = 2 * 80 = 160. Total = 20 + 80 + 160 = 260. The answer is 260."},
    {"q": "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How long does it take to download the file?",
     "a": "First attempt: 200 * 0.4 / 2 = 40 minutes. Restart: 20 minutes. Second download: 200 / 2 = 100 minutes. Total = 40 + 20 + 100 = 160. The answer is 160."},
]

def format_eval_prompt(q):
    p = ""
    for ex in GSM8K_FEWSHOT:
        p += f"Q: {ex['q']}\nA: {ex['a']}\n\n"
    p += f"Q: {q}\nA:"
    return p

def extract_answer(text):
    m = re.search(r'[Tt]he answer is\s*([\-\d\.,]+)', text)
    if m: return m.group(1).replace(',', '').strip().rstrip('.')
    m = re.search(r'####\s*([\-\d\.,]+)', text)
    if m: return m.group(1).replace(',', '').strip()
    nums = re.findall(r'[\-]?\d+[\.,]?\d*', text)
    return nums[-1].replace(',', '').strip().rstrip('.') if nums else ""

def numbers_equal(pred, expected):
    try:
        return abs(float(pred.replace(',', '').rstrip('.')) -
                   float(expected.replace(',', '').rstrip('.'))) < 0.01
    except Exception:
        return pred.strip() == expected.strip()

def load_gsm8k_test():
    ds = load_hf_dataset('gsm8k', 'main', cache_dir=CACHE_DIR)
    out = []
    for it in ds['test']:
        m = re.search(r'####\s*([\-\d\.,]+)', it['answer'])
        num = m.group(1).replace(',', '').strip() if m else it['answer'].strip()
        out.append({'question': it['question'], 'answer_number': num})
    return out

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def eval_pass(model, tokenizer, device, test_data, label):
    n = min(N_SAMPLES, len(test_data)) if N_SAMPLES > 0 else len(test_data)
    print(f"\n[{label}] running eval over {n} samples", flush=True)
    model.eval()
    per_sample = []
    correct = 0
    t0 = time.time()
    for i, item in enumerate(test_data[:n]):
        prompt = format_eval_prompt(item['question'])
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            gen = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = gen[0, inputs['input_ids'].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = extract_answer(gen_text)
        ok = numbers_equal(pred, item['answer_number'])
        if ok: correct += 1
        per_sample.append({
            "idx": i,
            "n_gen_tokens": int(gen_ids.shape[0]),
            "gen_ids_first16": [int(x) for x in gen_ids[:16].tolist()],
            "gen_ids_last8":   [int(x) for x in gen_ids[-8:].tolist()],
            "pred": pred,
            "gold": item['answer_number'],
            "correct": bool(ok),
        })
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] {i+1}/{n} acc_running={correct/(i+1)*100:.2f}% elapsed={elapsed:.1f}s", flush=True)
    acc = correct / n * 100.0
    dt = time.time() - t0
    print(f"[{label}] acc={acc:.4f}%  ({correct}/{n})  time={dt:.1f}s", flush=True)
    return acc, per_sample

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}  model={MODEL_NAME}  n_samples={N_SAMPLES}", flush=True)
    print(f"torch={torch.__version__}  cuda={torch.version.cuda}  "
          f"cudnn.deterministic=will be set  "
          f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True,
        torch_dtype=torch.bfloat16).to(device)
    model.eval()

    test_data = load_gsm8k_test()
    print(f"gsm8k test size={len(test_data)}", flush=True)

    set_seed(42)
    acc1, samples1 = eval_pass(model, tokenizer, device, test_data, "pass1")

    # No re-load, no seed reset between passes — measure raw eval determinism.
    acc2, samples2 = eval_pass(model, tokenizer, device, test_data, "pass2")

    # Compare
    mism_pred = []
    mism_tokens = []
    for a, b in zip(samples1, samples2):
        if a["pred"] != b["pred"]:
            mism_pred.append({"idx": a["idx"], "pred1": a["pred"], "pred2": b["pred"],
                              "gold": a["gold"]})
        if a["gen_ids_first16"] != b["gen_ids_first16"] or a["gen_ids_last8"] != b["gen_ids_last8"] \
                or a["n_gen_tokens"] != b["n_gen_tokens"]:
            mism_tokens.append({
                "idx": a["idx"],
                "n_gen_tokens1": a["n_gen_tokens"], "n_gen_tokens2": b["n_gen_tokens"],
                "first16_1": a["gen_ids_first16"], "first16_2": b["gen_ids_first16"],
                "last8_1": a["gen_ids_last8"], "last8_2": b["gen_ids_last8"],
                "pred1": a["pred"], "pred2": b["pred"], "gold": a["gold"],
            })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"test0a_eval_determinism_{ts}.json")
    out = {
        "test": "0a_eval_determinism",
        "timestamp": ts,
        "model": MODEL_NAME,
        "n_samples": N_SAMPLES,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "env": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn.deterministic": torch.backends.cudnn.deterministic,
            "cudnn.benchmark": torch.backends.cudnn.benchmark,
        },
        "pass1_acc": acc1,
        "pass2_acc": acc2,
        "delta_acc_pp": acc2 - acc1,
        "n_pred_mismatch": len(mism_pred),
        "n_token_mismatch": len(mism_tokens),
        "pred_mismatches": mism_pred,
        "token_mismatches_first20": mism_tokens[:20],
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== SUMMARY ===")
    print(f"pass1_acc = {acc1:.4f}%   pass2_acc = {acc2:.4f}%   Δ = {acc2 - acc1:+.4f} pp")
    print(f"pred_mismatch = {len(mism_pred)} / {N_SAMPLES}")
    print(f"token_mismatch = {len(mism_tokens)} / {N_SAMPLES}")
    print(f"saved: {out_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Master inventory of results/*.json → CSV. Full-schema recursive scanner."""
import csv, datetime, json, math, os
from glob import glob

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(RESULTS_DIR, "_audit_inventory.csv")

COLUMNS = [
    "filename", "mtime", "model", "seeds", "method",
    "acc_per_seed", "loss_per_seed", "acc_mean", "std_ddof0", "std_ddof1",
    "eval_samples", "compress_mode", "weight_quant", "lr", "epochs",
    "num_train_samples", "lora_target_modules", "max_seq_len",
]

def std(v, d):
    n = len(v)
    if n - d <= 0: return None
    m = sum(v)/n
    return math.sqrt(sum((x-m)**2 for x in v)/(n-d))

def to_pct(x):
    if not isinstance(x, (int, float)): return None
    return x*100.0 if 0.0 < x <= 1.5 else float(x)

def valid_pct_list(vals):
    conv = [to_pct(v) for v in vals]
    if any(c is None for c in conv): return None
    if not all(1.0 <= c <= 99.9 for c in conv): return None
    return conv

def fmt_list(xs):
    if xs is None or xs == []: return ""
    return "[" + ",".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in xs) + "]"

def cfg_get(cfg, *keys):
    for k in keys:
        if isinstance(cfg, dict) and k in cfg and cfg[k] is not None:
            return cfg[k]
    return None

def flat_cfg(data):
    out = {}
    cfg = data.get("config") or {}
    if not isinstance(cfg, dict): return out
    for k, v in cfg.items():
        if isinstance(v, dict):
            for k2, v2 in v.items(): out[k2] = v2
        else: out[k] = v
    return out

def scan_per_seed_dict(per_seed_dict):
    seeds, accs, losses = [], [], []
    if not isinstance(per_seed_dict, dict): return None, None, None
    for k, v in per_seed_dict.items():
        try: sk = int(k)
        except: sk = k
        seeds.append(sk)
        if isinstance(v, dict):
            a = v.get("accuracy") or v.get("acc")
            l = v.get("final_loss") or v.get("loss")
            if a is None: return None, None, None
            accs.append(a); losses.append(l)
        elif isinstance(v, (int, float)):
            accs.append(v); losses.append(None)
        else: return None, None, None
    conv = valid_pct_list(accs)
    if not conv: return None, None, None
    return seeds, conv, losses

ROWS = []

def emit(fn, mtime, model, seeds, method, accs, losses, cfg_flat):
    m = sum(accs)/len(accs) if accs else None
    s0 = std(accs, 0) if accs else None
    s1 = std(accs, 1) if accs else None
    lora_t = cfg_flat.get("lora_target_modules") or cfg_flat.get("target_modules") or cfg_flat.get("targets")
    ROWS.append({
        "filename": fn, "mtime": mtime, "model": model or "",
        "seeds": fmt_list(seeds), "method": method,
        "acc_per_seed": fmt_list(accs) if accs else "",
        "loss_per_seed": fmt_list(losses) if losses and any(l is not None for l in losses) else "",
        "acc_mean": f"{m:.4f}" if m is not None else "",
        "std_ddof0": f"{s0:.4f}" if s0 is not None else "",
        "std_ddof1": f"{s1:.4f}" if s1 is not None else "",
        "eval_samples": cfg_get(cfg_flat, "eval_samples", "gsm8k_eval_samples", "samples") or "",
        "compress_mode": cfg_get(cfg_flat, "compress_mode") or "",
        "weight_quant": cfg_get(cfg_flat, "weight_quant", "quant") or "",
        "lr": cfg_get(cfg_flat, "lr", "learning_rate") or "",
        "epochs": cfg_get(cfg_flat, "epochs", "num_epochs") or "",
        "num_train_samples": cfg_get(cfg_flat, "num_train_samples", "train_samples", "n_train") or "",
        "lora_target_modules": json.dumps(lora_t) if lora_t is not None else "",
        "max_seq_len": cfg_get(cfg_flat, "max_seq_len", "max_length", "seqlen") or "",
    })

def process(path):
    fn = os.path.basename(path)
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
    try: data = json.load(open(path))
    except Exception as e:
        ROWS.append({**{c: "" for c in COLUMNS}, "filename": fn, "mtime": mtime, "method": f"__parse_error__:{e}"})
        return
    if not isinstance(data, dict):
        ROWS.append({**{c: "" for c in COLUMNS}, "filename": fn, "mtime": mtime, "method": "__non_dict__"})
        return
    top_seeds = data.get("seeds") or []
    model = data.get("model") or ((data.get("config") or {}).get("model"))
    cfg = flat_cfg(data)
    emitted = False
    r = data.get("results")
    if isinstance(r, dict):
        for method, mres in r.items():
            if not isinstance(mres, dict): continue
            accs = mres.get("accuracies") or mres.get("acc_per_seed")
            losses = mres.get("losses")
            if isinstance(accs, list):
                conv = valid_pct_list(accs)
                if conv:
                    emit(fn, mtime, model, top_seeds, f"results/{method}", conv, losses, cfg); emitted = True
            for k, v in mres.items():
                if k.endswith("_per_seed"):
                    method_label = k[:-len("_per_seed")]
                    if isinstance(v, dict):
                        seeds, conv, lo = scan_per_seed_dict(v)
                        if conv:
                            emit(fn, mtime, model, seeds, f"results/{method}/{method_label}", conv, lo, cfg); emitted = True
                    elif isinstance(v, list):
                        conv = valid_pct_list(v)
                        if conv:
                            emit(fn, mtime, model, top_seeds, f"results/{method}/{method_label}", conv, None, cfg); emitted = True
    s = data.get("summary")
    if isinstance(s, dict):
        for method, mres in s.items():
            if not isinstance(mres, dict): continue
            ps = mres.get("per_seed")
            if isinstance(ps, dict):
                seeds, conv, lo = scan_per_seed_dict(ps)
                if conv:
                    emit(fn, mtime, model, seeds, f"summary/{method}", conv, lo, cfg); emitted = True
            elif isinstance(ps, list):
                conv = valid_pct_list(ps)
                if conv:
                    emit(fn, mtime, model, top_seeds, f"summary/{method}", conv, None, cfg); emitted = True
    st = data.get("summary_table")
    if isinstance(st, dict):
        for method, benches in st.items():
            if not isinstance(benches, dict): continue
            for b, vals in benches.items():
                if not isinstance(vals, dict): continue
                ps = vals.get("per_seed")
                if isinstance(ps, dict):
                    seeds, conv, lo = scan_per_seed_dict(ps)
                    if conv:
                        emit(fn, mtime, model, seeds, f"summary_table/{method}/{b}", conv, lo, cfg); emitted = True
    sr = data.get("seed_results")
    if isinstance(sr, list) and sr and isinstance(sr[0], dict):
        seeds = [row.get("seed") for row in sr]
        for m in [k for k in sr[0].keys() if k != "seed" and not k.endswith("_retention")]:
            vals = [row.get(m) for row in sr]
            conv = valid_pct_list(vals)
            if conv:
                emit(fn, mtime, model, seeds, f"seed_results/{m}", conv, None, cfg); emitted = True
    prr = data.get("per_run_results")
    if isinstance(prr, dict):
        for bucket, per_seed in prr.items():
            if isinstance(per_seed, dict):
                seeds, conv, lo = scan_per_seed_dict(per_seed)
                if conv:
                    emit(fn, mtime, model, seeds, f"per_run_results/{bucket}", conv, lo, cfg); emitted = True
    seed = data.get("seed")
    for k, v in data.items():
        if isinstance(v, (int, float)) and k.endswith("_accuracy"):
            pct = to_pct(v)
            if pct is not None:
                method_label = k[:-len("_accuracy")]
                emit(fn, mtime, model, [seed] if seed is not None else [], f"flat/{method_label}", [pct], None, cfg)
                emitted = True
    if not emitted:
        emit(fn, mtime, model, top_seeds, "__no_per_seed_accs__", [], None, cfg)

for p in sorted(glob(os.path.join(RESULTS_DIR, "*.json"))):
    process(p)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLUMNS); w.writeheader()
    for r in ROWS: w.writerow(r)
print(f"wrote {len(ROWS)} rows across {len(set(r['filename'] for r in ROWS))} files -> {OUT}")

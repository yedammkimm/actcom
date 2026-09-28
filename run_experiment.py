"""The single entry point for every training, accuracy and memory run.

Order of operations: build the ExperimentConfig from the command line,
collect the environment, validate both before touching the GPU, load the
model (device_map='auto' is refused), apply the dtype policy, which returns
the parameter pointers the pack hooks need, build the pack context
(nullcontext for method 'standard'), run the parity check, then dispatch on
the mode, 'accuracy' or 'memory', and write the result JSON.

Every exception is caught and written as a structured status. The process
never crashes on a CUDA OOM: an out-of-memory record for 70B in BF16 is a
result the paper cites, so it has to be written like any other.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Optional

# ensure oamp/ is importable when run as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

os.environ.setdefault("HF_HOME", "/app/hf_cache")

# Memory safety: rely on docker cgroup (--memory=110g --memory-swap=110g) plus
# an in-process reserved-memory watchdog (see cfg.mem_abort_gb). RLIMIT_AS was
# tried and rejected on 2026-08-18: it blocks the safetensors mmap path for 70B
# (~141 GB virtual footprint) even though physical RSS never approaches that.

import torch
import torch.nn as nn

from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from configs import ExperimentConfig, build_from_cli, total_train_steps
from oamp.data import (
    load_task, format_train_prompt, pack_samples_causal, seq_len_stats,
)
from oamp.dtype_policy import apply_dtype_policy
from oamp.env import collect_env
from oamp.evaluate import evaluate, set_seed
from oamp.pack_hooks import PackHooks, make_pack_hooks
from oamp.schema import (
    append_checkpoint, checkpoint_path_for,
    make_result_skeleton, mark_nan, mark_ok, mark_oom,
    validate_schema, write_result,
)
from oamp.sdpa_utils import (
    normalize_attention_mask, probe_backend_availability, probe_default_dispatch,
    sdpa_backend_context,
)


CACHE_DIR = os.environ.get("HF_HOME", "/app/hf_cache")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Exceptions used to route OOM/NaN into a structured status

class ParityError(RuntimeError):
    """The forward loss differed between the hooks-off and hooks-on paths."""


class NaNError(RuntimeError):
    """Loss became NaN/Inf during training."""


class MemoryBudgetError(RuntimeError):
    """Reserved memory crossed HMA_MEMORY_WATCHDOG_GB during a sweep step.

    Raised by _measure_one so the process exits with an OOM record instead of
    being killed by the docker cgroup (2026-08-18).
    """


# Determinism (2026-09-28 probe)

def _apply_determinism(cfg: ExperimentConfig) -> dict:
    """Apply cfg.deterministic before any CUDA work and return the env fields
    that describe the kernel condition of this process.

    CUBLAS_WORKSPACE_CONFIG has to be in the environment before the cuBLAS
    handle is created, which is why this runs before collect_env and before
    the model is loaded; the chain script also exports it, so setdefault does
    not override an explicit value. The SDPA probes run on synthetic tensors
    under fork_rng and leave the run's RNG streams untouched.
    """
    if cfg.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True, warn_only=cfg.deterministic_warn_only)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    info = {
        'deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()),
        'deterministic_warn_only': bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
        'sdpa_backend_requested': cfg.sdpa_backend,
        'sdpa_backend_probe': 'UNKNOWN',
        'sdpa_default_dispatch': 'UNKNOWN',
    }
    if torch.cuda.is_available():
        try:
            info['sdpa_backend_probe'] = probe_backend_availability()
            info['sdpa_default_dispatch'] = probe_default_dispatch()
        except Exception as e:                       # noqa: BLE001
            info['sdpa_backend_probe'] = f'probe failed: {type(e).__name__}: {e}'[:200]
    return info


def _hash_tensors(tensors) -> str:
    """sha256 over the raw bytes of the tensors, in the order given.

    bf16 and fp16 are viewed as int16, fp32 as int32, so the bytes are the
    stored bits and no conversion touches them. Two runs whose LoRA
    parameters hash equal are bit-for-bit identical, which is a stronger
    statement than equal losses."""
    h = hashlib.sha256()
    for t in tensors:
        t = t.detach().contiguous()
        if t.dtype in (torch.bfloat16, torch.float16):
            t = t.view(torch.int16)
        elif t.dtype == torch.float32:
            t = t.view(torch.int32)
        h.update(t.cpu().numpy().tobytes())
    return h.hexdigest()


# Model loading

def _load_model(cfg: ExperimentConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id, cache_dir=CACHE_DIR, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.model_id.endswith('-bnb-4bit'):
        # Pre-quantized checkpoint: config.json already carries quantization_config.
        # Passing bnb here again conflicts or triggers re-quantization on CPU
        # (which was the anon-RAM blowout on the online path, 2026-08-18).
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, cache_dir=CACHE_DIR, local_files_only=True,
            low_cpu_mem_usage=True,
            device_map={"": 0},
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False)
    elif cfg.weight_quant == 'nf4':
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # device_map={"": 0} forces layer-by-layer materialization on GPU 0, so
        # only one layer's BF16 shard sits in CPU RAM at a time. .to(DEVICE)
        # instead built the whole quantized model on CPU first, blowing anon RAM
        # past the 110 GB cgroup ceiling on 70B (2026-08-18).
        # 'auto' would trigger meta-device offload on the GB10, which is why it is refused.
        # max_memory tells accelerate the GPU budget explicitly; without it the
        # dispatcher can silently keep spilling BF16 tensors to CPU.
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, cache_dir=CACHE_DIR, local_files_only=True,
            quantization_config=bnb,
            low_cpu_mem_usage=True,
            device_map={"": 0},
            max_memory={0: "95GiB"},
        )
        # Gradient checkpointing is applied after get_peft_model, never through
        # prepare_model_for_kbit_training: enabling it here would also toggle
        # enable_input_require_grads on the embedding, which the validated dtype
        # setup did not have.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, cache_dir=CACHE_DIR, local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(DEVICE)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_targets),
        bias='none',
    )
    model = get_peft_model(model, lora_cfg)

    # GC applied after PEFT wrapping. Combining GC with a pack context is not
    # validated (the earlier LoRA+GC arm ran without hooks); refuse for safety.
    if cfg.gc_enabled:
        if cfg.method != 'standard':
            raise ValueError(
                "gc_enabled=True is only supported with method='standard'. "
                "GC + pack combination has not been validated.")
        model.gradient_checkpointing_enable()

    return model, tokenizer


# Pack context factory

def _build_pack_ctx(cfg: ExperimentConfig, *, param_ptrs: set, vocab_size: int):
    """Return a context manager and (optional) PackHooks reference.

    Standard runs use nullcontext so pack_stats stays None in the result JSON.
    """
    if cfg.method == 'standard':
        return contextlib.nullcontext(), None
    if cfg.dedupe:
        print("[warn] cfg.dedupe=True is EXPERIMENTAL. Cross-step storage_ptr "
              "recycling caused NaN at step 7 on 3B/L=4096 (2026-08-14). "
              "Do not use in production runs.", flush=True)
    ctx = make_pack_hooks(
        cfg.method,
        fp8_ratio=cfg.fp8_ratio,
        group_size=cfg.group_size,
        min_numel=cfg.min_numel,
        # Filter set from the loaded model's config, not the tokenizer.
        skip_last_dims={vocab_size},
        param_ptrs=param_ptrs,
        mask_seed=cfg.mask_seed,
        dedupe=cfg.dedupe,
        stochastic_rounding=cfg.pack_stochastic_rounding,
        sr_seed=cfg.sr_seed,
        pack_4d_mode=cfg.pack_4d_mode,
        body_encoding=cfg.body_encoding,
    )
    return ctx, ctx


# Parity check

def _parity_check(model, pack_ctx, tokenizer, cfg: ExperimentConfig,
                  vocab_size: int) -> None:
    """Forward-only parity: the hooks-off and hooks-on losses must be bit-exact.

    Runs on a synthetic (B=2, L=cfg.max_seq_len) batch in eval() so dropout
    adds no noise. Gradients are enabled so the hooks fire (the kept count must
    be positive), but each forward's graph is torn down between the two calls
    so the check does not inflate the training run's peak, and the peak
    statistics are reset afterwards so memory.peak_* reflects only the real
    workload.
    """
    print("[parity] running the forward-only parity check ...", flush=True)
    model.eval()
    torch.manual_seed(cfg.seed)
    ids = torch.randint(0, vocab_size, (2, cfg.max_seq_len), device=DEVICE)

    with torch.enable_grad(), sdpa_backend_context(cfg.sdpa_backend):
        loss_off_val = float(model(input_ids=ids, labels=ids).loss.item())
    torch.cuda.empty_cache()

    with torch.enable_grad(), sdpa_backend_context(cfg.sdpa_backend), pack_ctx:
        loss_on_val = float(model(input_ids=ids, labels=ids).loss.item())
    torch.cuda.empty_cache()

    d = abs(loss_off_val - loss_on_val)
    kept = pack_ctx.pack_stats.get('kept', 0) if isinstance(pack_ctx, PackHooks) else 0
    print(f"[parity] loss_off={loss_off_val:.10f}  loss_on={loss_on_val:.10f}  "
          f"|Δ|={d:.2e}  kept={kept}", flush=True)
    if d != 0.0:
        raise ParityError(
            f"parity check failed: |Δloss|={d:.2e} (must be 0). "
            f"pack context is leaking into forward — inspect filter changes.")
    if isinstance(pack_ctx, PackHooks) and kept == 0:
        raise ParityError(
            "parity check: pack context ran but stats.kept == 0. "
            "Filter set may be blocking every tensor.")

    # Reset stats so the real run starts from zero.
    if isinstance(pack_ctx, PackHooks):
        for k in pack_ctx.pack_stats:
            pack_ctx.pack_stats[k] = 0

    model.train()
    # Reset peak stats so 'memory.peak_*' reflects only the real workload.
    torch.cuda.reset_peak_memory_stats(DEVICE)


# Optimizer / scheduler

def _build_optimizer(model, cfg: ExperimentConfig, n_effective_samples: int = None):
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt_name = cfg.optimizer.lower()
    if opt_name == 'adamw':
        opt = torch.optim.AdamW(lora_params, lr=cfg.lr, weight_decay=0.01)
    elif opt_name == 'paged_adamw8bit':
        # bitsandbytes PagedAdamW8bit, as in the earlier NF4 runs.
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise RuntimeError(
                "optimizer='paged_adamw8bit' requires bitsandbytes.") from e
        opt = bnb.optim.PagedAdamW8bit(lora_params, lr=cfg.lr, weight_decay=0.01)
    else:
        raise ValueError(f"optimizer {cfg.optimizer!r} not supported")

    # data_packing runs pass the chunk count so the scheduler sees the true horizon.
    n_for_steps = n_effective_samples if n_effective_samples is not None else cfg.n_train
    total_steps = total_train_steps(
        n_train=n_for_steps, batch_size=cfg.batch_size,
        grad_accum_steps=cfg.grad_accum_steps, epochs=cfg.epochs)
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    if cfg.scheduler == 'cosine':
        sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
    elif cfg.scheduler == 'linear':
        sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    elif cfg.scheduler in ('constant', 'none'):
        sched = None
    else:
        raise ValueError(f"scheduler {cfg.scheduler!r} not supported")
    return opt, sched, total_steps, lora_params


# Accuracy mode

def _tokenize_train(tokenizer, cfg: ExperimentConfig, text: str):
    """Tokenize a single training example. Natural length; no padding."""
    return tokenizer(
        text, return_tensors='pt',
        truncation=True, max_length=cfg.max_seq_len,
        padding='max_length' if cfg.padding else False,
    )


def _run_accuracy(model, tokenizer, cfg: ExperimentConfig,
                  pack_ctx, param_ptrs, result: dict,
                  ckpt_path: str, output_path: str) -> None:
    import numpy as np

    # Save the partially-trained LoRA adapter before we raise NaNError (or any
    # other unrecoverable training error) so downstream postmortem can still
    # measure held-out perplexity on the last coherent checkpoint. Idempotent
    # side-effect; ignores save failures.
    def _save_partial_adapter(reason: str):
        if output_path.endswith('.json'):
            adir = output_path[:-len('.json')] + '_adapter_partial'
        else:
            adir = output_path + '_adapter_partial'
        try:
            model.save_pretrained(adir)
            result['results']['adapter_path_partial'] = adir
            result['results']['adapter_partial_reason'] = reason
            print(f"[adapter:partial] saved to {adir}  reason={reason}", flush=True)
        except Exception as e:                       # noqa: BLE001
            print(f"[adapter:partial] save FAILED: "
                  f"{type(e).__name__}: {e}", flush=True)

    # test_dtype_policy_3way used random.Random(seed).shuffle(train) -> [:n_train].
    # oamp.data.load_task(shuffle=True) uses the same Python random.Random(seed),
    # so the exact 500-sample slice matches when n_train=500.
    train_data = load_task(cfg.task, 'train',
                           n_samples=cfg.n_train, seed=cfg.seed,
                           cache_dir=CACHE_DIR, shuffle=True)
    test_data = load_task(cfg.task, 'test',
                          n_samples=cfg.eval_samples, seed=cfg.seed,
                          cache_dir=CACHE_DIR, shuffle=False)

    # Optional causal-LM packing: concat EOS-separated samples, chunk to a fixed
    # length. Keeps mask=None (FLASH-safe), long-sequence pilot condition (2026-08-18).
    packed_chunks = None
    if cfg.data_packing:
        packed_chunks = pack_samples_causal(
            train_data, tokenizer, cfg.task, cfg.packed_seq_len)
        print(f"[pack] {len(train_data)} samples -> {len(packed_chunks)} "
              f"chunks of {cfg.packed_seq_len} tokens", flush=True)
        if len(packed_chunks) == 0:
            raise ValueError(
                f"data_packing produced 0 chunks (packed_seq_len={cfg.packed_seq_len} "
                f"too large for {len(train_data)} samples).")

    n_effective = len(packed_chunks) if packed_chunks is not None else None
    opt, sched, total_steps, lora_params = _build_optimizer(
        model, cfg, n_effective_samples=n_effective)
    result['throughput']['steps'] = total_steps
    # AdamW state is lazy; this is memory pre-first-step, not post-state-alloc.
    result['memory']['mem_after_optimizer_init_gb'] = torch.cuda.memory_allocated(DEVICE) / 1e9
    # Reset peak so 'memory.peak_*' captures only the training loop, matching
    # the timing of test_dtype_policy_3way.
    torch.cuda.reset_peak_memory_stats(DEVICE)

    model.train()
    step_losses = []
    seen_lens = []
    running_loss = 0.0
    micro_step = 0
    opt_step = 0
    n_nonfinite_grad_steps = 0  # opt-step attempts where raw grad had inf/nan (skipped)
    grad_norm_trace = []        # (opt_step, pre_clip_norm) sampled every checkpoint_every
    skipped_step_indices = []   # opt_step values at each skip (first 100 only)
    # Determinism probe (2026-09-28). All of it is off unless the flags are set,
    # so a default run records exactly what it recorded before.
    probe = cfg.param_hash_every > 0
    micro_losses = [] if cfg.record_micro_losses else None   # repr(float) per micro-step
    grad_hash_trace = []        # (opt_step, sha256 of every LoRA grad before clipping)
    param_hash_trace = []       # (opt_step, sha256 of every LoRA parameter after the update)
    step_wall_s = []            # per optimizer step, synchronised, hashing excluded
    stopped_at_opt_step = None
    stop_training = False
    if probe:
        lora_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        assert [id(p) for _, p in lora_named] == [id(p) for p in lora_params]
    t_train0 = time.time()
    t_step0 = time.perf_counter()

    # As in the earlier build_data_order: one np.random.RandomState(seed) reused across epochs.
    # Pre-materializing avoids interference with the global RNG stream used by
    # dropout / bnb during training.
    rng = np.random.RandomState(cfg.seed)
    epoch_orders = []
    data_len = len(packed_chunks) if packed_chunks is not None else len(train_data)
    for _ in range(cfg.epochs):
        idx = np.arange(data_len)
        rng.shuffle(idx)
        epoch_orders.append(idx.tolist())

    for epoch, indices in enumerate(epoch_orders):
        for idx in indices:
            if packed_chunks is not None:
                input_ids = torch.tensor(
                    [packed_chunks[idx]], dtype=torch.long, device=DEVICE)
                labels = input_ids.clone()
                attn = None
            else:
                sample = train_data[idx]
                text = format_train_prompt(cfg.task, sample)
                enc = _tokenize_train(tokenizer, cfg, text)
                input_ids = enc['input_ids'].to(DEVICE)
                attn = normalize_attention_mask(enc['attention_mask'].to(DEVICE))
                labels = input_ids.clone()

            seen_lens.append(int(input_ids.shape[1]))

            sdpa_ctx = sdpa_backend_context(cfg.sdpa_backend)   # nullcontext by default
            with sdpa_ctx:
                with pack_ctx:
                    if attn is None:
                        out = model(input_ids=input_ids, labels=labels)
                    else:
                        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                    loss = out.loss / cfg.grad_accum_steps

                if not torch.isfinite(loss):
                    _save_partial_adapter(
                        f"nan_loss at epoch={epoch} idx={idx} micro_step={micro_step}")
                    raise NaNError(
                        f"loss became {loss.item()} at epoch {epoch}, sample idx={idx}, "
                        f"micro_step={micro_step}, opt_step={opt_step}")

                loss.backward()
            loss_val = loss.item()
            running_loss += loss_val
            if micro_losses is not None:
                micro_losses.append(repr(float(loss_val)))
            micro_step += 1

            if micro_step % cfg.grad_accum_steps == 0:
                # Detect nonfinite grads BEFORE clip_grad_norm_: the clip's `0 * inf`
                # writes NaN into every grad and Adam's m/v are permanently poisoned
                # once NaN enters them (2026-08-17 Qwen: 20/500 sacrificial steps).
                # This mirrors torch.cuda.amp.GradScaler skip semantics.
                has_nonfinite_grad = False
                for p in lora_params:
                    g = p.grad
                    if g is not None and not torch.isfinite(g).all():
                        has_nonfinite_grad = True
                        break

                # Count in both modes so the control run records how many steps
                # would have been skipped; only act on it when the guard is on.
                if has_nonfinite_grad:
                    n_nonfinite_grad_steps += 1
                    if len(skipped_step_indices) < 100:
                        skipped_step_indices.append(opt_step)

                if has_nonfinite_grad and cfg.nonfinite_skip:
                    opt.zero_grad(set_to_none=True)
                    running_loss = 0.0
                    micro_step = 0
                    if n_nonfinite_grad_steps > total_steps * 0.3:
                        _save_partial_adapter(
                            f"skipped_nnf={n_nonfinite_grad_steps}/{total_steps} "
                            f"(>30%) at opt_step={opt_step}")
                        raise NaNError(
                            f"skipped {n_nonfinite_grad_steps}/{total_steps} "
                            f"steps (>30%): quantization not viable for this model")
                    continue

                if probe and (opt_step + 1) % cfg.param_hash_every == 0:
                    grad_hash_trace.append(
                        (opt_step + 1, _hash_tensors([p.grad for p in lora_params if p.grad is not None])))

                if cfg.grad_clip > 0:
                    pre_clip_norm = torch.nn.utils.clip_grad_norm_(lora_params, cfg.grad_clip)
                else:
                    with torch.no_grad():
                        pre_clip_norm = torch.linalg.vector_norm(
                            torch.stack([torch.linalg.vector_norm(p.grad) for p in lora_params
                                         if p.grad is not None]))
                grad_norm_finite = True
                opt.step()
                if sched is not None:
                    sched.step()
                opt.zero_grad(set_to_none=True)

                opt_step += 1
                step_loss = running_loss
                step_losses.append(float(step_loss))
                running_loss = 0.0
                result['throughput']['steps_completed'] = opt_step

                if probe:
                    torch.cuda.synchronize(DEVICE)
                    step_wall_s.append(time.perf_counter() - t_step0)
                    if opt_step % cfg.param_hash_every == 0:
                        param_hash_trace.append((opt_step, _hash_tensors(lora_params)))
                    t_step0 = time.perf_counter()

                if opt_step == 1:
                    result['memory']['mem_after_first_step_gb'] = (
                        torch.cuda.memory_allocated(DEVICE) / 1e9)

                if opt_step % cfg.checkpoint_every == 0 or opt_step == total_steps:
                    elapsed = time.time() - t_train0
                    peak_alloc = torch.cuda.max_memory_allocated(DEVICE) / 1e9
                    peak_res = torch.cuda.max_memory_reserved(DEVICE) / 1e9
                    grad_norm_val = float(pre_clip_norm) if grad_norm_finite else float('inf')
                    grad_norm_trace.append((opt_step, grad_norm_val, grad_norm_finite))
                    append_checkpoint(
                        ckpt_path,
                        step=opt_step, loss=step_loss,
                        elapsed_s=elapsed, peak_alloc_gb=peak_alloc,
                        extra={
                            'peak_reserved_gb': peak_res,
                            'epoch': epoch,
                            'lr': (sched.get_last_lr()[0] if sched is not None else cfg.lr),
                            'grad_norm': grad_norm_val,
                            'grad_norm_finite': grad_norm_finite,
                            'n_nonfinite_grad_steps_so_far': n_nonfinite_grad_steps,
                        })
                    print(f"[train] ep{epoch+1} step {opt_step}/{total_steps}  "
                          f"loss={step_loss:.4f}  peak={peak_alloc:.2f}GB  "
                          f"elapsed={elapsed:.1f}s", flush=True)
                    if peak_res > cfg.mem_abort_gb:
                        mark_oom(result, stage='memory_sweep', step=opt_step,
                                 allocated_gb=peak_alloc,
                                 error=f"reserved={peak_res:.2f}GB > mem_abort_gb="
                                       f"{cfg.mem_abort_gb} at opt_step={opt_step}")
                        result['memory']['peak_reserved_gb'] = peak_res
                        result['memory']['peak_allocated_gb'] = peak_alloc
                        result['results']['step_losses'] = step_losses
                        result['results']['final_loss'] = step_losses[-1] if step_losses else None
                        result['results']['n_nonfinite_grad_steps'] = n_nonfinite_grad_steps
                        result['results']['grad_norm_trace'] = grad_norm_trace
                        result['results']['skipped_step_indices'] = skipped_step_indices
                        try:
                            write_result(output_path, result)
                        except Exception:
                            pass
                        print(f"[watchdog] reserved={peak_res:.2f}GB exceeds "
                              f"abort={cfg.mem_abort_gb}GB at opt_step={opt_step}; "
                              f"exiting immediately.", flush=True)
                        sys.exit(1)

                if cfg.max_opt_steps > 0 and opt_step >= cfg.max_opt_steps:
                    stopped_at_opt_step = opt_step
                    stop_training = True
                    print(f"[train] stopping at opt_step={opt_step} (max_opt_steps)", flush=True)
                    break
        if stop_training:
            break

    train_wall = time.time() - t_train0

    # Sequence stats
    result['sequence'].update(seq_len_stats(seen_lens))
    # Throughput
    result['throughput']['train_wall_s'] = train_wall
    if step_losses:
        result['throughput']['seconds_per_step'] = train_wall / len(step_losses)
    result['throughput']['tokens_per_second'] = (
        result['sequence']['tokens_total'] / train_wall if train_wall > 0 else 0.0)
    # Memory peaks
    result['memory']['peak_allocated_gb'] = torch.cuda.max_memory_allocated(DEVICE) / 1e9
    result['memory']['peak_reserved_gb']  = torch.cuda.max_memory_reserved(DEVICE) / 1e9
    # Accuracy mode's peak covers the entire training loop after the pre-loop
    # reset (see reset_peak_memory_stats above).
    result['memory']['peak_measurement']  = 'training_loop_max_post_reset'
    # Loss
    result['results']['step_losses'] = step_losses
    result['results']['final_loss']  = step_losses[-1] if step_losses else None
    result['results']['n_nonfinite_grad_steps'] = n_nonfinite_grad_steps
    result['results']['grad_norm_trace'] = grad_norm_trace
    result['results']['skipped_step_indices'] = skipped_step_indices
    # Determinism-probe records; the keys exist only when the flags were set.
    if cfg.max_opt_steps > 0:
        result['results']['stopped_at_opt_step'] = stopped_at_opt_step
    if micro_losses is not None:
        result['results']['micro_losses'] = micro_losses
    if probe:
        result['results']['grad_hash_trace'] = grad_hash_trace
        result['results']['param_hash_trace'] = param_hash_trace
        result['results']['step_wall_s'] = step_wall_s
        result['results']['param_hash_by_name'] = {
            n: _hash_tensors([p]) for n, p in lora_named}

    # Save the LoRA adapter before eval
    # Eval on 70B risks KV-cache OOM; if that happens post-eval save would lose
    # the trained adapter (2026-08-18). Save first so re-evaluation is cheap.
    if output_path.endswith('.json'):
        adapter_dir = output_path[:-len('.json')] + '_adapter'
    else:
        adapter_dir = output_path + '_adapter'
    try:
        model.save_pretrained(adapter_dir)
        result['results']['adapter_path'] = adapter_dir
        print(f"[adapter] saved to {adapter_dir}", flush=True)
    except Exception as e:                       # noqa: BLE001
        result['results']['adapter_path'] = None
        print(f"[adapter] WARNING: save failed: {type(e).__name__}: {e}", flush=True)

    if cfg.skip_eval:
        result['results']['eval_skipped'] = True
        print("[eval] skipped (skip_eval)", flush=True)
        return

    # Eval
    # on_abort: persist a partial JSON with the abort diagnostics before
    # sys.exit(1), so a postmortem can distinguish "OOM in eval" from a silent hang.
    def _persist_partial_eval(diag: dict) -> None:
        result['results']['eval_aborted'] = True
        result['results']['eval_abort_diagnostics'] = {
            k: v for k, v in diag.items() if k != 'partial_per_sample'
        }
        result['results']['per_sample'] = diag.get('partial_per_sample', [])
        result['results']['gsm8k_accuracy'] = diag.get('partial_accuracy_pct')
        result['results']['n_correct'] = diag.get('partial_n_correct')
        result['results']['n_eval_samples'] = diag.get('abort_at_sample')
        try:
            write_result(output_path, result)
            print(f"[eval:watchdog] partial result written to {output_path}",
                  flush=True)
        except Exception as e:                       # noqa: BLE001
            print(f"[eval:watchdog] write_result FAILED: "
                  f"{type(e).__name__}: {e}", flush=True)

    eval_out = evaluate(
        model, tokenizer, cfg.task, test_data,
        device=DEVICE, n_samples=cfg.eval_samples,
        max_new_tokens=cfg.eval_max_new_tokens,
        label='eval', log_every=50,
        abort_gb=cfg.mem_abort_gb,
        abort_check_every=20,
        on_abort=_persist_partial_eval,
    )

    result['results']['gsm8k_accuracy'] = eval_out['accuracy_pct']
    result['results']['n_correct'] = eval_out['n_correct']
    result['results']['n_eval_samples'] = eval_out['n_samples']
    result['results']['per_sample'] = eval_out['per_sample']
    result['results']['eval_diagnostics'] = eval_out['eval_diagnostics']
    result['throughput']['eval_wall_s'] = eval_out['eval_wall_s']


# Memory mode

def _run_memory(model, tokenizer, cfg: ExperimentConfig,
                pack_ctx, param_ptrs, result: dict,
                ckpt_path: str, output_path: str) -> None:
    """Measure ONE (B, L) in this process.

    Multi-config sweeps must launch one process per (B, L) so CUDA allocator
    fragmentation from an earlier config never inflates the next one's peak.
    A shell loop over ``--mem_batch_size`` × ``--mem_seq_len`` produces one
    JSON per config, which downstream tables can concatenate.
    """
    opt, _, _, lora_params = _build_optimizer(model, cfg)
    result['memory']['mem_after_optimizer_init_gb'] = torch.cuda.memory_allocated(DEVICE) / 1e9

    # Dedicated Generator so the global RNG is untouched.
    gen = torch.Generator(device=DEVICE).manual_seed(cfg.seed)
    vocab = model.config.vocab_size

    B, L = cfg.mem_batch_size, cfg.mem_seq_len
    print(f"[mem] B={B} L={L}  warmup={cfg.mem_warmup}  steps={cfg.mem_steps}  "
          f"abort_gb={cfg.mem_abort_gb}", flush=True)

    stats = _measure_one(model, cfg, pack_ctx, lora_params, opt,
                         B=B, L=L, vocab=vocab, gen=gen,
                         result=result, output_path=output_path)

    # Flat schema: no per-(B, L) sweep list.
    result['memory']['peak_allocated_gb'] = stats['peak_allocated_gb']
    result['memory']['peak_reserved_gb']  = stats['peak_reserved_gb']
    result['memory']['peak_measurement']  = stats['peak_measurement']
    result['memory']['mem_after_first_step_gb'] = None    # not applicable in memory mode
    result['results']['final_loss'] = stats['final_loss']
    result['throughput']['seconds_per_step'] = stats['seconds_per_step']
    result['throughput']['train_wall_s'] = stats['train_wall_s']
    result['throughput']['tokens_per_second'] = stats['tokens_per_second']
    result['throughput']['steps'] = cfg.mem_steps
    result['sequence'].update(seq_len_stats([L] * (cfg.mem_steps * B)))

    append_checkpoint(
        ckpt_path,
        step=cfg.mem_steps, loss=stats['final_loss'],
        elapsed_s=stats['train_wall_s'],
        peak_alloc_gb=stats['peak_allocated_gb'],
        extra={'B': B, 'L': L,
               'peak_reserved_gb': stats['peak_reserved_gb'],
               'peak_measurement': stats['peak_measurement']})


def _measure_one(model, cfg: ExperimentConfig, pack_ctx, lora_params, opt,
                 *, B: int, L: int, vocab: int, gen: torch.Generator,
                 result: dict, output_path: str) -> dict:
    """Measure one (B, L) configuration.

    The peak is the largest per-step peak_allocated after warmup, the same
    definition as the earlier memory benchmark, which isolates the
    forward-plus-backward peak from what model loading and optimizer
    initialisation leave behind.
    """
    torch.cuda.empty_cache()
    model.train()
    losses = []
    step_times = []
    step_peaks = []
    step_reserved = []
    t0 = time.time()
    step = 0

    # Mid-step watchdog (2026-08-20 experiment A): forward/backward/step peaks
    # can spike between the per-step boundary that used to be the only guard.
    # sys.exit(1) prevents outer try/except from silently continuing while
    # reserved memory keeps climbing toward the container/host limit.
    def _mem_guard(phase):
        r = torch.cuda.memory_reserved(DEVICE) / 1e9
        if r > cfg.mem_abort_gb:
            alloc_gb = torch.cuda.memory_allocated(DEVICE) / 1e9
            mark_oom(result, stage='memory_sweep', step=step,
                     allocated_gb=alloc_gb,
                     error=f"reserved={r:.2f}GB > mem_abort_gb={cfg.mem_abort_gb} "
                           f"at phase={phase} step={step} (B={B}, L={L})")
            result['memory']['peak_reserved_gb'] = r
            result['memory']['peak_allocated_gb'] = alloc_gb
            result['memory']['peak_measurement'] = f'watchdog:{phase}'
            try:
                write_result(output_path, result)
            except Exception:
                pass
            print(f"[watchdog:{phase}] reserved={r:.2f}GB > abort={cfg.mem_abort_gb}GB "
                  f"at step={step}; exiting immediately.", flush=True)
            sys.exit(1)

    for step in range(cfg.mem_warmup + cfg.mem_steps):
        ids = torch.randint(0, vocab, (B, L), device=DEVICE, generator=gen)
        # Reset per-step so peak_allocated captures only this step's workload.
        torch.cuda.reset_peak_memory_stats(DEVICE)
        t_s = time.perf_counter()
        with pack_ctx:
            out = model(input_ids=ids, labels=ids)
            loss = out.loss
        _mem_guard('post_forward')
        if not torch.isfinite(loss):
            raise NaNError(f"memory mode: NaN loss at step={step} B={B} L={L}")
        loss.backward()
        _mem_guard('post_backward')
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(lora_params, cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(DEVICE)
        _mem_guard('post_step')
        t_e = time.perf_counter()
        if step >= cfg.mem_warmup:
            step_times.append(t_e - t_s)
            losses.append(float(loss.item()))
            step_peaks.append(torch.cuda.max_memory_allocated(DEVICE) / 1e9)
            step_reserved.append(torch.cuda.max_memory_reserved(DEVICE) / 1e9)
    wall = time.time() - t0
    tokens = B * L * len(step_times)
    return {
        'peak_allocated_gb': max(step_peaks) if step_peaks else 0.0,
        'peak_reserved_gb':  max(step_reserved) if step_reserved else 0.0,
        'peak_measurement':  'per_step_max_after_warmup',
        'seconds_per_step':  sum(step_times) / max(1, len(step_times)),
        'train_wall_s':      wall,
        'tokens_per_second': tokens / wall if wall > 0 else 0.0,
        'final_loss':        losses[-1] if losses else None,
    }


# Path helper

def _default_output_path(cfg: ExperimentConfig) -> str:
    if getattr(cfg, 'output_path', None):
        return cfg.output_path
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_tag = cfg.model_id.split('/')[-1]
    rms_tag = 'rmsbf16' if cfg.bf16_rmsnorm else 'rmsfp32'
    stem = (f'{cfg.mode}__{cfg.method}__{model_tag}__{cfg.weight_quant}'
            f'__{rms_tag}__seed{cfg.seed}__{ts}.json')
    return os.path.join(cfg.output_dir, stem)


# main

def main(argv=None) -> int:
    cfg = build_from_cli(argv)
    det_info = _apply_determinism(cfg)     # before any CUDA work
    env = collect_env()
    env.update(det_info)
    env['timestamp_start'] = datetime.now().isoformat()

    # (3) validation before any GPU work
    validate_schema(cfg.asdict(), env)

    output_path = _default_output_path(cfg)
    ckpt_path = checkpoint_path_for(output_path)
    print(f"[start] mode={cfg.mode} method={cfg.method} model={cfg.model_id} "
          f"weight_quant={cfg.weight_quant} seed={cfg.seed}", flush=True)
    print(f"[start] output={output_path}", flush=True)
    print(f"[start] checkpoint sidecar={ckpt_path}", flush=True)

    result = make_result_skeleton(cfg.asdict(), env)

    current_stage = 'model_load'
    try:
        # (4)+(5): load model, apply the dtype policy
        set_seed(cfg.seed)
        # set_seed fixes the cudnn flags for every run; record what the training saw.
        env['cudnn_deterministic'] = bool(torch.backends.cudnn.deterministic)
        model, tokenizer = _load_model(cfg)
        model, param_ptrs, dtype_report = apply_dtype_policy(
            model, weight_quant=cfg.weight_quant, bf16_rmsnorm=cfg.bf16_rmsnorm,
            bf16_params=cfg.bf16_params, bf16_lora=cfg.bf16_lora)
        result['model']['dtype_report'] = dtype_report
        result['model']['n_params']    = sum(p.numel() for p in model.parameters())
        result['model']['n_trainable'] = sum(p.numel() for p in model.parameters()
                                             if p.requires_grad)
        result['memory']['mem_after_load_gb'] = torch.cuda.memory_allocated(DEVICE) / 1e9

        vocab_size = model.config.vocab_size    # NOT tokenizer.vocab_size (§8)

        # (6) pack context
        pack_ctx, pack_hooks_ref = _build_pack_ctx(
            cfg, param_ptrs=param_ptrs, vocab_size=vocab_size)

        # (7) parity check
        if pack_hooks_ref is not None:
            current_stage = 'parity'
            _parity_check(model, pack_hooks_ref, tokenizer, cfg, vocab_size)

        # (8) mode dispatch
        if cfg.mode == 'accuracy':
            current_stage = 'forward'
            _run_accuracy(model, tokenizer, cfg, pack_ctx, param_ptrs, result, ckpt_path, output_path)
            current_stage = 'eval'  # eval OOM would trip here after training
        else:
            current_stage = 'forward'
            _run_memory(model, tokenizer, cfg, pack_ctx, param_ptrs, result,
                        ckpt_path, output_path)

        # Final pack stats + coverage
        if pack_hooks_ref is not None:
            result['pack']['pack_stats'] = dict(pack_hooks_ref.pack_stats)
            result['pack']['effective_bits'] = pack_hooks_ref.effective_bits()

        mark_ok(result)

    except ParityError as e:
        traceback.print_exc()
        result['status'] = 'PARITY_FAIL'
        result['error'] = f'ParityError: {e}'
    except NaNError as e:
        traceback.print_exc()
        step = result['throughput'].get('steps_completed', 0)
        mark_nan(result, step=step, error=str(e))
    except torch.cuda.OutOfMemoryError as e:
        traceback.print_exc()
        alloc = torch.cuda.memory_allocated(DEVICE) / 1e9 if torch.cuda.is_available() else 0.0
        step = result['throughput'].get('steps_completed', 0)
        mark_oom(result, stage=current_stage, step=step,
                 allocated_gb=alloc, error=str(e)[:400])
    except RuntimeError as e:
        # bitsandbytes and safetensors mmap sometimes wrap CUDA/system OOM as
        # plain RuntimeError; filter by message so real bugs still surface.
        msg = str(e).lower()
        if ('out of memory' in msg or 'cannot allocate memory' in msg
                or 'unable to mmap' in msg):
            traceback.print_exc()
            alloc = torch.cuda.memory_allocated(DEVICE) / 1e9 if torch.cuda.is_available() else 0.0
            step = result['throughput'].get('steps_completed', 0)
            mark_oom(result, stage=current_stage, step=step,
                     allocated_gb=alloc, error=str(e)[:400])
        else:
            raise
    except MemoryBudgetError as e:
        traceback.print_exc()
        alloc = torch.cuda.memory_allocated(DEVICE) / 1e9 if torch.cuda.is_available() else 0.0
        step = result['throughput'].get('steps_completed', 0)
        mark_oom(result, stage='memory_sweep', step=step,
                 allocated_gb=alloc, error=str(e)[:400])
    except MemoryError as e:
        traceback.print_exc()
        alloc = torch.cuda.memory_allocated(DEVICE) / 1e9 if torch.cuda.is_available() else 0.0
        step = result['throughput'].get('steps_completed', 0)
        mark_oom(result, stage=current_stage, step=step,
                 allocated_gb=alloc,
                 error=f"MemoryError: {str(e)[:200]}")
    except Exception as e:
        traceback.print_exc()
        result['status'] = 'ERROR'
        result['error'] = f'{type(e).__name__}: {e}'
    finally:
        env['timestamp_end'] = datetime.now().isoformat()
        result['env'] = env
        try:
            write_result(output_path, result)
        except Exception as e:
            # last-resort save with PENDING allowed
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'w') as f:
                import json as _json
                _json.dump(result, f, indent=2, default=str)
            print(f"[warn] fallback save: {e}", flush=True)
        print(f"[done] status={result['status']}  -> {output_path}", flush=True)

    return 0 if result['status'] == 'OK' else 1


if __name__ == '__main__':
    sys.exit(main())

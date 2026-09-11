"""Check if tied embed<->lm_head weight survives apply_dtype_policy."""
import os, sys
os.environ.setdefault("HF_HOME", "/app/hf_cache")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from oamp.dtype_policy import apply_dtype_policy

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
CACHE_DIR = "/app/hf_cache"
DEVICE = torch.device("cuda:0")


def build(weight_quant):
    lcfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                      lora_dropout=0.0,
                      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                      bias='none')
    if weight_quant == 'nf4':
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                 bnb_4bit_quant_type='nf4',
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True,
            quantization_config=bnb).to(DEVICE)
        m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=False)
    else:
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True,
            torch_dtype=torch.bfloat16).to(DEVICE)
    return get_peft_model(m, lcfg)


def find_lm_head(m):
    for name, mod in m.named_modules():
        if name.endswith('lm_head') and getattr(mod, 'weight', None) is not None:
            return mod
    return None


def check_tie(m, label):
    emb = m.get_input_embeddings()
    head = find_lm_head(m)
    if emb is None or head is None:
        print(f"  {label}: embed or lm_head not found")
        return None
    d_data = emb.weight.data_ptr() == head.weight.data_ptr()
    d_storage = emb.weight.untyped_storage().data_ptr() == head.weight.untyped_storage().data_ptr()
    ident = emb.weight is head.weight
    print(f"  {label:20s}: data_ptr={d_data}  storage_ptr={d_storage}  is_same_object={ident}")
    return d_storage


for wq in ('bf16', 'nf4'):
    print(f"\n=== {wq.upper()} ===")
    m = build(wq)
    before = check_tie(m, 'before_apply')
    m, ptrs, report = apply_dtype_policy(m, weight_quant=wq, bf16_rmsnorm=True)
    after = check_tie(m, 'after_apply')
    print(f"  report.lm_head_tied = {report['lm_head_tied']}")
    print(f"  param_ptrs len       = {len(ptrs)}")
    print(f"  TIE {'PRESERVED' if before and after else 'BROKEN' if before and not after else 'N/A'}")
    del m
    torch.cuda.empty_cache()

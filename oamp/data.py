"""Task-agnostic data loading + prompt formatting + scoring.

Spec v1 §5 (extension). GSM8K is the only backend today; the interface below
is what future tasks (MATH / HumanEval / MBPP) must implement.

Interface
---------

``load_task(task, split, ...)``           -> ``list[dict]``
``format_train_prompt(task, sample)``      -> ``str`` (target text for LoRA loss)
``format_eval_prompt(task, sample, fewshot)`` -> ``str`` (few-shot eval prompt)
``extract_answer(task, gen_text)``         -> ``str``
``is_correct(task, pred, gold)``           -> ``bool``
``default_fewshot(task)``                  -> ``list[dict]``

Sample schema (uniform across tasks)::

    {'question': str, 'answer_text': str, 'answer_number': str}

For GSM8K, ``answer_text`` is the original CoT string and ``answer_number`` is
the extracted final number.
"""

from __future__ import annotations

import random
import re
from statistics import mean
from typing import Callable, List, Optional


# ============================================================
# Task backend: GSM8K
# ============================================================

_GSM8K_FEWSHOT = [
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

# Few-shot prompt separates examples with '\n\nQ:'; stop as soon as the model
# tries to emit the next question so we don't burn tokens on hallucinated Q/A.
_GSM8K_STOP_STRINGS = ["\nQ:", "\n\nQ:"]


def _gsm8k_extract_gold(answer_str: str) -> str:
    m = re.search(r'####\s*([\-\d\.,]+)', answer_str)
    return m.group(1).replace(',', '').strip() if m else answer_str.strip()


def _gsm8k_load(split: str, cache_dir: str) -> List[dict]:
    from datasets import load_dataset
    if split not in ('train', 'test'):
        raise ValueError(f"gsm8k: split must be 'train' or 'test', got {split!r}")
    ds = load_dataset('gsm8k', 'main', cache_dir=cache_dir)
    return [
        {
            'question':      it['question'],
            'answer_text':   it['answer'],
            'answer_number': _gsm8k_extract_gold(it['answer']),
        }
        for it in ds[split]
    ]


def _gsm8k_format_train(sample: dict) -> str:
    # Matches legacy benchmark_multiseed.format_training_prompt exactly.
    return (f"Question: {sample['question']}\n"
            f"Let's solve this step by step.\n"
            f"{sample['answer_text']}")


def _gsm8k_format_eval(sample: dict, fewshot: List[dict]) -> str:
    parts = []
    for ex in fewshot:
        parts.append(f"Q: {ex['q']}\nA: {ex['a']}\n\n")
    parts.append(f"Q: {sample['question']}\nA:")
    return ''.join(parts)


def _gsm8k_extract_with_source(text: str) -> tuple:
    """Returns (pred, source). source ∈ {'answer_is','hash','fallback','empty'}.

    The 3-stage fallback matters for diagnostics: training targets end with
    '#### N' while few-shot demos end with 'The answer is N.' — a model that
    can't produce either falls through to 'last-number', which may pick up a
    calculator subexpression like <<16-3-4=9>> instead of the final answer.
    """
    m = re.search(r'[Tt]he answer is\s*([\-\d\.,]+)', text)
    if m:
        return m.group(1).replace(',', '').strip().rstrip('.'), 'answer_is'
    m = re.search(r'####\s*([\-\d\.,]+)', text)
    if m:
        return m.group(1).replace(',', '').strip(), 'hash'
    nums = re.findall(r'[\-]?\d+[\.,]?\d*', text)
    if nums:
        return nums[-1].replace(',', '').strip().rstrip('.'), 'fallback'
    return "", 'empty'


def _gsm8k_extract(text: str) -> str:
    return _gsm8k_extract_with_source(text)[0]


def _gsm8k_is_correct(pred: str, gold: str) -> bool:
    try:
        return abs(float(pred.replace(',', '').rstrip('.')) -
                   float(gold.replace(',', '').rstrip('.'))) < 0.01
    except (ValueError, AttributeError):
        return pred.strip() == gold.strip()


# ============================================================
# Backend registry
# ============================================================

class _TaskBackend:
    __slots__ = ('load', 'format_train', 'format_eval', 'extract',
                 'extract_with_source', 'is_correct', 'fewshot', 'stop_strings')

    def __init__(self, *, load, format_train, format_eval, extract,
                 extract_with_source, is_correct, fewshot, stop_strings):
        self.load = load
        self.format_train = format_train
        self.format_eval = format_eval
        self.extract = extract
        self.extract_with_source = extract_with_source
        self.is_correct = is_correct
        self.fewshot = fewshot
        self.stop_strings = stop_strings


_BACKENDS = {
    'gsm8k': _TaskBackend(
        load=_gsm8k_load,
        format_train=_gsm8k_format_train,
        format_eval=_gsm8k_format_eval,
        extract=_gsm8k_extract,
        extract_with_source=_gsm8k_extract_with_source,
        is_correct=_gsm8k_is_correct,
        fewshot=_GSM8K_FEWSHOT,
        stop_strings=_GSM8K_STOP_STRINGS,
    ),
}


def _backend(task: str) -> _TaskBackend:
    if task not in _BACKENDS:
        raise ValueError(f"Unknown task {task!r}. Available: {sorted(_BACKENDS)}")
    return _BACKENDS[task]


# ============================================================
# Public API
# ============================================================

def load_task(task: str, split: str, *,
              n_samples: int, seed: int,
              cache_dir: str,
              shuffle: bool = True) -> List[dict]:
    """Load ``n_samples`` from a task split with a deterministic seed shuffle.

    When ``n_samples`` is 0 or negative, the whole split is returned. When
    ``shuffle=False`` the natural dataset order is preserved (useful for eval
    reproducibility of Test 0a).
    """
    data = _backend(task).load(split, cache_dir)
    if shuffle:
        rng = random.Random(seed)
        data = list(data)
        rng.shuffle(data)
    if n_samples and n_samples > 0:
        data = data[:n_samples]
    return data


def format_train_prompt(task: str, sample: dict) -> str:
    return _backend(task).format_train(sample)


def format_eval_prompt(task: str, sample: dict,
                       fewshot: Optional[List[dict]] = None) -> str:
    if fewshot is None:
        fewshot = _backend(task).fewshot
    return _backend(task).format_eval(sample, fewshot)


def extract_answer(task: str, text: str) -> str:
    return _backend(task).extract(text)


def extract_answer_with_source(task: str, text: str):
    """Returns ``(pred, source)`` for diagnostics. See ``_gsm8k_extract_with_source``."""
    return _backend(task).extract_with_source(text)


def is_correct(task: str, pred: str, gold: str) -> bool:
    return _backend(task).is_correct(pred, gold)


def default_fewshot(task: str) -> List[dict]:
    return _backend(task).fewshot


def default_stop_strings(task: str) -> List[str]:
    return list(_backend(task).stop_strings)


def pack_samples_causal(samples: List[dict], tokenizer, task: str,
                        packed_seq_len: int) -> List[List[int]]:
    """Concatenate training samples with EOS and chunk to ``packed_seq_len``.

    Each sample's text (from ``format_train_prompt``) is tokenized without
    special tokens, followed by a single EOS token as delimiter. All ids are
    then chunked in order into ``packed_seq_len``-sized blocks. The trailing
    remainder is discarded (never enough for a full opt step).

    Uses causal-LM label semantics: label = input_ids (all tokens are
    predicted). No block-diagonal attention mask is emitted — samples share
    causal-only attention exactly like pretraining packed sequences. This
    keeps SDPA on the FLASH backend (INVARIANT: mask=None + is_causal=True).
    """
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no eos_token_id; cannot pack samples.")
    ids: List[int] = []
    for s in samples:
        text = format_train_prompt(task, s)
        toks = tokenizer(text, add_special_tokens=False).input_ids
        ids.extend(toks)
        ids.append(eos)
    n_chunks = len(ids) // packed_seq_len
    return [ids[i * packed_seq_len : (i + 1) * packed_seq_len] for i in range(n_chunks)]


# ============================================================
# Sequence-length statistics (schema.sequence.*)
# ============================================================

def seq_len_stats(lengths: List[int]) -> dict:
    """Compute the ``sequence.*`` fields of the result JSON.

    Returns ``{seq_len_mean, seq_len_max, seq_len_p95, tokens_total}``. Empty
    input yields zeros so a run with no training tokens still has a valid
    JSON block.
    """
    if not lengths:
        return {'seq_len_mean': 0.0, 'seq_len_max': 0,
                'seq_len_p95': 0, 'tokens_total': 0}
    sorted_lens = sorted(lengths)
    p95_idx = min(len(sorted_lens) - 1, int(round(0.95 * (len(sorted_lens) - 1))))
    return {
        'seq_len_mean': float(mean(lengths)),
        'seq_len_max':  int(max(lengths)),
        'seq_len_p95':  int(sorted_lens[p95_idx]),
        'tokens_total': int(sum(lengths)),
    }

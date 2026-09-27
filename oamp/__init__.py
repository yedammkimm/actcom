"""The oamp package.

`oamp` is the historical name of the package. The paper's method is the
backward-only compression in pack_hooks.py, selected by `--method naive_fp4`
in run_experiment.py (see the Naming note in the README). The modules the
training runs use are:

    pack_hooks     saved-tensor pack/unpack hooks: the filter chain, the INT4
                   and E2M1 body grids, FP8 and per-channel INT4 for the head views
    dtype_policy   bf16 casts and the RMSNorm swap applied after get_peft_model
    data           GSM8K loading, prompt formatting and answer scoring
    evaluate       greedy-decoding accuracy loop
    env, schema    environment snapshot and the result JSON schema
    sdpa_utils     helpers that keep attention on the flash backend

bilevel, quantize, selector, memory and cuda_ops are an earlier, standalone
implementation of the max-abs anchor idea: they quantize the activations
themselves in the forward pass through a straight-through estimator, with
optional CUDA kernels from oamp_cuda/. Nothing in the training path imports
them.
"""

from .version import __version__, __author__, __email__

__all__ = ["__version__", "__author__", "__email__"]

# Task Accuracy Cannot Validate Activation Compression

Code, run records, and paper source for

> **Task Accuracy Cannot Validate Activation Compression: Run-to-Run Variance and a Five-Minute Gradient Probe**
> Yedam Kim (Sorbonne University), Shinya Takamaeda-Yamazaki (CASYS Lab, The University of Tokyo)

The compiled paper is [docs/OAMP/OAMP_paper.pdf](docs/OAMP/OAMP_paper.pdf); the LaTeX source is in the same directory.

## What the paper shows

Fine-tuning a 70B model on a single 128 GB device requires compressing the activations stored for the backward pass. Prior work certifies such a method by an in-distribution task-accuracy check. We show that this check does not see the failure four-bit compression actually produces:

- Compressing every stored tensor to four bits, four of eight otherwise identical Llama-3.2-3B runs end with held-out perplexity 5 to 15% above the uncompressed baseline while the other four match it. The effect is in the dispersion, not the mean (variance ratio at exact permutation p = 0.03; difference in means at p = 0.10).
- Which runs fail is a property of the run, not the seed: re-running a damaged configuration produced a safe model twice out of two.
- Nine measurements fail to detect it, including task accuracy, training loss, in-distribution perplexity, gradient norms, and five multiple-choice benchmarks over 21 adapters, including MMLU.
- The damage follows gradient fidelity, not the bit budget. The sensitive tensors are the query and key head views only: gradient cosine falls to 0.36 there and stays above 0.99 everywhere else. A five-minute forward probe identifies them before any training run.
- Storing those two tensors in FP8 removes the failure in eight of eight runs at 0.60 bits per element at 3B (0.44 at 70B). Scaling them along the channel axis instead costs no extra bits but leaves one run of eight above the threshold.

## Repository layout

```
oamp/                  Python package: saved-tensor pack/unpack hooks, INT4 and E2M1
                       body quantizers, rank-4 dispatch to FP8 or per-channel INT4,
                       filter order, dtype policy, evaluation
oamp_cuda/             CUDA / C++ packing kernels (optional; a Python fallback exists)
run_experiment.py      Single entry point for every training, accuracy and memory run
configs/               Fixed experiment settings (Table 17 of the paper)
scripts/               Run chains behind each experiment, and scripts/audit/ which
                       recomputes every reported statistic from results/
results/               Run-level records: one JSON per run, plus perplexity, MMLU and
                       probe outputs (Appendices B and D point here)
paper_data/MANIFEST.md Maps every number in the paper to the file and script it comes from
docs/PREREG_*.md       Pre-registration documents cited in Appendix E
docs/development_history.txt
                       Commit log of the private development repository (see below)
docs/OAMP/             LaTeX source, references, compiled PDF
figures/               Figures used by the paper
logs/                  The base-model perplexity diagnostic run cited in Section 5.4
```

## Installation

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -e .            # Python package; the CUDA extension builds if nvcc is present
```

Experiments were run with PyTorch 2.x, CUDA 12.x and Python 3.10 on a single NVIDIA GB10 (128 GB unified memory). Models are read from `$HF_HOME`.

## Reproducing the reported statistics

Most statistics in the paper are recomputed from the files under `results/`; `paper_data/MANIFEST.md` lists, for each number, the file it is read from and the script that reads it. One group is not reproducible from this repository alone: the same-seed pair measurements of Section 5.4, which read the LoRA adapters themselves. MMLU was evaluated only for the blockwise arm, the base model and two INT4 runs (the 22-adapter chain was not resumed; see `results/mc_mmlu_launch.log`), and the paper reports MMLU statistics for those cells only. The main entry points are

```bash
python scripts/make_appendix_tables.py     # Table 23: every run behind the aggregates
python scripts/audit/mmlu_stats.py         # Table 25: every MMLU statistic
python scripts/prereg_loo.py               # Table 24: leave-one-arm-out prediction
python scripts/audit/_audit_numbers.py     # remaining statistics of Sections 5.3 and 5.4
```

Arm membership is always resolved from (method, body encoding, rank-4 mode) jointly, and accuracy is always recomputed as `100 * n_correct / n`; Appendix F of the paper states the four rules the scripts follow.

Re-running the training itself goes through `run_experiment.py`; the shell scripts under `scripts/` are the chains that produced each block of `results/`, kept as they ran (they contain the paths of the machine they ran on). Two of them carry pre-registered quantities in their headers: `scripts/run_axis2_a4dfp4_seeds.sh` (the damage threshold) and `scripts/run_mc_chain.sh` (the MMLU tests).

## Development history

This repository was published as a single commit. The paper's Appendix E cites commit hashes and timestamps from the private development repository in which the experiments were run; that history is not carried here. Its full commit log, with hashes, dates and messages, is archived in `docs/development_history.txt`.

## Citation

```bibtex
@misc{kim2026taskaccuracy,
  title  = {Task Accuracy Cannot Validate Activation Compression:
            Run-to-Run Variance and a Five-Minute Gradient Probe},
  author = {Kim, Yedam and Takamaeda-Yamazaki, Shinya},
  year   = {2026},
}
```

## License

Apache License 2.0 (see `LICENSE`). The fine-tuned adapters are not part of this repository; they are packaged separately by `scripts/audit/make_adapter_archive.py` under the licenses of the base models.

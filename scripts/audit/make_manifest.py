"""Write paper_data/MANIFEST.md: every number in the paper, the script that
computes it, the files it is read from, and their sha256.

Nothing is copied. Each entry points at the file where the experiment wrote it,
so a script keeps one path and the manifest is a checksummed index of the tree
as committed. It replaces results/_INDEX.md as the paper's index.

Usage: python scripts/audit/make_manifest.py   (standard library only)
"""
import glob
import hashlib
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'paper_data', 'MANIFEST.md')

# (title, where the numbers appear, scripts, file patterns)
GROUPS = [
    ('Held-out perplexity: WikiText-2, GSM8K test, NarrativeQA, GovReport',
     'tab:allruns, tab:dose_summary, tab:threecorpora, tab:2x2, tab:rerun, tab:recompute; '
     'sections 1, 5.3, 5.4, 5.6; the damage counts 3/3, 4/8, 1/8, 0/8, 0/2 at tau = 16.65',
     ['scripts/make_appendix_tables.py', 'scripts/audit/_check_levels.py', 'scripts/audit/_audit_numbers.py',
      'scripts/audit/_bf_definitive.py (perplexities transcribed from these files)'],
     ['results/ppl_eval/*.json', 'results/ppl_axis2/*.json', 'results/ppl_axis3/*.json', 'results/ppl_axis4/*.json',
      'results/ppl_axis6/*.json', 'results/ppl_axis7/*.json', 'results/ppl_ood_extra/*.json', 'results/ppl_70b/*.json']),
    ('GSM8K accuracy under the unified re-evaluation (protocol c)',
     'tab:accuracy, tab:allruns (acc column), tab:anchor_acc (70B row); sections 1, 5.5, 5.6 '
     '(53.25 vs 54.50, p = 0.51); accuracy is 100 * n_correct / n_samples, never the stored percentage',
     ['scripts/build_result_index.py', 'scripts/make_appendix_tables.py'],
     ['results/reeval_stop_mnt512_n200/*.json', 'results/reeval_stop_mnt512/*.json',
      'results/70b_accuracy/*reeval*.json', 'results/70b_accuracy/base_eval*.json']),
    ('Training runs: step losses, gradient-norm traces, non-finite counters, in-training accuracy, pack statistics',
     'sections 3.2 (15.04 % four-dimensional share, byte_bits), 5.1 (586, 0.62/0.61/0.59), 5.6 '
     '(last-500-step losses, tab:graddetect), tab:rerun (first three losses), tab:anchor_acc (3B row), '
     'Figure 2 (Fig. 4 in-training panels)',
     ['scripts/build_result_index.py (loss over the last 500 optimizer steps from the .checkpoints.jsonl sidecars)',
      'scripts/audit/_loss_divergence.py'],
     ['results/naive4bit_e2m1/*.json', 'results/axis2_seeds/*.json', 'results/axis3_b4dfp8/*.json',
      'results/axis4_rerun/*.json', 'results/pilot_e2m1/*.json', 'results/pilot_e2m1_seeds/*.json',
      'results/pilot_pack4d_fp8_matrix/*.json', 'results/naive4bit_int4/*.json', 'results/axis6_int4_fp8/*.json',
      'results/axis7_chan_int4/*.json', 'results/70b_accuracy/accuracy__*.json', 'results/70b_accuracy_e2m1/*.json',
      'results/**/*.checkpoints.jsonl']),
    ('Multiple-choice per-item records (ARC-C, ARC-E, PIQA, WinoGrande, HellaSwag), 21 adapters and the base model',
     'tab:churn, tab:levels; section 5.5 (7.39 %, 6.59 %, 6,653 items, p = 0.001)',
     ['scripts/audit/_check_levels.py', 'scripts/audit/_pa_diag.py', 'scripts/analyze_mc.py'],
     ['results/mc_downstream/*.json']),
    ('MMLU 5-shot, the pre-registered primary metric (scripts/run_mc_chain.sh header, commit 27ae5b6)',
     'section 5.4 and tab:levels (MMLU row) once arm B has landed; results/audit/mmlu_stats_<date>.json holds the '
     'arm-A primary Spearman, the damaged-vs-safe interval, the sign test, item churn and the run-level Mantel test',
     ['scripts/audit/mmlu_stats.py', 'scripts/run_mc_chain.sh', 'scripts/run_mmlu_chain.sh', 'scripts/run_mmlu_chain_full.sh',
      'scripts/eval_multichoice.py'],
     ['results/mc_mmlu/*.json', 'results/mc_mmlu_launch.log', 'results/audit/mmlu_stats_*.json']),
    ('LoRA endpoint geometry: row-space principal angles and Frobenius distances',
     'tab:subspace, tab:levels; sections 5.1 (arm E, 0.1339 vs 0.1314, z = 5.9-7.1, norms 24.5 vs 31.0), '
     '5.4 (0.061, 0.82, 28/65/80/93 %, same-seed pairs); Appendix E tab:prereg_loo',
     ['scripts/weight_distance.py', 'scripts/pa_all_arms.py', 'scripts/prereg_loo.py',
      'scripts/audit/_pa_checks.py', 'scripts/audit/_pa_trend.py', 'scripts/audit/_pa_trend2.py',
      'scripts/audit/_pa_diag.py', 'scripts/audit/_arm_e_check.py', 'scripts/audit/_arm_e_diagnose.py',
      'scripts/audit/_norm_normalise.py', 'scripts/audit/_sameseed_pairs.py'],
     ['results/_wd/*']),
    ('70B memory and step time',
     'tab:70b_memory, tab:70b_time, Figure 1; sections 1, 4.2, 4.3, 6 (slope 0.02884 GB/token)',
     ['scripts/make_fig1_vram.py'],
     ['results/70b_mem_sweep/*.json', 'results/70b_mem_sweep_e2m1/*.json', 'results/70b_mem_chan_int4/*.json',
      'results/70b_gc/*.json', 'results/70b_oom/*.json', 'results/70b_steptime/*.json', 'results/70b_std_slope/*.json']),
    ('3B memory and step time',
     'tab:gc, tab:dtype; section 3.4 (2.82 and 5.72 GB), section 4.3',
     [],
     ['results/matrix_nf4/*.json', 'results/dtype_decomp/*.json']),
    ('Per-role gradient probe',
     'Figure 3, tab:layerdepth, tab:dose_x (cosines 0.371/0.587/0.910/0.992); sections 5.1 (0.358, 0.992, '
     '0.226, rotary 0.9997), 5.2 (dynamic range 9362/125), Appendix C (43-49 of 56 query views)',
     ['scripts/make_fig3_per_role.py', 'oamp_train_engine/audit/gradient_error_probe.py'],
     ['results/audit/mechanism_int4_vs_e2m1_20260831_080053.json', 'results/audit/dose_response_qk_20260904_191655.json',
      'results/audit/dose_response_prod_20260904_192407.json', 'results/audit/rotary_only_e2m1_step0_*.json',
      'results/coverage_probe/*.json', 'results/gradient_error_probe_2026081*.json']),
    ('Anchor ablation: max-abs against gradient-norm ranking',
     'section 5.2 (Spearman 0.13, top-20 % overlap 0.248)',
     [],
     ['results/gradient_correlation_20260329_000456.json']),
    ('Qwen2.5-3B non-finite safeguard',
     'sections 3.5, 5.4, 5.5, Appendix A (21 skipped steps, 42.4 % vs 68.2 %, 40.0 % vs 70.8 %)',
     [],
     ['results/qwen_safeguard_control/*.json', 'results/qwen3b_skip_check/*.json', 'results/qwen3b_e2m1/*.json',
      'results/qwen3b_e2m1_k040/*.json', 'results/qwen3b_k_sweep/*.json', 'results/pilot_qwen3b_pack4d_fp8/*.json']),
    ('Rank census, filter shares, base-model perplexity',
     'section 3.2 footnote (1,064 tensors; 224/392/448), section 3.3 (22.7/32.2/68.5/55.3/12.3/8.7 %), '
     'section 5.4 footnote and 5.6 (16.19, re-evaluation to four decimals)',
     ['scripts/audit/_probe_rank_distribution.py', 'scripts/audit/_measure_4d_composition.py',
      'scripts/audit/_diag_wikitext_ppl.py'],
     ['results/rank_probe/rank.log', 'results/audit/shape_coverage_dropout05_094917.log', 'logs/diag_wikitext_ppl.log']),
    ('Figures as included by docs/OAMP/OAMP_paper.tex',
     'Figures 1-4 and the method figure',
     ['scripts/make_fig1_vram.py', 'scripts/make_fig2_dose_response.py', 'scripts/make_fig3_per_role.py',
      'scripts/make_fig4_insensitivity.py', 'scripts/make_fig_method.py'],
     ['figures/*.pdf']),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    tracked = set(subprocess.run(['git', 'ls-files'], cwd=ROOT, capture_output=True, text=True).stdout.split('\n'))
    head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    lines = ['# Paper data manifest', '',
             f'Generated by `scripts/audit/make_manifest.py` at {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())} '
             f'from the tree at commit `{head}`.', '',
             'Every number in `docs/OAMP/OAMP_paper.tex` is read from a file listed here by the script named in the same '
             'section. Nothing is copied: each path is the one the experiment wrote, and each script keeps that one path. '
             'This manifest replaces `results/_INDEX.md` as the index for the paper. A file marked *not tracked* exists in '
             'the working tree but not in the repository; the file tables list tracked files only. Arm membership is always resolved from '
             '(method, body_encoding, pack_4d_mode) jointly, and accuracy always as 100 * n_correct / n_samples.', '']
    total = 0
    untracked_matches = []
    for i, (title, where, scripts, patterns) in enumerate(GROUPS, 1):
        matched = sorted({f for p in patterns for f in glob.glob(os.path.join(ROOT, p), recursive=True)
                          if os.path.isfile(f) and 'adapter_config' not in f})
        # Only tracked files are listed, so the manifest is identical in a clone;
        # untracked matches are collected and reported at the end.
        files = [f for f in matched if os.path.relpath(f, ROOT) in tracked]
        untracked_matches += [os.path.relpath(f, ROOT) for f in matched if os.path.relpath(f, ROOT) not in tracked]
        lines += [f'## {i}. {title}', '', f'**Where:** {where}', '']
        if scripts:
            lines += ['**Scripts:** ' + ', '.join(
                f'`{s}`' + ('' if s.split(' ')[0] in tracked else ' *(not tracked)*') for s in scripts), '']
        lines += [f'**Files:** {len(files)}', '', '| file | bytes | sha256 |', '|---|---:|---|']
        for f in files:
            rel = os.path.relpath(f, ROOT)
            lines.append(f'| `{rel}` | {os.path.getsize(f):,} | `{sha256(f)}` |')
        lines.append('')
        total += len(files)
    lines += ['## Inputs that are not in the repository', '',
              '- The LoRA adapters (`results/**/*_adapter/adapter_model.safetensors`, about 2.3 GB for the 85 '
              'adapter directories in the working tree of 2026-09-08). They are the inputs of '
              '`scripts/weight_distance.py`, `scripts/pa_all_arms.py` and `scripts/audit/_check_levels.py`; their '
              'outputs in `results/_wd/` are tracked, so everything downstream of the adapters reproduces from a clone.',
              '- Model weights and the HF cache.', '']
    lines += [f'Total files listed: {total}.']
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(lines) + '\n')
    print(f'wrote {OUT}: {total} files in {len(GROUPS)} groups')
    # Untracked matches depend on the working tree, so they go to stdout, not into the file.
    for u in sorted(set(untracked_matches)):
        print(f'  matched by a pattern but not tracked (not listed): {u}')


if __name__ == '__main__':
    main()

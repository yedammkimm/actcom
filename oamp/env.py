"""Snapshot of the software and hardware a run executed on.

collect_env returns the fields that oamp.schema.validate_schema requires:
Python, torch, CUDA and library versions, the GPU name and compute
capability, the driver, and the git commit with its dirty flag. All of it
can be gathered before any GPU work. A dirty git tree produces a warning,
because a result should be tied to a commit.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)


def _run(cmd, cwd=None, env=None) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=cwd, env=env,
                                      stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ''


def _git(cwd, *args) -> str:
    """Run git with safe.directory supplied through GIT_CONFIG_* variables.

    git 2.34 ignores `-c safe.directory=...` on the command line when the value
    does not come from a config file; GIT_CONFIG_COUNT is the portable route
    documented in the git manual.
    """
    env = os.environ.copy()
    if cwd is not None:
        # Chain any pre-existing GIT_CONFIG_COUNT entries so we don't clobber them.
        base_n = int(env.get('GIT_CONFIG_COUNT', '0') or '0')
        env['GIT_CONFIG_COUNT'] = str(base_n + 1)
        env[f'GIT_CONFIG_KEY_{base_n}']   = 'safe.directory'
        env[f'GIT_CONFIG_VALUE_{base_n}'] = cwd
    return _run(['git', *args], cwd=cwd, env=env)


def _git_commit(cwd) -> str:
    return _git(cwd, 'rev-parse', 'HEAD')


def _git_dirty(cwd) -> bool:
    if not shutil.which('git'):
        return False
    porcelain = _git(cwd, 'status', '--porcelain')
    return bool(porcelain)


def _pkg_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, '__version__', 'unknown')
    except ImportError:
        return 'not_installed'


def _nvidia_driver() -> str:
    if not shutil.which('nvidia-smi'):
        return ''
    return _run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']).split('\n')[0]


def collect_env(project_root: str | None = None, *, warn_on_dirty: bool = True) -> dict:
    """Return the environment snapshot. `project_root` is where git runs (default:
    the parent of this package); with `warn_on_dirty` a non-empty
    `git status --porcelain` logs a warning.
    """
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # torch / cuda info; imported lazily so this module stays cheap without torch
    try:
        import torch
        torch_version = torch.__version__
        cuda_version = torch.version.cuda or ''
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            compute_capability = f'{cc[0]}.{cc[1]}'
        else:
            gpu_name = ''
            compute_capability = ''
    except ImportError:
        torch_version = cuda_version = gpu_name = compute_capability = 'not_installed'

    git_commit = _git_commit(project_root)
    git_dirty = _git_dirty(project_root)
    if git_dirty and warn_on_dirty:
        logger.warning(
            "collect_env: git tree is dirty. Full-scale re-runs must land on a clean commit. "
            "Current HEAD=%s", git_commit[:12] or '<no-repo>')

    return {
        'hostname':                platform.node(),
        'python_version':          sys.version.split()[0],
        'gpu_name':                gpu_name,
        'compute_capability':      compute_capability,
        'driver_version':          _nvidia_driver(),
        'cuda_version':            cuda_version,
        'torch_version':           torch_version,
        'transformers_version':    _pkg_version('transformers'),
        'peft_version':            _pkg_version('peft'),
        'bitsandbytes_version':    _pkg_version('bitsandbytes'),
        'torchao_version':         _pkg_version('torchao'),
        'git_commit':              git_commit,
        'git_dirty':               git_dirty,
        'timestamp_start':         '',    # filled by run_experiment
        'timestamp_end':           '',    # filled by run_experiment
    }


REQUIRED_ENV_FIELDS = frozenset({
    'hostname', 'gpu_name', 'compute_capability', 'driver_version',
    'cuda_version', 'torch_version', 'transformers_version', 'peft_version',
    'bitsandbytes_version', 'git_commit', 'git_dirty',
    'timestamp_start', 'timestamp_end',
})

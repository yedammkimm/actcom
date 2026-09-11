"""Environment capture — snapshot of the exact stack a run executed on.

Spec v1 §6.1. All fields are safe to call before GPU work so ``env`` can be
fed to :func:`oamp.schema.validate_schema` on the CPU-only path.

Emits a warning if the git tree is dirty. Full-scale re-runs must land on a
clean commit so results can be tied to a SHA.
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
    """Run git with safe.directory injected via GIT_CONFIG_* env vars.

    Older git (2.34) ignores ``-c safe.directory=...`` from the command line
    when the value doesn't come from a config file. GIT_CONFIG_COUNT is the
    portable path documented in git's manpage.
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
    """Return the environment snapshot required by :func:`schema.validate_schema`.

    Parameters
    ----------
    project_root : Optional[str]
        Directory where ``git rev-parse`` should run. Defaults to the parent
        of this file's package.
    warn_on_dirty : bool
        Emit a logger.warning when ``git status --porcelain`` is non-empty.
    """
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # torch / cuda info — import lazily so this module stays cheap without torch
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
            "collect_env: git tree is dirty. Full-scale re-runs must land on a clean commit "
            "(see spec v1 §6.1). Current HEAD=%s", git_commit[:12] or '<no-repo>')

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

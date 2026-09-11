"""OAMP package — unified pipeline (post-2026-08 audit).

The legacy α-path (``GenericOAMPWrapper``, ``NaiveFP4Wrapper``) and every
``benchmark_*.py`` file that produced paper v1 results were moved to
``legacy/`` during the 2026-08 audit. Do not import from ``legacy`` in this
package; copy code across if reuse is needed. See ``legacy/README.md``.

Planned modules (see docs / spec v1):
  oamp.dtype_policy — Arm 4 dtype normalization (BF16RMSNorm swap)
  oamp.pack_hooks   — Single PackHooks class covering all methods
  oamp.sdpa_utils   — SDPA backend helpers (kept; see oamp_train_engine/benchmarks/)
  oamp.schema       — Result JSON schema + validate_schema()
  oamp.env          — Environment capture
  oamp.data         — GSM8K loading / ordering
  oamp.evaluate     — GSM8K accuracy eval (ported from test0a)
"""

from .version import __version__, __author__, __email__

__all__ = ["__version__", "__author__", "__email__"]

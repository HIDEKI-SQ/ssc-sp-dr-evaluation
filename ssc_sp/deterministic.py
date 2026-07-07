"""Deterministic execution contract.

Pins BLAS threading to one and provides a single entry point to fix seeds, so
that repeated runs on the recorded environment reproduce identical outputs.
The recorded reference environment is Python 3.10.19, numpy 1.24.3,
scipy 1.10.1, scikit-learn 1.3.0, with OPENBLAS/MKL/OMP threads set to 1.
"""

from __future__ import annotations

import os

_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")


def enforce_single_thread_blas() -> None:
    for var in _THREAD_VARS:
        os.environ[var] = "1"


def set_seed(seed: int) -> "np.random.Generator":
    import numpy as np
    return np.random.default_rng(seed)


def environment_report() -> dict:
    import platform
    import numpy as np
    import scipy
    import sklearn
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "blas_threads": {v: os.environ.get(v) for v in _THREAD_VARS},
    }


enforce_single_thread_blas()

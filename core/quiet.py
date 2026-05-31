import contextlib
import logging
import os
import sys
import warnings

for k, v in {
    "NEMO_TESTING": "0", "HYDRA_FULL_ERROR": "0",
    "TRANSFORMERS_VERBOSITY": "error", "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "PYTHONWARNINGS": "ignore", "TQDM_DISABLE": "1", "ONE_LOGGER_DISABLED": "1",
}.items():
    os.environ[k] = v

warnings.filterwarnings("ignore")

for name in ("nemo", "nemo_logger", "NeMo", "pytorch_lightning", "lightning"):
    log = logging.getLogger(name)
    log.setLevel(logging.CRITICAL)
    log.propagate = False


@contextlib.contextmanager
def muted():
    sys.stdout.flush()
    sys.stderr.flush()
    out, err = os.dup(1), os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.dup2(null, 2)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(out, 1)
        os.dup2(err, 2)
        for fd in (out, err, null):
            os.close(fd)

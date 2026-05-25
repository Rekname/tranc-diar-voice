import os
import sys
import logging
import warnings
import contextlib

os.environ["NEMO_TESTING"] = "0"
os.environ["HYDRA_FULL_ERROR"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TQDM_DISABLE"] = "1"
os.environ["ONE_LOGGER_DISABLED"] = "1"

warnings.filterwarnings("ignore")

for name in ("nemo", "nemo_logger", "NeMo", "pytorch_lightning", "lightning"):
    logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger(name).propagate = False


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


def silence_nemo() -> None:
    try:
        from nemo.utils import logging as nemo_logging
        nemo_logging.setLevel(logging.CRITICAL)
    except Exception:
        pass
    try:
        from pytorch_lightning.utilities import rank_zero
        rank_zero.rank_zero_info = lambda *a, **k: None
        rank_zero.rank_zero_warn = lambda *a, **k: None
    except Exception:
        pass

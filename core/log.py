import time
import contextlib
from typing import Callable, Optional

StageCallback = Callable[[str, str, float], None]


@contextlib.contextmanager
def stage(name: str, on_event: Optional[StageCallback] = None):
    if on_event:
        on_event(name, "start", 0.0)
    t0 = time.time()
    try:
        yield
    finally:
        if on_event:
            on_event(name, "done", time.time() - t0)

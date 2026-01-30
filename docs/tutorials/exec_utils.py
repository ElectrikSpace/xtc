from typing_extensions import override
import sys
import io
import traceback
import multiprocessing as mp

def _child_exec(code: str, out_q: "mp.Queue"):
    """Runs in a separate process."""
    class QWriter(io.TextIOBase):
        @override
        def write(self, s: str) -> int:
            if s:
                out_q.put(("chunk", s))
            return len(s)

        @override
        def flush(self):
            return None

    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = QWriter()
        sys.stderr = QWriter()

        # A tiny "sandbox" globals dict; you can expand/lock this down.
        env = {"__name__": "__sandbox__"}

        exec(code, env, env)
        out_q.put(("done", None))
    except BaseException:
        out_q.put(("chunk", "\n" + traceback.format_exc()))
        out_q.put(("done", None))
    finally:
        sys.stdout, sys.stderr = old_out, old_err

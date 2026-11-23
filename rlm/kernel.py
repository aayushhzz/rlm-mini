"""Host side of the persistent IPython kernel (jupyter_client). Not a sandbox: code runs as you."""
import os
import queue
import re
import time

from jupyter_client.manager import KernelManager

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class Kernel:
    def __init__(self, cwd, timeout=None, env=None):
        self.timeout = float(timeout or os.environ.get("RLM_CELL_TIMEOUT", 240))
        self.km = KernelManager(kernel_name="python3")
        self.km.start_kernel(cwd=str(cwd), env=dict(os.environ, **(env or {})))
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=60)

    def execute(self, code):
        """Run one cell; return its text output (stdout, results, tracebacks)."""
        msg_id = self.kc.execute(code, store_history=True)
        out, deadline, timed_out = [], time.time() + self.timeout, False
        while True:
            left = deadline - time.time()
            if left <= 0 and not timed_out:
                timed_out = True
                self.km.interrupt_kernel()
                deadline = time.time() + 10  # give the interrupt time to land
                continue
            try:
                msg = self.kc.get_iopub_msg(timeout=max(left, 0.1))
            except queue.Empty:
                if timed_out:
                    break
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            t, c = msg["msg_type"], msg["content"]
            if t == "stream":
                out.append(c["text"])
            elif t in ("execute_result", "display_data"):
                out.append(c["data"].get("text/plain", "") + "\n")
            elif t == "error":
                out.append(_ANSI.sub("", "\n".join(c["traceback"])) + "\n")
            elif t == "status" and c["execution_state"] == "idle":
                break
        text = "".join(out)
        if timed_out:
            text += "\n[cell timed out after %ss and was interrupted]" % int(self.timeout)
        return text

    def shutdown(self):
        try:
            self.kc.stop_channels()
        finally:
            self.km.shutdown_kernel(now=True)

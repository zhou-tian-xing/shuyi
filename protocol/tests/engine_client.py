"""A minimal client for the 数弈 engine protocol (protocol/ENGINE_PROTOCOL.md).

It is deliberately small and dependency free: the protocol is line based, every
command may carry an id and every reply repeats it, so a test (or another front
end written in Python) can drive an engine without any shared code.

    engine = EngineProcess(path)
    reply = engine.command("analyze red 128")
    reply.ok, reply.text, reply.infos

`analyze` streams until it is finished or stopped, so it has no immediate reply:
use `send` + `wait_reply` when another command (typically `stop`) has to go out
while it is still running.  Replies that arrive out of order are buffered, which
is exactly what a front end has to do: a `stop` can overtake the `stopped` reply
of the analysis it interrupted.
"""
import json
import queue
import subprocess
import threading
import time


class Reply:
    def __init__(self, ok, text, infos):
        self.ok = ok
        self.text = text
        self.infos = infos

    def __repr__(self):
        return f"Reply(ok={self.ok}, text={self.text!r}, infos={len(self.infos)})"


class EngineProcess:
    def __init__(self, binary, cwd=None):
        self.proc = subprocess.Popen(
            [str(binary)],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.lines = queue.Queue()
        self.stderr = []
        self._next_id = 0
        self._pending = {}
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._err_reader, daemon=True).start()

    def _reader(self):
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\n"))

    def _err_reader(self):
        for line in self.proc.stderr:
            self.stderr.append(line.rstrip("\n"))

    def send(self, text):
        """Send a command without waiting; returns its id for `wait_reply`."""
        self._next_id += 1
        cid = str(self._next_id)
        assert "\n" not in text
        self.proc.stdin.write(f"{cid} {text}\n")
        self.proc.stdin.flush()
        return cid

    def _consume(self, cid, line, infos):
        if line.startswith(f"{cid} info "):
            infos.append(json.loads(line[len(f"{cid} info "):]))
            return None
        if line.startswith(f"{cid} ="):
            return Reply(True, line[len(cid) + 2:].strip(), infos)
        if line.startswith(f"{cid} ?"):
            return Reply(False, line[len(cid) + 2:].strip(), infos)
        return None

    def wait_reply(self, cid, timeout=120.0):
        """Collect the command's `info` lines, then its final reply."""
        infos = []
        deadline = time.time() + timeout
        while True:
            buffered = self._pending.get(cid)
            while buffered:
                reply = self._consume(cid, buffered.pop(0), infos)
                if reply:
                    return reply
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"no reply to command {cid}; stderr={self.stderr[-5:]}")
            try:
                line = self.lines.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                continue
            line_id = line.split(" ", 1)[0]
            if line_id.isdigit() and line_id != cid:
                self._pending.setdefault(line_id, []).append(line)
                continue
            reply = self._consume(cid, line, infos)
            if reply:
                return reply

    def command(self, text, timeout=120.0):
        """Send one command and wait for its final reply."""
        return self.wait_reply(self.send(text), timeout)

    def close(self):
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 - closing must never mask a test result
            self.proc.kill()
            self.proc.wait(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

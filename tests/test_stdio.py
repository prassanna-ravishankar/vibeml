"""The stdio transport owns stdout, even when SkyPilot writes to it."""

import json
import queue
import subprocess
import sys
import textwrap
import threading
import time
from typing import IO, Any

# Runs the real server with a fake SDK that writes to stdout the three ways
# SkyPilot does: call-time print(), a logging handler bound to sys.stdout at
# import, and raw writes to file descriptor 1.
NOISY_SERVER = textwrap.dedent(
    """
    import logging, os, sys
    from unittest.mock import MagicMock

    import sky
    from vibeml import server
    from vibeml.cloud import skypilot

    noisy = logging.getLogger("fake-sky")
    noisy.addHandler(logging.StreamHandler(sys.stdout))
    noisy.propagate = False

    def download_logs(name, job_ids):
        print("NOISE-print")
        noisy.warning("NOISE-handler")
        os.write(1, b"NOISE-fd\\n")
        return {"1": "/logs/" + name}

    fake = MagicMock()
    fake.ClusterStatus = sky.ClusterStatus
    fake.StatusRefreshMode = sky.StatusRefreshMode
    fake.get.return_value = [
        MagicMock(name="record", launched_at=None, status=sky.ClusterStatus.UP)
    ]
    fake.get.return_value[0].name = "vibeml-a"
    fake.download_logs.side_effect = download_logs
    skypilot.sky = fake

    server.run_server()
    """
)


TIMEOUT = 30


def _lines_into(stream: IO[str], lines: "queue.Queue[str]") -> None:
    """Forward every line of a stream into a queue."""
    for line in stream:
        lines.put(line)


def _read_into(stream: IO[str], chunks: list[str]) -> None:
    """Read a stream to EOF into a list."""
    chunks.append(stream.read())


def rpc(id_: int | None, method: str, params: dict[str, Any]) -> str:
    """Encode one JSON-RPC message as a line."""
    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
    if id_ is not None:
        message["id"] = id_
    return json.dumps(message) + "\n"


def test_sdk_output_never_reaches_the_protocol_stream() -> None:
    """Every stdout line is JSON-RPC; SDK noise is diverted to stderr."""
    requests = "".join(
        [
            rpc(
                1,
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            ),
            rpc(None, "notifications/initialized", {}),
            rpc(
                2,
                "tools/call",
                {"name": "stop_training", "arguments": {"cluster_id": "vibeml-a"}},
            ),
        ]
    )

    proc = subprocess.Popen(  # noqa: S603 - runs this interpreter on a fixed script
        [sys.executable, "-c", NOISY_SERVER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert (
        proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    )
    # Drain both pipes on threads so a stalled server fails at the deadline
    # instead of hanging, and stderr can never back-pressure the server.
    stdout_lines: queue.Queue[str] = queue.Queue()
    stderr_chunks: list[str] = []
    threading.Thread(
        target=_lines_into, args=(proc.stdout, stdout_lines), daemon=True
    ).start()
    stderr_reader = threading.Thread(
        target=_read_into, args=(proc.stderr, stderr_chunks), daemon=True
    )
    stderr_reader.start()

    responses: dict[int, Any] = {}
    deadline = time.monotonic() + TIMEOUT
    try:
        proc.stdin.write(requests)
        proc.stdin.flush()
        while 2 not in responses:
            line = stdout_lines.get(timeout=max(deadline - time.monotonic(), 0))
            message = json.loads(line)  # fails if anything but JSON-RPC is on stdout
            if "id" in message:
                responses[message["id"]] = message
        proc.stdin.close()
        proc.wait(timeout=TIMEOUT)
    finally:
        proc.kill()
        proc.wait()
    stderr_reader.join(timeout=TIMEOUT)
    stderr = "".join(stderr_chunks)

    result = responses[2]["result"]["structuredContent"]
    assert result["status"] == "stopped"
    assert result["log_paths"] == {"1": "/logs/vibeml-a"}
    for marker in ("NOISE-print", "NOISE-handler", "NOISE-fd"):
        assert marker in stderr

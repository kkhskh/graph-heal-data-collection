#!/usr/bin/env python3
"""Host-layer fault-injection stub for unit tests & future stress-ng integration."""

from __future__ import annotations

import argparse
import logging
import os
import platform
import subprocess
from typing import List

logger = logging.getLogger("host_fault")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0  # type: ignore[attr-defined]


def _build_stress_cmd(cpu: int, mem: str) -> List[str]:  # pragma: no cover
    return ["stress", "--cpu", str(cpu), "--vm", "1", "--vm-bytes", mem, "--timeout", "30s"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Inject synthetic CPU / memory load on the host machine.")
    p.add_argument("--cpu", type=int, default=2, help="Number of CPU workers to spawn (default: 2)")
    p.add_argument("--mem", default="1G", help="Total memory stress e.g. 512M / 1G (default: 1G)")
    args = p.parse_args(argv)

    if platform.system().lower() != "linux" or not _is_root():
        logger.warning("Host fault injection requires root privileges on Linux – stubbed no-op")
        return 0

    cmd = _build_stress_cmd(args.cpu, args.mem)  # pragma: no cover
    logger.info("Executing host fault: %s", " ".join(cmd))  # pragma: no cover

    try:  # pragma: no cover
        subprocess.run(cmd, check=True)
    except FileNotFoundError:  # pragma: no cover
        logger.error("'stress' command not found – please install stress-ng")
        return 1
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        logger.error("stress command failed with exit code %s", exc.returncode)
        return exc.returncode

    logger.info("Host fault injection complete")  # pragma: no cover
    return 0  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main()) 
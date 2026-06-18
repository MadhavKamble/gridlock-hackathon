from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from lib.paths import ROOT

SUMO_CONFIG_DIR = ROOT / "config" / "sumo_config"
SUMO_NET_PATH = SUMO_CONFIG_DIR / "bangalore.net.xml"


def sumo_capability_report() -> dict[str, Any]:
    """Return local SUMO availability without making SUMO a hard dependency."""
    binaries = {name: shutil.which(name) for name in ("sumo", "netconvert", "duarouter")}
    return {
        "available": all(binaries.values()) and SUMO_NET_PATH.exists(),
        "binaries": binaries,
        "network_path": str(SUMO_NET_PATH),
        "network_exists": SUMO_NET_PATH.exists(),
        "note": (
            "SUMO automation runs when sumo, netconvert, duarouter, and config/sumo_config/bangalore.net.xml exist. "
            "The graph-based Bernoulli simulator is used otherwise."
        ),
    }


def run_sumo_if_available(routes: list[dict[str, Any]]) -> dict[str, Any]:
    """Run a lightweight SUMO feasibility check when local SUMO assets exist.

    Full microscopic simulation requires a SUMO `.net.xml` plus demand/routes.
    This hook deliberately reports capability and does not fail the pipeline when
    SUMO is not installed.
    """
    capability = sumo_capability_report()
    if not capability["available"]:
        capability["ran"] = False
        capability["reason"] = "SUMO binaries or SUMO network file are not available."
        return capability
    try:
        completed = subprocess.run(
            [capability["binaries"]["sumo"], "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        capability.update(
            {
                "ran": completed.returncode == 0,
                "version": (completed.stdout or completed.stderr).strip().splitlines()[0] if (completed.stdout or completed.stderr) else "",
                "candidate_routes": len(routes),
            }
        )
    except Exception as exc:
        capability.update({"ran": False, "reason": str(exc)})
    return capability

#!/usr/bin/env python3
"""Minimal GPU-free probe for a remote openpi/smolvla/gr00t websocket policy.

Connects to a (tunnelled) policy server, prints its metadata, then sends ONE
dummy observation built from the server's expected_obs_keys and prints the FULL
response or the FULL server-side error (the openpi client surfaces the server
traceback as a RuntimeError when the server replies with a string frame).

Usage:
  python3 scripts/level23_eval/probe_server.py --port 18083 --prompt "pick up the beaker"
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "packages" / "openpi-client" / "src"))

from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--prompt", default="pick up the beaker")
    ap.add_argument("--state-dim", type=int, default=8)
    ap.add_argument("--img", type=int, default=256)
    args = ap.parse_args()

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    meta = client.get_server_metadata()
    print("=== SERVER METADATA ===")
    for k, v in (meta or {}).items():
        print(f"  {k}: {v}")

    keys = (meta or {}).get("expected_obs_keys") or [
        "observation/image",
        "observation/image_2",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    ]
    print(f"\n=== building dummy obs for keys: {keys} ===")
    img = np.zeros((args.img, args.img, 3), dtype=np.uint8)
    obs = {}
    for k in keys:
        if k == "prompt":
            obs[k] = args.prompt
        elif "state" in k:
            obs[k] = np.zeros((args.state_dim,), dtype=np.float32)
        elif "image" in k:
            obs[k] = img.copy()
        else:
            obs[k] = img.copy()
    # Always include a prompt even if not advertised — many policies require it.
    obs.setdefault("prompt", args.prompt)
    for k, v in obs.items():
        shp = getattr(v, "shape", f"str:{v!r}")
        dt = getattr(v, "dtype", "")
        print(f"  {k}: {shp} {dt}")

    print("\n=== infer() ===")
    try:
        result = client.infer(obs)
        print("SUCCESS. result keys:", list(result.keys()))
        for k, v in result.items():
            shp = getattr(v, "shape", v)
            print(f"  {k}: {shp} {getattr(v, 'dtype', '')}")
    except Exception as e:
        print("INFER FAILED — full error below:")
        print(repr(e))
        print("-" * 60)
        print(str(e))
        print("-" * 60)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

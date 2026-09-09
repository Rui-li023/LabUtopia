"""Run the original main loop with a read-only, per-step pipette pose sidecar.

python -m experiments.pipette.observe_pipette_grip --output /new/path/grip.jsonl -- \
    --config-dir outputs/example --config-name attempt --headless --no-video

No source file, action, state, scene, or success criterion is changed. The
wrapper returns the exact original step result. An extra diagnostic replay
never adds an independent sample. The paired evaluator may use this entry
for predeclared collect/replay attempts. The output must not exist.
"""

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("main_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    main_args = args.main_args[1:] if args.main_args[:1] == ["--"] else args.main_args
    if not main_args:
        parser.error("Provide main.py arguments after --")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        os.environ["LABUTOPIA_PIPETTE_GRASP_DEBUG"] = "1"
        root = Path(__file__).resolve().parents[2]
        sys.argv = [str(root / "experiments/main.py"), *main_args]
        # main creates SimulationApp before importing Isaac-dependent classes.
        # Delay these imports until that mandatory initialization has happened.
        simulation = importlib.import_module("experiments.main")
        module = importlib.import_module("controllers.pipette_rack_controller")
        telemetry = importlib.import_module("experiments.support.grasp_telemetry")
        controller_type = module.PipetteRackTaskController
        original = controller_type.step
        own_step = controller_type.__dict__.get("step")
        original_close = simulation.simulation_app.close
        last_step_terminal = False
        sources = [root / "experiments/main.py", Path(module.__file__), Path(telemetry.__file__), Path(__file__)]
        stream.write(
            json.dumps(
                {
                    "type": "metadata",
                    "argv": main_args,
                    "read_only_observer": True,
                    "source_sha256": {
                        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
                    },
                }
            )
            + "\n"
        )

        def emit(sample: dict) -> None:
            nonlocal last_step_terminal
            last_step_terminal = sample["done"]
            stream.write(json.dumps(sample, allow_nan=False) + "\n")
            if sample["done"]:
                stream.flush()

        def close_with_trace(*close_args, **close_kwargs):
            # SimulationApp.close may exit without returning to this launcher.
            stream.write(json.dumps({"type": "shutdown_requested", "last_step_terminal": last_step_terminal}) + "\n")
            stream.flush()
            return original_close(*close_args, **close_kwargs)

        controller_type.step = telemetry.observed_step(original, emit)
        simulation.simulation_app.close = close_with_trace
        try:
            simulation.main()
            stream.write(json.dumps({"type": "completed"}) + "\n")
        finally:
            if own_step is None:
                del controller_type.step
            else:
                controller_type.step = own_step
            simulation.simulation_app.close = original_close


if __name__ == "__main__":
    main()

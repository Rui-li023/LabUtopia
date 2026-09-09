"""Experimental acrylic-only material setup using the unchanged main loop.

Unlike observe_pipette_grip, this entry point DOES change scene materials:
it applies cfg.task.rack_contact to the four acrylic rack pieces, never the
silicone feet. It also saves read-only physical grasp telemetry. This is a
development probe, not a new independent paired-acceptance sample.

python -m experiments.pipette.probe_pipette_rack_contact --output /new/grip.jsonl -- \
    --config-dir outputs/probe_config --config-name attempt --headless
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
    root = Path(__file__).resolve().parents[2]
    os.environ["LABUTOPIA_PIPETTE_GRASP_DEBUG"] = "1"
    sys.argv = [str(root / "experiments/main.py"), *main_args]
    with args.output.open("x", encoding="utf-8") as stream:
        # Isaac/pxr-dependent imports require main's SimulationApp startup.
        simulation = importlib.import_module("experiments.main")
        task_type = importlib.import_module("tasks.pipette_rack_task").PipetteRackTask
        controller_type = importlib.import_module("controllers.pipette_rack_controller").PipetteRackTaskController
        materials = importlib.import_module("utils.pipette_rack_material")
        telemetry = importlib.import_module("experiments.support.grasp_telemetry")
        original_setup, original_step = task_type.setup_objects, controller_type.step
        own_step = controller_type.__dict__.get("step")
        original_close = simulation.simulation_app.close
        last_terminal = False
        sources = [Path(__file__), Path(materials.__file__), Path(telemetry.__file__), root / "experiments/main.py"]
        stream.write(
            json.dumps(
                {
                    "type": "metadata",
                    "argv": main_args,
                    "experimental_scene_change": "cfg.task.rack_contact applied to acrylic only",
                    "source_sha256": {
                        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
                    },
                }
            )
            + "\n"
        )

        def setup(task):
            settings = task.cfg.task.get("rack_contact")
            if settings is None or task.cfg.task.obj_paths[1].get("friction") is not None:
                raise ValueError("Probe requires rack_contact and forbids a whole-rack friction override")
            original_setup(task)
            paths = materials.configure_acrylic_rack_contact(
                task.stage,
                str(task.cfg.task.obj_paths[1].path),
                static_friction=float(settings.static_friction),
                dynamic_friction=float(settings.dynamic_friction),
            )
            stream.write(
                json.dumps(
                    {
                        "type": "material_setup",
                        "paths": paths,
                        "static_friction": float(settings.static_friction),
                        "dynamic_friction": float(settings.dynamic_friction),
                    }
                )
                + "\n"
            )
            stream.flush()

        def emit(sample):
            nonlocal last_terminal
            last_terminal = sample["done"]
            stream.write(json.dumps(sample, allow_nan=False) + "\n")
            if last_terminal:
                stream.flush()

        def close(*close_args, **close_kwargs):
            stream.write(json.dumps({"type": "shutdown_requested", "last_step_terminal": last_terminal}) + "\n")
            stream.flush()
            return original_close(*close_args, **close_kwargs)

        task_type.setup_objects = setup
        controller_type.step = telemetry.observed_step(original_step, emit)
        simulation.simulation_app.close = close
        try:
            simulation.main()
        finally:
            task_type.setup_objects = original_setup
            if own_step is None:
                del controller_type.step
            else:
                controller_type.step = own_step
            simulation.simulation_app.close = original_close


if __name__ == "__main__":
    main()

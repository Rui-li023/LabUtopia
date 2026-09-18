"""Three or four independent pipettes and one rack; observations only."""

from typing import Any

from pxr import Sdf, Usd

from .pipette_rack_task import PipetteRackTask


class PipetteRackSequenceTask(PipetteRackTask):
    def setup_objects(self) -> None:
        self.pipette_paths = list(self.cfg.task.pipette_paths)
        self.slot_indices = list(self.cfg.task.slot_indices)
        source = Usd.Stage.Open(str(self.cfg.usd_path))
        count = source.GetPrimAtPath("/World/pipette_rack").GetAttribute("reference:slotCount").Get()
        slot_count = 6 if count is None else int(count)
        tool_count = len(self.pipette_paths)
        if tool_count not in (3, 4) or len(set(self.pipette_paths)) != tool_count:
            raise ValueError("Level 4 requires three or four distinct pipette rigid bodies")
        if (
            len(self.slot_indices) != tool_count
            or len(set(self.slot_indices)) != tool_count
            or any(type(i) is not int or i not in range(slot_count) for i in self.slot_indices)
        ):
            raise ValueError(f"Level 4 requires {tool_count} distinct slots in 0..{slot_count - 1}")
        for path in self.pipette_paths[1:]:
            if not Sdf.CopySpec(
                source.GetRootLayer(), Sdf.Path(self.pipette_paths[0]), self.stage.GetRootLayer(), Sdf.Path(path)
            ):
                raise RuntimeError(f"Could not clone the complete pipette mesh: {path}")
            self.stage.GetRootLayer().GetPrimAtPath(path).specifier = Sdf.SpecifierDef
            # The source root is an over whose Xform type comes from the base
            # lab layer. A copied over has no inherited type at its new path.
            self.stage.GetRootLayer().GetPrimAtPath(path).typeName = "Xform"
        super().setup_objects()

    def reset(self) -> None:
        super().reset()
        configs = {item["path"]: item for item in self.cfg.task.obj_paths}
        for path in self.pipette_paths[1:]:
            self._randomize_with_geometry_correction(path, configs[path]["position_range"])
        self._episode_init_state["extra"]["pipette_slots"] = dict(zip(self.pipette_paths, self.slot_indices))

    def reset_with_init_state(self, init_state: dict) -> None:
        saved = init_state.get("extra", {}).get("pipette_slots", {})
        if saved != dict(zip(self.pipette_paths, self.slot_indices)):
            raise ValueError("Replay pipette/slot assignment differs from the recorded episode")
        # Reject four-tool data in a three-tool scene BEFORE any state writes.
        super().reset_with_init_state(init_state)

    def step(self) -> dict[str, Any] | None:
        state = super().step()
        if state is None:
            return None
        pipettes = []
        for path, slot in zip(self.pipette_paths, self.slot_indices):
            seat = self.target_obj + f"/Seat{slot}"
            pipettes.append(
                {
                    "object_path": path,
                    "object_name": path.rsplit("/", 1)[-1],
                    "object_position": self.object_utils.get_geometry_center(object_path=path),
                    "object_size": self.object_utils.get_object_size(object_path=path),
                    "object_reference_position": self._marker_position(path + "/Reference"),
                    "object_grasp_position": self._grasp_position(path),
                    "object_axis": self._marker_position(path + "/AxisHead")
                    - self._marker_position(path + "/AxisBase"),
                    "object_quaternion": self._world_quaternion(path),
                    "target_seat_position": self._marker_position(seat),
                    "target_quaternion": self._world_quaternion(seat),
                    "slot_index": slot,
                }
            )
        state["pipettes"] = pipettes
        return state

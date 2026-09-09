"""Keep explicitly substepped, multi-scene USD stages on one physics clock."""

from pxr import PhysxSchema, Usd, UsdPhysics


def synchronize_physics_scene_steps(stage: Usd.Stage, steps_per_second: int) -> list[dict]:
    """Set scene cadence BEFORE simulation starts; never change body ownership.

    A referenced lab may contain its own PhysicsScene in addition to the one
    created by World. Updating only World's scene leaves mixed-rate PhysX
    scenes and an ambiguous global physics-step callback. The action/render
    clock remains the caller's responsibility.
    """
    if isinstance(steps_per_second, bool) or not isinstance(steps_per_second, int) or not 60 <= steps_per_second <= 480:
        raise ValueError("Physics cadence must be an integer in [60, 480] Hz")
    scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if not scenes:
        raise ValueError("Cannot synchronize a stage without a PhysicsScene")
    records = []
    for prim in scenes:
        api = PhysxSchema.PhysxSceneAPI.Apply(prim)
        before = api.GetTimeStepsPerSecondAttr().Get()
        api.CreateTimeStepsPerSecondAttr(steps_per_second)
        records.append({"path": str(prim.GetPath()), "previous_hz": before, "configured_hz": steps_per_second})
    return records

from .dual_object_task import DualObjectTask


class PickPlaceTask(DualObjectTask):
    """Pick-and-place task: pick source object then place on target platform.

    Fully inherits dual-object behaviour (randomised positions, standard step
    state with ``object_position`` / ``target_position``).  The controller
    accesses these fields directly from the state dict.
    """

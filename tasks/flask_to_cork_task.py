from .dual_object_task import DualObjectTask


class FlaskToCorkTask(DualObjectTask):
    """Level-2 pick-and-place: grasp a round-bottom flask and seat it in a cork ring.

    Inherits all dual-object behaviour from DualObjectTask (randomised positions,
    standard state with ``object_position`` / ``target_position``).
    """

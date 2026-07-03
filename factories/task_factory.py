from typing import Dict, Type
from tasks.base_task import BaseTask
from tasks.open_close_task import OpenCloseTask
from tasks.pick_task import PickTask
from tasks.place_task import PlaceTask
from tasks.press_task import PressTask
from tasks.shake_task import ShakeTask
from tasks.stir_task import StirTask
from tasks.pick_pour_task import PickPourTask
from tasks.pick_place_task import PickPlaceTask
from tasks.place_press_task import PlacePressTask
from tasks.clean_beaker_task import CleanBeakerTask
from tasks.device_operate_task import DeviceOperateTask
from tasks.open_transport_pour_task import OpenTransportPourTask
from tasks.liquid_mixing_task import LiquidMixingTask
from tasks.navigation_task import NavigationTask
from tasks.mobile_pick_task import MobilePickTask
from tasks.mobile_transport_place_task import MobileTransportPlaceTask

_task_registry: Dict[str, Type[BaseTask]] = {}

def register_task(name: str, task_class: Type[BaseTask]):

    _task_registry[name] = task_class

def create_task(task_name: str, *args, **kwargs) -> BaseTask:

    if task_name not in _task_registry:
        raise ValueError(f"Unknown task type: '{task_name}'. Available: {list(_task_registry.keys())}")
    return _task_registry[task_name](*args, **kwargs)


register_task("pick", PickTask)
register_task("place", PlaceTask)
register_task("press", PressTask)
register_task("shake", ShakeTask)
register_task("stir", StirTask)
register_task("open_close", OpenCloseTask)
register_task("device_operate", DeviceOperateTask)
register_task("pick_pour", PickPourTask)
register_task("pick_place", PickPlaceTask)
register_task("place_press", PlacePressTask)
register_task("clean_beaker", CleanBeakerTask)
register_task("open_transport_pour", OpenTransportPourTask)
register_task("liquid_mixing", LiquidMixingTask)
register_task("navigation", NavigationTask)
register_task("mobile_pick", MobilePickTask)
register_task("mobile_transport_place", MobileTransportPlaceTask)

from tasks.flask_to_cork_task import FlaskToCorkTask
register_task("flask_to_cork", FlaskToCorkTask)

from tasks.stopper_flask_task import StopperFlaskTask
register_task("stopper_flask", StopperFlaskTask)

from tasks.pipette_rack_task import PipetteRackTask
register_task("pipette_rack", PipetteRackTask)

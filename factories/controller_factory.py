from typing import Dict, Type
from controllers.base_controller import BaseController
from controllers.open_controller import OpenTaskController
from controllers.pick_pour_controller import PickPourTaskController
from controllers.place_press_controller import PlacePressTaskController
from controllers.pick_controller import PickTaskController
from controllers.pick_wide_controller import PickWideTaskController
from controllers.pour_controller import PourTaskController
from controllers.place_controller import PlaceTaskController
from controllers.press_controller import PressTaskController
from controllers.shake_controller import ShakeTaskController
from controllers.stir_controller import StirTaskController
from controllers.stir_glassrod_controller import StirGlassrodTaskController
from controllers.pick_place_controller import PickPlaceTaskController
from controllers.shake_beaker_controller import ShakeBeakerTaskController
from controllers.clean_beaker_controller import CleanBeakerTaskController
from controllers.device_operate_controller import DeviceOperateController
from controllers.open_transport_pour_controller import OpenTransportPourController
from controllers.liquid_mixing_controller import LiquidMixingController
from controllers.close_controller import CloseTaskController
from controllers.open_close_controller import OpenCloseTaskController
from controllers.navigation_controller import NavigationController
from controllers.mobile_pick_controller import MobilePickController
from controllers.mobile_pour_controller import MobilePourController
from controllers.mobile_shake_controller import MobileShakeController
from controllers.mobile_transport_place_controller import MobileTransportPlaceController

_controller_registry: Dict[str, Type[BaseController]] = {}

def register_controller(name: str, controller_class: Type[BaseController]):
    _controller_registry[name] = controller_class

def create_controller(controller_name: str, *args, **kwargs) -> BaseController:
    if controller_name not in _controller_registry:
        raise ValueError(f"Unknown controller type: '{controller_name}'. Available: {list(_controller_registry.keys())}")
    return _controller_registry[controller_name](*args, **kwargs)

register_controller("pick_pour", PickPourTaskController)
register_controller("open", OpenTaskController)
register_controller("close", CloseTaskController)
register_controller("open_close", OpenCloseTaskController)
register_controller("pick", PickTaskController)
register_controller("pick_wide", PickWideTaskController)
register_controller("pour", PourTaskController)
register_controller("place", PlaceTaskController)
register_controller("pick_place", PickPlaceTaskController)
register_controller("place_press", PlacePressTaskController)
register_controller("press", PressTaskController)
register_controller("shake", ShakeTaskController)
register_controller("stir", StirTaskController)
register_controller("stir_glassrod", StirGlassrodTaskController)
register_controller("shake_beaker", ShakeBeakerTaskController)
register_controller("clean_beaker", CleanBeakerTaskController)
register_controller("device_operate", DeviceOperateController)
register_controller("open_transport_pour", OpenTransportPourController)
register_controller("liquid_mixing", LiquidMixingController)
register_controller("navigation", NavigationController)
register_controller("mobile_pick", MobilePickController)
register_controller("mobile_pour", MobilePourController)
register_controller("mobile_shake", MobileShakeController)
register_controller("mobile_transport_place", MobileTransportPlaceController)

from controllers.flask_to_cork_controller import FlaskToCorkTaskController
register_controller("flask_to_cork", FlaskToCorkTaskController)

from controllers.stopper_flask_controller import StopperFlaskTaskController
register_controller("stopper_flask", StopperFlaskTaskController)

from controllers.pipette_rack_controller import PipetteRackTaskController
register_controller("pipette_rack", PipetteRackTaskController)

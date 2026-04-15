from typing import Dict, Type
from controllers.base_controller import BaseController
from controllers.open_controller import OpenTaskController
from controllers.pickpour_controller import PickPourTaskController
from controllers.placepress_controller import PlacePressTaskController
from controllers.pick_controller import PickTaskController
from controllers.pour_controller import PourTaskController
from controllers.place_controller import PlaceTaskController
from controllers.press_controller import PressTaskController
from controllers.shake_controller import ShakeTaskController
from controllers.stir_controller import StirTaskController
from controllers.stirglassrod_controller import StirGlassrodTaskController
from controllers.pickplace_controller import PickPlaceTaskController
from controllers.shakebeaker_controller import ShakeBeakerTaskController
from controllers.cleanbeaker_controller import CleanBeakerTaskController
from controllers.cleanbeaker7policy_controller import CleanBeaker7PolicyTaskController
from controllers.device_operate_controller import DeviceOperateController
from controllers.opentransportpour_controller import OpenTransportPourController
from controllers.liquid_mixing_controller import LiquidMixingController
from controllers.close_controller import CloseTaskController
from controllers.openclose_controller import OpenCloseTaskController
from controllers.navigation_controller import NavigationController
from controllers.mobile_pick_controller import MobilePickController

_controller_registry: Dict[str, Type[BaseController]] = {}

def register_controller(name: str, controller_class: Type[BaseController]):
    _controller_registry[name] = controller_class

def create_controller(controller_name: str, *args, **kwargs) -> BaseController:
    if controller_name not in _controller_registry:
        raise ValueError(f"Unknown controller type: '{controller_name}'. Available: {list(_controller_registry.keys())}")
    return _controller_registry[controller_name](*args, **kwargs)

register_controller("pickpour", PickPourTaskController)
register_controller("open", OpenTaskController)
register_controller("close", CloseTaskController)
register_controller("openclose", OpenCloseTaskController)
register_controller("pick", PickTaskController)
register_controller("pour", PourTaskController)
register_controller("place", PlaceTaskController)
register_controller("pickplace", PickPlaceTaskController)
register_controller("placepress", PlacePressTaskController)
register_controller("press", PressTaskController)
register_controller("shake", ShakeTaskController)
register_controller("stir", StirTaskController)
register_controller("stirglassrod", StirGlassrodTaskController)
register_controller("shakebeaker", ShakeBeakerTaskController)
register_controller("cleanbeaker", CleanBeakerTaskController)
register_controller("cleanbeaker7policy", CleanBeaker7PolicyTaskController)
register_controller("device_operate", DeviceOperateController)
register_controller("OpenTransportPour", OpenTransportPourController)
register_controller("LiquidMixing", LiquidMixingController)
register_controller("navigation", NavigationController)
register_controller("mobile_pick", MobilePickController)

from controllers.flask_to_cork_controller import FlaskToCorkTaskController
register_controller("flask_to_cork", FlaskToCorkTaskController)

from controllers.stopper_flask_controller import StopperFlaskTaskController
register_controller("stopper_flask", StopperFlaskTaskController)

from controllers.pipette_rack_controller import PipetteRackTaskController
register_controller("pipette_rack", PipetteRackTaskController)
